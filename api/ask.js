const { pool } = require('../utils/db');
const { sanitizeQuestion } = require('../utils/sanitize');
const { findInSemanticCache, saveToSemanticCache } = require('../utils/semanticCache');
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { ChatAnthropic } = require("@langchain/anthropic");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "rag-do-an";
const MODEL_NAME = "claude-3-5-haiku-20241022";
const MAX_QUESTION_LENGTH = 500;
const SIMILARITY_THRESHOLD = 0.55;
const CHAT_HISTORY_LIMIT = 3;

function classifyScoreType(question) {
    const lowerQ = question.toLowerCase();

    const trainingKeywords = ['rèn luyện', 'ren luyen', 'đánh giá rèn luyện', 'đanh gia ren luyen'];
    if (trainingKeywords.some(k => lowerQ.includes(k))) {
        return 'TRAINING';
    }

    const gpaKeywords = ['gpa', 'điểm trung bình', 'diem trung binh', 'điểm tb', 'diem tb', 'điểm chữ', 'diem chu'];
    if (gpaKeywords.some(k => lowerQ.includes(k))) {
        return 'GPA';
    }

    const scoreMatch = lowerQ.match(/(\d+(?:\.\d+)?)\s*(?:điểm|diem)/);
    if (scoreMatch) {
        const score = parseFloat(scoreMatch[1]);

        if (score > 10) return 'TRAINING';

        if (score <= 4 && gpaKeywords.some(k => lowerQ.includes(k))) {
            return 'GPA';
        }

        const examKeywords = ['thi', 'kiểm tra', 'kiem tra', 'bài thi', 'bai thi', 'môn học', 'mon hoc', 'môn', 'mon'];
        if (score <= 10 && examKeywords.some(k => lowerQ.includes(k))) {
            return 'EXAM';
        }

        if (score <= 10) return 'EXAM';
    }

    if (lowerQ.includes('thi') || lowerQ.includes('kiểm tra') || lowerQ.includes('kiem tra')) {
        return 'EXAM';
    }

    return 'GENERAL';
}

async function getChatHistory(chatId, limit = CHAT_HISTORY_LIMIT) {
    if (!chatId) return "";

    try {
        const res = await pool.query(
            `SELECT role, content FROM messages 
             WHERE chat_id = $1 
             ORDER BY created_at DESC 
             LIMIT $2`,
            [chatId, limit]
        );

        if (res.rows.length === 0) return "";

        return res.rows.reverse().map(msg => {
            return `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`;
        }).join('\n');
    } catch (error) {
        return "";
    }
}

async function rewriteQuestion(rawQuestion, history, apiKey) {
    try {
        const scoreType = classifyScoreType(rawQuestion);

        const rewriteModel = new ChatAnthropic({
            modelName: "claude-3-haiku-20240307",
            apiKey: apiKey,
            temperature: 0,
            maxTokens: 200
        });

        const prompt = `Bạn là chuyên gia quy chế TNUT. Viết lại câu hỏi để tìm kiếm tài liệu.

Lịch sử: ${history || "Không có"}
Câu hỏi: "${rawQuestion}"
Loại: ${scoreType}

QUY TẮC:
- ${scoreType === 'EXAM' ? 'Điểm thi môn học (0-10): thêm "điểm thi", "tín chỉ", "kết quả học tập"' : ''}
- ${scoreType === 'TRAINING' ? 'Điểm rèn luyện (50-100): thêm "xếp loại rèn luyện"' : ''}
- ${scoreType === 'GPA' ? 'GPA (0-4): thêm "điểm trung bình", "học bổng"' : ''}
- ${scoreType === 'GENERAL' ? 'Làm rõ ngữ cảnh, giữ thuật ngữ chuyên ngành' : ''}

VÍ DỤ:
"4 điểm tích gì" → "điểm thi 4.0 thang 10 được tích tín chỉ kết quả học tập như thế nào"
"90 điểm rèn luyện" → "xếp loại đánh giá rèn luyện 90 điểm thang 100"

CHỈ GHI CÂU VIẾT LẠI:`;

        const result = await rewriteModel.invoke(prompt);
        return result.content ? result.content.trim() : result.toString().trim();

    } catch (error) {
        return rawQuestion;
    }
}

async function expandQuery(originalQuery, apiKey) {
    try {
        const scoreType = classifyScoreType(originalQuery);

        const expansionModel = new ChatAnthropic({
            modelName: "claude-3-haiku-20240307",
            apiKey: apiKey,
            temperature: 0.2,
            maxTokens: 150
        });

        const prompt = `Tạo 1 biến thể tìm kiếm.

Câu gốc: "${originalQuery}"
Loại: ${scoreType}

Biến thể phải:
- Giữ số điểm và loại điểm
- ${scoreType === 'EXAM' ? 'Dùng: "quy đổi tín chỉ", "điểm số học phần", "kết quả môn học"' : ''}
- ${scoreType === 'TRAINING' ? 'Dùng: "xếp loại sinh viên", "đánh giá thái độ học tập"' : ''}
- ${scoreType === 'GPA' ? 'Dùng: "điểm tích lũy", "xếp hạng học lực"' : ''}
- Khác góc độ nhưng cùng ý nghĩa

CHỈ 1 DÒNG:`;

        const result = await expansionModel.invoke(prompt);
        const content = result.content ? result.content.trim() : result.toString().trim();
        const variant = content.split('\n')[0].trim();

        return [originalQuery, variant];

    } catch (error) {
        return [originalQuery];
    }
}

async function hybridSearch(queries, embeddings, pinecone, indexName) {
    const index = pinecone.Index(indexName);
    const allResults = new Map();

    for (const query of queries) {
        try {
            const queryVector = await embeddings.embedQuery(query);
            const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
                pineconeIndex: index
            });

            const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 5);

            for (const [doc, score] of results) {
                const key = doc.pageContent.substring(0, 100).trim();

                if (allResults.has(key)) {
                    const existing = allResults.get(key);
                    if (score > existing[1]) {
                        allResults.set(key, [doc, score]);
                    }
                } else {
                    allResults.set(key, [doc, score]);
                }
            }
        } catch (error) {
        }
    }

    return Array.from(allResults.values())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
}

function rerankChunks(results, question) {
    const questionLower = question.toLowerCase();
    const questionWords = questionLower
        .split(/\s+/)
        .filter(word => word.length > 2);

    const rankedResults = results.map(([doc, score]) => {
        let relevanceScore = score;
        const content = doc.pageContent.toLowerCase();

        let keywordMatchCount = 0;
        questionWords.forEach(word => {
            if (content.includes(word)) {
                keywordMatchCount++;
            }
        });
        relevanceScore += keywordMatchCount * 0.05;

        if (doc.metadata.title) {
            relevanceScore += 0.1;
        }

        if (doc.pageContent.length > 500) {
            relevanceScore += 0.05;
        }

        if (/\d{10,11}|@/.test(content)) {
            relevanceScore += 0.08;
        }

        return {
            doc,
            originalScore: score,
            relevanceScore,
            keywordMatchCount
        };
    });

    rankedResults.sort((a, b) => b.relevanceScore - a.relevanceScore);

    return rankedResults;
}

function buildContext(rankedResults, topK = 5) {
    const topChunks = rankedResults.slice(0, topK);
    const uniqueChunks = [];

    const seenContent = new Set();
    for (const chunk of topChunks) {
        const signature = chunk.doc.pageContent.substring(0, 100).trim();
        if (!seenContent.has(signature)) {
            uniqueChunks.push(chunk);
            seenContent.add(signature);
        }
    }

    return uniqueChunks
        .map((chunk, index) => {
            let section = `[Tài liệu ${index + 1}]`;
            if (chunk.doc.metadata.title) {
                section += ` ${chunk.doc.metadata.title}`;
            }
            section += `\n${chunk.doc.pageContent}`;
            return section;
        })
        .join('\n\n---\n\n');
}

async function ensureChatId(chatId, userId, question) {
    if (chatId) return chatId;

    try {
        const title = question.length > 50
            ? question.substring(0, 47) + "..."
            : question;

        const result = await pool.query(
            `INSERT INTO chats (user_id, title) VALUES ($1, $2) RETURNING id`,
            [userId, title]
        );

        return result.rows[0].id;
    } catch (error) {
        throw error;
    }
}

async function saveMessage(chatId, role, content, sources = null) {
    try {
        await pool.query(
            `INSERT INTO messages (chat_id, role, content, sources) 
             VALUES ($1, $2, $3, $4)`,
            [chatId, role, content, sources ? JSON.stringify(sources) : null]
        );
    } catch (error) {
        throw error;
    }
}

module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method not allowed" });
    }

    try {
        let { question, userId = 1, chatId } = req.body;

        if (!userId || userId < 1) {
            return res.status(400).json({ error: "userId không hợp lệ" });
        }

        const sanitizeResult = sanitizeQuestion(question);
        const hasXSS = sanitizeResult.hasXSS;
        question = sanitizeResult.sanitized;

        if (!question) {
            return res.status(400).json({ error: "Câu hỏi không hợp lệ" });
        }

        if (question.length > MAX_QUESTION_LENGTH) {
            question = question.substring(0, MAX_QUESTION_LENGTH);
        }

        if (hasXSS) {
            const xssWarningAnswer = `⚠️ **Cảnh báo bảo mật**

Tôi phát hiện câu hỏi của bạn chứa các ký tự đặc biệt có thể gây rủi ro bảo mật (XSS - Cross-Site Scripting).

**Điều này có nghĩa là:**
- Câu hỏi chứa mã HTML/JavaScript nguy hiểm như \`<script>\`, \`onerror=\`, \`javascript:\`...
- Những ký tự này có thể được sử dụng để tấn công hệ thống
- Tôi đã tự động loại bỏ các ký tự nguy hiểm này

**Khuyến nghị:**
- Vui lòng đặt câu hỏi bằng ngôn ngữ tự nhiên bình thường
- Không cần dùng các ký tự đặc biệt như <, >, {, }, \\
- Nếu bạn có ý định tốt, hãy diễn đạt lại câu hỏi

Nếu bạn cần hỗ trợ về quy chế đào tạo, học vụ của TNUT, tôi luôn sẵn sàng giúp bạn! 😊`;

            chatId = await ensureChatId(chatId, userId, question);
            await saveMessage(chatId, 'user', question);
            await saveMessage(chatId, 'assistant', xssWarningAnswer, { warning: "XSS_DETECTED" });

            return res.status(200).json({
                answer: xssWarningAnswer,
                chatId,
                warning: true,
                cached: false
            });
        }

        let chatHistory = "";
        if (chatId) {
            chatHistory = await getChatHistory(chatId);
        }

        const refinedQuestion = await rewriteQuestion(
            question,
            chatHistory,
            process.env.ANTHROPIC_API_KEY
        );

        const queries = await expandQuery(refinedQuestion, process.env.ANTHROPIC_API_KEY);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: process.env.GEMINI_API_KEY,
        });

        const queryVector = await embeddings.embedQuery(refinedQuestion);

        let cachedAnswer = null;
        try {
            if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
                cachedAnswer = await findInSemanticCache(refinedQuestion, queryVector);
            }
        } catch (cacheError) {
        }

        if (cachedAnswer) {
            chatId = await ensureChatId(chatId, userId, question);
            await saveMessage(chatId, 'user', question);
            await saveMessage(chatId, 'assistant', cachedAnswer, { source: "cache" });

            return res.status(200).json({
                answer: cachedAnswer,
                chatId,
                cached: true
            });
        }

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

        const results = await hybridSearch(queries, embeddings, pinecone, PINECONE_INDEX_NAME);
        const relevantDocs = results.filter(r => r[1] > SIMILARITY_THRESHOLD);

        let context = "";
        let sources = [];

        if (relevantDocs.length > 0) {
            const rankedResults = rerankChunks(relevantDocs, refinedQuestion);
            context = buildContext(rankedResults, 5);
            sources = rankedResults.slice(0, 5).map(item => item.doc.metadata);
        } else {
            context = "Không tìm thấy thông tin cụ thể trong tài liệu.";
        }

        const scoreType = classifyScoreType(question);

        const template = `Bạn là trợ lý AI chuyên nghiệp hỗ trợ sinh viên Trường Đại học Kỹ thuật Công nghiệp - Đại học Thái Nguyên (TNUT).

<loại_câu_hỏi>${scoreType}</loại_câu_hỏi>

<history>
{chat_history}
</history>

<context>
{context}
</context>

Câu hỏi: "{question}"
Ý định: "{refined_question}"

**NGUYÊN TẮC TRẢ LỜI:**

1. **ƯU TIÊN TUYỆT ĐỐI: SỬ DỤNG THÔNG TIN TỪ CONTEXT**
   - LUÔN phân tích kỹ context trước khi trả lời
   - Trích xuất TOÀN BỘ thông tin liên quan (số liệu, bảng, danh sách)
   - Nếu context có bảng điểm chi tiết → PHẢI liệt kê đầy đủ
   - Nếu context có quy định cụ thể → PHẢI trích dẫn chính xác

2. **PHÂN BIỆT LOẠI ĐIỂM (CHỈ KHI CẦN THIẾT):**

${scoreType === 'EXAM' ? `
   ✓ Câu hỏi về ĐIỂM THI MÔN HỌC (0-10):
   - Nếu context có bảng quy đổi điểm số/chữ → Trích đầy đủ
   - Trả lời: kết quả (đạt/không đạt), tín chỉ được tích, ảnh hưởng GPA
   - KHÔNG nhắc điểm rèn luyện trừ khi context có liên kết rõ ràng
` : ''}

${scoreType === 'TRAINING' ? `
   ✓ Câu hỏi về ĐIỂM RÈN LUYỆN (50-100):
   - Nếu context có bảng xếp loại → Trích đầy đủ
   - Trả lời: xếp loại, ý nghĩa của loại đó
   - KHÔNG nhắc điểm thi môn học
` : ''}

${scoreType === 'GPA' ? `
   ✓ Câu hỏi về GPA/ĐIỂM TRUNG BÌNH (0-4):
   - Trả lời: điều kiện học bổng, tốt nghiệp, xếp hạng
   - KHÔNG nhắc điểm rèn luyện
` : ''}

3. **PHONG CÁCH:**
   - BẮT ĐẦU: "TNUT quy định..." hoặc "Theo quy chế..."
   - KHÔNG nói: "Dựa trên context", "Theo tài liệu", "Thông tin cho thấy"
   - In đậm số liệu quan trọng (điểm số, số tiền, hạn chót)
   - Nếu context có bảng/danh sách chi tiết → PHẢI liệt kê đầy đủ

4. **ĐỘ DÀI & CHI TIẾT:**
   - Trả lời ngắn gọn nhưng ĐẦY ĐỦ thông tin từ context
   - Nếu context có 5 mục → liệt kê cả 5 mục
   - Nếu context có bảng điểm → copy toàn bộ bảng
   - Ưu tiên độ chính xác hơn độ ngắn gọn

5. **LIÊN HỆ:**
   - Nếu context có: tên, chức vụ, SĐT, email → Ghi đầy đủ
   - Nếu không: "Liên hệ Phòng Đào tạo để biết thêm chi tiết"

**VÍ DỤ TRẢ LỜI TỐT:**

Context: "Bảng 1. Xếp loại điểm: 
- Từ 9.0 đến 10: tương đương 4.0 điểm số học A+ 
- Từ 8.5 đến 8.9: tương đương 3.7 điểm số học A
- ..."

Câu hỏi: "4 điểm được tích gì"

Trả lời TỐT: "TNUT quy định điểm thi môn học theo thang 10 được quy đổi như sau:
- Từ **9.0 đến 10 điểm**: tương đương 4.0 điểm số học A+
- Từ **8.5 đến 8.9 điểm**: tương đương 3.7 điểm số học A
- Từ **8.0 đến 8.4 điểm**: tương đương 3.5 điểm số học B+
[liệt kê tiếp...]

Với **4 điểm** (thang 10), sinh viên đạt môn học và được tích đầy đủ số tín chỉ của học phần đó."

Trả lời:`;

        const model = new ChatAnthropic({
            modelName: MODEL_NAME,
            apiKey: process.env.ANTHROPIC_API_KEY,
            temperature: 0.2,
            maxTokens: 1024
        });

        const chain = PromptTemplate.fromTemplate(template)
            .pipe(model)
            .pipe(new StringOutputParser());

        const answer = await chain.invoke({
            context,
            question,
            refined_question: refinedQuestion,
            chat_history: chatHistory
        });

        chatId = await ensureChatId(chatId, userId, question);
        await saveMessage(chatId, 'user', question);
        await saveMessage(chatId, 'assistant', answer, sources);

        try {
            if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
                await saveToSemanticCache(refinedQuestion, answer, queryVector);
            }
        } catch (cacheError) {
        }

        res.status(200).json({
            answer,
            chatId,
            sources,
            cached: false
        });

    } catch (error) {
        res.status(500).json({
            error: "Lỗi hệ thống. Vui lòng thử lại sau."
        });
    }
};