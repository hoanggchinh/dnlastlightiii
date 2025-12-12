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

    const gpaKeywords = ['gpa', 'điểm trung bình', 'diem trung binh', 'điểm tb', 'diem tb'];
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

        let prompt = `Viết lại câu hỏi để tìm kiếm trong tài liệu quy chế TNUT.

Lịch sử: ${history || "Không có"}
Câu hỏi: "${rawQuestion}"

`;

        if (scoreType === 'EXAM') {
            prompt += `Đây là câu hỏi về điểm thi/điểm môn học (thang 10).
Thêm từ khóa: "điểm thi", "điểm số", "tín chỉ", "kết quả học tập", "quy đổi điểm"

VD: "5 điểm được tích gì" → "điểm thi 5.0 thang 10 quy đổi điểm chữ và tích tín chỉ"`;
        } else if (scoreType === 'TRAINING') {
            prompt += `Đây là câu hỏi về điểm rèn luyện (thang 100).
Thêm từ khóa: "điểm rèn luyện", "xếp loại rèn luyện"`;
        } else if (scoreType === 'GPA') {
            prompt += `Đây là câu hỏi về GPA/điểm trung bình.
Thêm từ khóa: "GPA", "điểm trung bình", "học bổng"`;
        } else {
            prompt += `Làm rõ ý định câu hỏi, giữ thuật ngữ chuyên ngành.`;
        }

        prompt += `\n\nCHỈ GHI CÂU VIẾT LẠI:`;

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

        let prompt = `Tạo 1 biến thể câu hỏi để tìm kiếm tốt hơn.

Câu gốc: "${originalQuery}"

`;

        if (scoreType === 'EXAM') {
            prompt += `Dùng từ khóa: "quy đổi điểm chữ", "điểm số thang 10", "kết quả môn học", "tín chỉ tích lũy"`;
        } else if (scoreType === 'TRAINING') {
            prompt += `Dùng từ khóa: "xếp loại rèn luyện", "đánh giá sinh viên"`;
        } else if (scoreType === 'GPA') {
            prompt += `Dùng từ khóa: "điểm trung bình tích lũy", "học bổng"`;
        } else {
            prompt += `Dùng từ đồng nghĩa, mở rộng ngữ cảnh`;
        }

        prompt += `\n\nCHỈ GHI 1 DÒNG:`;

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

<history>
{chat_history}
</history>

<context>
{context}
</context>

Câu hỏi: "{question}"
Ý định: "{refined_question}"

QUY TẮC TRẢ LỜI:

1. PHONG CÁCH:
   - BẮT ĐẦU trực tiếp: "TNUT quy định..." hoặc "Theo quy chế TNUT..."
   - KHÔNG nói: "Dựa trên context", "Theo tài liệu", "Dựa trên thông tin"
   - Nói như chuyên gia nắm rõ quy chế
   - In đậm số liệu quan trọng (điểm số, số tiền, hạn chót)

2. NỘI DUNG:
   - ƯU TIÊN phân tích kỹ context trước khi trả lời
   - Nếu context có bảng điểm/quy đổi → Trích xuất thông tin chi tiết
   - Trả lời ngắn gọn, đầy đủ thông tin thiết yếu
   
3. VỚI CÂU HỎI VỀ ĐIỂM SỐ:
   ${scoreType === 'EXAM' ? `
   - Đây là câu hỏi về ĐIỂM THI/MÔN HỌC (thang 10)
   - Ưu tiên trả lời: 
     + Kết quả đạt/không đạt (điểm >= 4.0 là đạt)
     + Điểm chữ tương ứng (nếu context có bảng quy đổi)
     + Số tín chỉ được tích (nếu đạt)
   - VD: "8 điểm được tích gì" → "Điểm 8.0 đạt môn, tương đương điểm chữ A/B+ (tùy bảng quy đổi), được tích đầy đủ tín chỉ môn học"
   - CHỈ nhắc điểm rèn luyện nếu context có liên kết RÕ RÀNG
   ` : ''}
   
   ${scoreType === 'TRAINING' ? `
   - Đây là câu hỏi về ĐIỂM RÈN LUYỆN (thang 100)
   - Trả lời xếp loại: Xuất sắc/Giỏi/Khá/Trung bình/Yếu/Kém
   - KHÔNG nhắc điểm thi môn học
   ` : ''}
   
   ${scoreType === 'GPA' ? `
   - Đây là câu hỏi về GPA/ĐIỂM TRUNG BÌNH (thang 4)
   - Trả lời về: học bổng, tốt nghiệp, xếp hạng
   ` : ''}

4. ĐỘ DÀI:
   - Trả lời NGẮN GỌN, đi thẳng vào vấn đề
   - 2-4 câu là đủ cho câu hỏi đơn giản
   - Chỉ liệt kê chi tiết khi cần thiết

5. LƯU Ý:
   - Nếu context có thông tin → Dùng context
   - Nếu context không rõ → Trả lời chung theo quy chế đại học
   - Luôn thêm 1 câu ngắn khuyến nghị cuối (nếu cần)

6. LIÊN HỆ:
   - Nếu context có tên, chức vụ, SĐT, email → Ghi cụ thể
   - Nếu không: "Liên hệ Phòng Đào tạo để biết thêm chi tiết"

Trả lời:`;

        const model = new ChatAnthropic({
            modelName: MODEL_NAME,
            apiKey: process.env.ANTHROPIC_API_KEY,
            temperature: 0.3,
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