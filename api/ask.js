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
        console.error('Failed to load chat history:', error.message);
        return "";
    }
}

async function rewriteQuestion(rawQuestion, history, apiKey) {
    try {
        const rewriteModel = new ChatAnthropic({
            modelName: "claude-3-haiku-20240307",
            apiKey: apiKey,
            temperature: 0,
            maxTokens: 200
        });

        const prompt = `Bạn là chuyên gia về quy chế đào tạo TNUT. Viết lại câu hỏi để tìm kiếm trong tài liệu quy chế.

Lịch sử hội thoại:
"""
${history || "Không có"}
"""

Câu hỏi: "${rawQuestion}"

THUẬT NGỮ TNUT (quan trọng):
- "điểm tích", "X điểm tích" (X >= 50) → "xếp loại rèn luyện", "điểm rèn luyện"
- Điểm 50-100 = điểm rèn luyện (thang 100)
- Điểm 1-10 = điểm thi môn học (thang 10) HOẶC điểm chữ A/B/C/D/F (thang 4)
- "rớt môn", "trượt môn", "fail" → "học lại", "không đạt môn học"
- "GPA", "điểm TB", "điểm trung bình" → "điểm trung bình tích lũy"
- "học phí", "tiền học" → "mức học phí"
- "thi lại", "kiểm tra lại" → "thi cải thiện điểm"
- "điểm A", "điểm B", "điểm C", "điểm D", "điểm F" → "thang điểm chữ 4.0"
- "3 điểm", "3.5 điểm" (nếu < 10) → "điểm thi môn học hoặc GPA"

YÊU CẦU:
1. Nếu câu hỏi thiếu ngữ cảnh, DÙNG LỊCH SỬ để bổ sung
2. Phân biệt rõ: 
   - Số 1-10: điểm thi môn học (thang 10) HOẶC GPA (thang 4)
   - Số 50-100: điểm rèn luyện (thang 100)
   - Chữ A/B/C/D/F: thang điểm chữ 4.0
3. Chuyển thuật ngữ sinh viên → thuật ngữ quy chế
4. CHỈ TRẢ VỀ CÂU VIẾT LẠI, KHÔNG GIẢI THÍCH

Câu hỏi viết lại:`;

        const result = await rewriteModel.invoke(prompt);
        const rewritten = result.content ? result.content.trim() : result.toString().trim();

        return rewritten;
    } catch (error) {
        return rawQuestion;
    }
}

async function expandQuery(originalQuery, apiKey) {
    try {
        const expansionModel = new ChatAnthropic({
            modelName: "claude-3-haiku-20240307",
            apiKey: apiKey,
            temperature: 0.2,
            maxTokens: 250
        });

        const prompt = `Bạn là chuyên gia về hệ thống quy chế đào tạo đại học TNUT. Nhiệm vụ: tạo 2 biến thể câu hỏi để TÌM KIẾM HIỆU QUẢ trong cơ sở dữ liệu vector.

CÂU HỎI GỐC: "${originalQuery}"

PHƯƠNG PHÁP TẠO BIẾN THỂ:

1. **Biến thể mở rộng ngữ cảnh** - Thêm từ khóa quan trọng:
   - "xếp loại rèn luyện" → "điều kiện xếp loại rèn luyện sinh viên TNUT"
   - "học phí" → "mức học phí đào tạo đại học chính quy"
   - "thi lại" → "quy định thi cải thiện điểm môn học"
   - "điểm A/B/C/D" → "thang điểm chữ 4.0 quy đổi tín chỉ"
   - Thêm: điều kiện, quy định, mức, thủ tục, tiêu chuẩn (nếu phù hợp)

2. **Biến thể đồng nghĩa/liên quan** - Dùng thuật ngữ khác:
   - "điểm rèn luyện" → "đánh giá kết quả rèn luyện sinh viên"
   - "tốt nghiệp" → "điều kiện công nhận tốt nghiệp đại học"
   - "học bổng" → "xét cấp học bổng khuyến khích học tập"
   - "GPA" → "điểm trung bình tích lũy hệ số 4"
   - Dùng: đánh giá, xét, cấp, công nhận, thực hiện (nếu phù hợp)

3. **Biến thể khác góc nhìn** - Hỏi từ khía cạnh khác:
   - "được bao nhiêu điểm?" → "tiêu chuẩn đạt điểm tối thiểu là gì?"
   - "khi nào?" → "thời gian quy định thực hiện"
   - "có được không?" → "điều kiện đủ để thực hiện"

THUẬT NGỮ TNUT CẦN LƯU Ý:
- Điểm 0-10: có thể là điểm thi môn học (thang 10) HOẶC GPA (thang 4)
- Điểm 50-100: điểm rèn luyện (thang 100)
- Điểm A/B/C/D/F: thang điểm chữ (hệ 4.0)
- "học lại" = không đạt môn học
- "thi cải thiện" = thi lại để nâng điểm

YÊU CẦU OUTPUT:
- Tạo ĐÚNG 2 biến thể
- Mỗi biến thể phải KHÁC GÓC NHÌN với câu gốc
- CHỈ GHI 2 DÒNG, KHÔNG số thứ tự, KHÔNG giải thích
- Mỗi dòng là 1 câu hỏi hoàn chỉnh

VÍ DỤ:

Input: "90 điểm rèn luyện được xếp loại gì?"
Output:
điều kiện xếp loại xuất sắc rèn luyện sinh viên TNUT là bao nhiêu điểm
tiêu chuẩn đánh giá kết quả rèn luyện xếp hạng cao nhất

Input: "3 điểm được tích gì?"
Output:
điểm trung bình tích lũy GPA 3.0 quy đổi thang điểm chữ như thế nào
điểm thi môn học đạt 3.0 tương ứng xếp loại gì

Input: "điểm A là bao nhiêu?"
Output:
thang điểm chữ A quy đổi sang điểm số hệ 4.0 là bao nhiêu
tiêu chuẩn đạt điểm A trong hệ thống tín chỉ TNUT

Bây giờ hãy tạo 2 biến thể cho câu hỏi trên:`;

        const result = await expansionModel.invoke(prompt);
        const content = result.content ? result.content.trim() : result.toString().trim();
        const variants = content.split('\n').filter(v => v.trim()).map(v => v.trim());

        const queries = [originalQuery, ...variants.slice(0, 2)];

        return queries;
    } catch (error) {
        console.error('Query expansion failed:', error.message);
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
            console.error('Search failed:', error.message);
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
            console.error('Cache check failed:', cacheError.message);
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

        const template = `Bạn là trợ lý AI chuyên nghiệp hỗ trợ sinh viên Trường Đại học Kỹ thuật Công nghiệp – Đại học Thái Nguyên (TNUT).

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
   - BẮT ĐẦU trực tiếp bằng "TNUT có..." hoặc "Trường ĐHKTCN có..." - KHÔNG dùng "Dựa trên tài liệu/context..."
   - Nói như chuyên gia nắm rõ, KHÔNG đề cập đến nguồn thông tin
   - In đậm số liệu quan trọng (số tiền, điểm số, hạn chót)

2. PHÂN BIỆT ĐIỂM SỐ (RẤT QUAN TRỌNG):
   - Điểm 0-4.0: GPA hoặc thang điểm chữ (A/B/C/D/F) - VD: "GPA 3.5", "điểm A = 4.0"
   - Điểm 5.0-10: Điểm thi môn học (thang 10) - VD: "đạt 8.0", "điểm thi 7.5"
   - Điểm 50-100: Điểm rèn luyện (thang 100) - VD: "đạt 90 điểm rèn luyện", "xếp loại Xuất sắc"
   - KHÔNG nhầm lẫn giữa 3 loại điểm này
   - Nếu context nói về GPA/điểm chữ thì TRẢ LỜI về GPA/điểm chữ
   - Nếu context nói về điểm rèn luyện thì TRẢ LỜI về điểm rèn luyện
   - CHỈ TRẢ LỜI những gì có trong context, KHÔNG suy đoán

3. ĐỘ DÀI:
   - Trả lời NGẮN GỌN, đi thẳng vào vấn đề
   - Danh sách: Liệt kê ĐẦY ĐỦ TẤT CẢ items từ context (VD: nếu có 8 khoa thì liệt kê cả 8)
   - Lưu ý: CHỈ 1 câu ngắn hoặc bỏ qua nếu không cần thiết

4. LIÊN HỆ:
   - Ưu tiên thông tin chi tiết từ context: tên người, chức vụ, SĐT, email
   - VD: "Liên hệ: ThS. Nguyễn Văn A - Trưởng phòng Đào tạo - 0280.3858568 - daotao@tnut.edu.vn"
   - Chỉ nói chung "Liên hệ Phòng Đào tạo" nếu context KHÔNG có thông tin cụ thể

5. CẤU TRÚC:
   - Câu mở đầu: Trả lời trực tiếp
   - Nội dung: Thông tin chi tiết (danh sách đầy đủ nếu có)
   - Kết thúc: Thông tin liên hệ CỤ THỂ (nếu có trong context)

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
            console.error('Failed to save to cache:', cacheError.message);
        }

        res.status(200).json({
            answer,
            chatId,
            sources,
            cached: false
        });

    } catch (error) {
        console.error('Request failed:', error.message);

        res.status(500).json({
            error: "Lỗi hệ thống. Vui lòng thử lại sau."
        });
    }
};