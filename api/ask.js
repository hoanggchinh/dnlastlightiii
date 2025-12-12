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

const logger = {
    info: (message, meta = {}) => {
        console.log(JSON.stringify({
            level: 'info',
            message,
            timestamp: new Date().toISOString(),
            ...sanitizeMeta(meta)
        }));
    },
    warn: (message, meta = {}) => {
        console.warn(JSON.stringify({
            level: 'warn',
            message,
            timestamp: new Date().toISOString(),
            ...sanitizeMeta(meta)
        }));
    },
    error: (message, error = null) => {
        const errorInfo = error ? {
            message: error.message,
            name: error.name,
            code: error.code,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        } : undefined;

        console.error(JSON.stringify({
            level: 'error',
            message,
            timestamp: new Date().toISOString(),
            error: errorInfo
        }));
    }
};

function sanitizeMeta(meta) {
    const sanitized = {};
    for (const [key, value] of Object.entries(meta)) {
        if (typeof value === 'string') {
            if (value.includes('<!DOCTYPE') || value.includes('<html')) {
                sanitized[key] = '[HTML Content - ' + value.length + ' chars]';
            }
            else if (value.length > 500) {
                sanitized[key] = value.substring(0, 500) + '... [truncated]';
            }
            else {
                sanitized[key] = value;
            }
        } else {
            sanitized[key] = value;
        }
    }
    return sanitized;
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
        logger.error('Failed to load chat history', error);
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

        const prompt = `Bạn là trợ lý AI thông minh. Nhiệm vụ của bạn là viết lại câu hỏi của người dùng để tìm kiếm trong tài liệu quy chế đào tạo đại học.

Lịch sử hội thoại (Context):
"""
${history || "Không có lịch sử"}
"""

Câu hỏi hiện tại của người dùng: "${rawQuestion}"

Yêu cầu:
1. Nếu câu hỏi thiếu chủ ngữ hoặc phụ thuộc vào lịch sử (ví dụ: "còn 6 điểm thì sao?", "thang điểm 4"), hãy DÙNG LỊCH SỬ để điền đầy đủ thông tin.
2. Sửa lỗi chính tả, từ lóng (ví dụ: "tích gì" -> "xếp loại gì", "rớt môn" -> "học lại").
3. Viết lại thành một câu truy vấn đầy đủ, rõ ràng, đúng thuật ngữ hành chính.
4. CHỈ TRẢ VỀ CÂU ĐÃ VIẾT LẠI, KHÔNG GIẢI THÍCH GÌ THÊM.

Câu hỏi viết lại:`;

        const result = await rewriteModel.invoke(prompt);
        const rewritten = result.content ? result.content.trim() : result.toString().trim();

        logger.info('Question rewritten', {
            original: rawQuestion,
            rewritten
        });

        return rewritten;
    } catch (error) {
        logger.warn('Question rewrite failed, using original', { error: error.message });
        return rawQuestion;
    }
}

async function expandQuery(originalQuery, apiKey) {
    try {
        const expansionModel = new ChatAnthropic({
            modelName: "claude-3-haiku-20240307",
            apiKey: apiKey,
            temperature: 0,
            maxTokens: 150
        });

        const prompt = `Tạo 2 biến thể của câu hỏi để tìm kiếm tốt hơn trong tài liệu quy chế.

Câu hỏi gốc: "${originalQuery}"

QUY TẮC:
1. Biến thể 1: Thêm từ khóa hành động (ví dụ: "học phí quốc phòng" → "đóng học phí quốc phòng")
2. Biến thể 2: Dùng từ đồng nghĩa (ví dụ: "quốc phòng" → "an ninh")

CHỈ TRẢ VỀ 2 DÒNG, mỗi dòng 1 biến thể, KHÔNG số thứ tự, KHÔNG giải thích:`;

        const result = await expansionModel.invoke(prompt);
        const content = result.content ? result.content.trim() : result.toString().trim();
        const variants = content.split('\n').filter(v => v.trim()).map(v => v.trim());

        const queries = [originalQuery, ...variants.slice(0, 2)];

        logger.info('Query expanded', {
            original: originalQuery,
            variants: queries
        });

        return queries;
    } catch (error) {
        logger.warn('Query expansion failed', { error: error.message });
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
            logger.warn('Search failed for query', { query, error: error.message });
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
        logger.error('Failed to create chat', error);
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
        logger.error('Failed to save message', error);
        throw error;
    }
}

module.exports = async (req, res) => {
    const requestId = Math.random().toString(36).substring(7);
    const startTime = Date.now();

    logger.info('Request received', { requestId, method: req.method });

    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method not allowed" });
    }

    try {
        let { question, userId = 1, chatId } = req.body;

        if (!userId || userId < 1) {
            return res.status(400).json({ error: "userId không hợp lệ" });
        }

        question = sanitizeQuestion(question);
        if (!question) {
            return res.status(400).json({ error: "Câu hỏi không hợp lệ" });
        }

        if (question.length > MAX_QUESTION_LENGTH) {
            question = question.substring(0, MAX_QUESTION_LENGTH);
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
            logger.error('Cache check failed', cacheError);
        }

        if (cachedAnswer) {
            logger.info('Cache HIT', { requestId });
            chatId = await ensureChatId(chatId, userId, question);
            await saveMessage(chatId, 'user', question);
            await saveMessage(chatId, 'assistant', cachedAnswer, { source: "cache" });

            return res.status(200).json({
                answer: cachedAnswer,
                chatId,
                cached: true
            });
        }

        logger.info('Cache MISS', { requestId });

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

        const results = await hybridSearch(queries, embeddings, pinecone, PINECONE_INDEX_NAME);
        const relevantDocs = results.filter(r => r[1] > SIMILARITY_THRESHOLD);

        logger.info('Search results', {
            requestId,
            totalResults: results.length,
            relevantResults: relevantDocs.length
        });

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

2. ĐỘ DÀI:
   - Trả lời NGẮN GỌN, đi thẳng vào vấn đề
   - Danh sách: Liệt kê ĐẦY ĐỦ TẤT CẢ items từ context (VD: nếu có 8 khoa thì liệt kê cả 8)
   - Lưu ý: CHỈ 1 câu ngắn hoặc bỏ qua nếu không cần thiết

3. LIÊN HỆ:
   - Ưu tiên thông tin chi tiết từ context: tên người, chức vụ, SĐT, email
   - VD: "Liên hệ: ThS. Nguyễn Văn A - Trưởng phòng Đào tạo - 0280.3858568 - daotao@tnut.edu.vn"
   - Chỉ nói chung "Liên hệ Phòng Đào tạo" nếu context KHÔNG có thông tin cụ thể

4. CẤU TRÚC:
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
            logger.error('Failed to save to cache', cacheError);
        }

        const duration = Date.now() - startTime;
        logger.info('Request completed', { requestId, duration: `${duration}ms` });

        res.status(200).json({
            answer,
            chatId,
            sources,
            cached: false
        });

    } catch (error) {
        const duration = Date.now() - startTime;

        logger.error('Request failed', {
            message: error.message,
            requestId,
            duration: `${duration}ms`
        });

        res.status(500).json({
            error: "Lỗi hệ thống. Vui lòng thử lại sau.",
            requestId
        });
    }
};