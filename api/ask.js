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
            ...meta
        }));
    },
    warn: (message, meta = {}) => {
        console.warn(JSON.stringify({
            level: 'warn',
            message,
            timestamp: new Date().toISOString(),
            ...meta
        }));
    },
    error: (message, error = null) => {
        console.error(JSON.stringify({
            level: 'error',
            message,
            timestamp: new Date().toISOString(),
            error: error ? {
                message: error.message,
                stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
            } : undefined
        }));
    }
};


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

        const history = res.rows.reverse().map(msg => {
            return `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`;
        }).join('\n');

        logger.info('Chat history loaded', { chatId, messageCount: res.rows.length });
        return history;
    } catch (error) {
        logger.error('Failed to load chat history', error);
        return "";
    }
}


async function rewriteQuestion(rawQuestion, history, apiKey) {
    const startTime = Date.now();

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

        const duration = Date.now() - startTime;
        logger.info('Question rewritten', {
            original: rawQuestion,
            rewritten,
            duration: `${duration}ms`
        });

        return rewritten;
    } catch (error) {
        logger.warn('Question rewrite failed, using original', { error: error.message });
        return rawQuestion;
    }
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

    logger.info('Chunks reranked', {
        totalChunks: rankedResults.length,
        topScore: rankedResults[0]?.relevanceScore.toFixed(3),
        avgKeywordMatch: (rankedResults.reduce((sum, r) => sum + r.keywordMatchCount, 0) / rankedResults.length).toFixed(1)
    });

    return rankedResults;
}


function buildContext(rankedResults, topK = 5) {
    const topChunks = rankedResults.slice(0, topK);
    const uniqueChunks = [];

    // Deduplication
    const seenContent = new Set();
    for (const chunk of topChunks) {
        const signature = chunk.doc.pageContent.substring(0, 100).trim();
        if (!seenContent.has(signature)) {
            uniqueChunks.push(chunk);
            seenContent.add(signature);
        }
    }

    logger.info('Context built', {
        originalChunks: topChunks.length,
        uniqueChunks: uniqueChunks.length,
        totalCharacters: uniqueChunks.reduce((sum, c) => sum + c.doc.pageContent.length, 0)
    });

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

        const newChatId = result.rows[0].id;
        logger.info('New chat created', { chatId: newChatId, userId });
        return newChatId;
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
        logger.info('Message saved', { chatId, role, contentLength: content.length });
    } catch (error) {
        logger.error('Failed to save message', error);
        throw error;
    }
}

module.exports = async (req, res) => {
    const requestId = Math.random().toString(36).substring(7);
    const startTime = Date.now();

    logger.info('Request received', {
        requestId,
        method: req.method,
        body: req.body
    });

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
            logger.warn('Question truncated', {
                originalLength: question.length,
                maxLength: MAX_QUESTION_LENGTH
            });
            question = question.substring(0, MAX_QUESTION_LENGTH);
        }

        logger.info('Question received', { requestId, question });


        let chatHistory = "";
        if (chatId) {
            chatHistory = await getChatHistory(chatId);
        }

        const refinedQuestion = await rewriteQuestion(
            question,
            chatHistory,
            process.env.ANTHROPIC_API_KEY
        );


        logger.info('Creating embedding', { requestId });
        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: process.env.GEMINI_API_KEY,
        });

        const queryVector = await embeddings.embedQuery(refinedQuestion);

        logger.info('Embedding created', {
            requestId,
            vectorLength: queryVector?.length,
            vectorType: typeof queryVector
        });

        let cachedAnswer = null;
        try {
            if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
                cachedAnswer = await findInSemanticCache(refinedQuestion, queryVector);
                logger.info('Cache checked', { requestId, found: !!cachedAnswer });
            } else {
                logger.warn('Invalid query vector, skipping cache', {
                    requestId,
                    vectorType: typeof queryVector,
                    isArray: Array.isArray(queryVector)
                });
            }
        } catch (cacheError) {
            logger.error('Cache check failed', cacheError);
        }

        if (cachedAnswer) {
            logger.info('Cache HIT', { requestId });

            chatId = await ensureChatId(chatId, userId, question);

            await saveMessage(chatId, 'user', question);
            await saveMessage(chatId, 'assistant', cachedAnswer, { source: "cache" });

            const duration = Date.now() - startTime;
            logger.info('Request completed from cache', { requestId, duration: `${duration}ms` });

            return res.status(200).json({
                answer: cachedAnswer,
                chatId,
                cached: true
            });
        }

        logger.info('Cache MISS', { requestId });


        logger.info('Searching Pinecone', { requestId });
        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: index
        });

        const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 10);
        const relevantDocs = results.filter(r => r[1] > SIMILARITY_THRESHOLD);

        logger.info('Search results', {
            requestId,
            totalResults: results.length,
            relevantResults: relevantDocs.length,
            threshold: SIMILARITY_THRESHOLD,
            topScore: results[0]?.[1].toFixed(3)
        });

        let context = "";
        let sources = [];

        if (relevantDocs.length > 0) {
            const rankedResults = rerankChunks(relevantDocs, refinedQuestion);
            context = buildContext(rankedResults, 5);
            sources = rankedResults.slice(0, 5).map(item => item.doc.metadata);
        } else {
            context = "Không tìm thấy thông tin cụ thể trong tài liệu.";
            logger.warn('No relevant documents found', { requestId });
        }

        logger.info('Calling Claude API', { requestId });
        const model = new ChatAnthropic({
            modelName: MODEL_NAME,
            apiKey: process.env.ANTHROPIC_API_KEY,
            temperature: 0.3,
            maxTokens: 1024
        });

        const template = `
Bạn là trợ lý AI chuyên nghiệp hỗ trợ sinh viên Trường Đại học Kỹ thuật Công nghiệp – Đại học Thái Nguyên (TNUT).

<history>
{chat_history}
</history>

<context>
{context}
</context>

Câu hỏi gốc của sinh viên: "{question}"
(Ý định thực sự: "{refined_question}")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HƯỚNG DẪN TRẢ LỜI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SỬ DỤNG CONTEXT & HISTORY:
   - Kết hợp thông tin từ Context và Lịch sử trò chuyện để trả lời mạch lạc.
   - Nếu Context rỗng hoặc không liên quan, hãy nói khéo léo là chưa tìm thấy thông tin trong quy chế hiện tại.

2. PHONG CÁCH TƯ VẤN:
   - Trả lời TRỰC DIỆN, không vòng vo.
   - Không nói "Dựa trên context...", hãy nói như một chuyên gia nắm rõ quy chế.
   - Dùng format danh sách, in đậm các ý chính (số tiền, điểm số, hạn chót).

3. XỬ LÝ CÂU HỎI THIẾU THÔNG TIN:
   - Nếu không đủ dữ liệu để khẳng định, hãy hướng dẫn sinh viên liên hệ Phòng Đào tạo hoặc Giáo vụ khoa.

4. CẤU TRÚC:
   - Mở đầu: Trả lời ngay vấn đề.
   - Thân bài: Chi tiết quy định/giải thích.
   - Kết thúc: Lưu ý hoặc lời khuyên bổ ích.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BẮT ĐẦU TRẢ LỜI:
`;

        const chain = PromptTemplate.fromTemplate(template)
            .pipe(model)
            .pipe(new StringOutputParser());

        const answer = await chain.invoke({
            context,
            question,
            refined_question: refinedQuestion,
            chat_history: chatHistory
        });

        logger.info('Answer generated', {
            requestId,
            answerLength: answer.length
        });

        chatId = await ensureChatId(chatId, userId, question);

        await saveMessage(chatId, 'user', question);
        await saveMessage(chatId, 'assistant', answer, sources);

        try {
            if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
                await saveToSemanticCache(refinedQuestion, answer, queryVector);
                logger.info('Answer cached successfully', { requestId });
            }
        } catch (cacheError) {
            logger.error('Failed to save to cache', cacheError);
        }

        const duration = Date.now() - startTime;
        logger.info('Request completed successfully', {
            requestId,
            duration: `${duration}ms`,
            chatId
        });

        res.status(200).json({
            answer,
            chatId,
            sources,
            cached: false
        });

    } catch (error) {
        const duration = Date.now() - startTime;
        logger.error('Request failed', error);

        res.status(500).json({
            error: "Lỗi hệ thống. Vui lòng thử lại sau.",
            requestId,
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};