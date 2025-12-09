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

module.exports = async (req, res) => {
    if (req.method !== 'POST') return res.status(405).json({ error: "Method not allowed" });

    try {
        let { question, userId = 1, chatId } = req.body;

        question = sanitizeQuestion(question);
        if (!question) return res.status(400).json({ error: "Câu hỏi không hợp lệ" });

        console.log("Đang tạo Vector...");
        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: process.env.GEMINI_API_KEY,
        });
        const queryVector = await embeddings.embedQuery(question);

        const cachedAnswer = await findInSemanticCache(queryVector);

        if (!chatId) {
            const newChat = await pool.query(
                `INSERT INTO chats (user_id, title) VALUES ($1, $2) RETURNING id`,
                [userId, question.substring(0, 50)]
            );
            chatId = newChat.rows[0].id;
        }

        await pool.query(`INSERT INTO messages (chat_id, role, content) VALUES ($1, 'user', $2)`, [chatId, question]);

        if (cachedAnswer) {
            await pool.query(
                `INSERT INTO messages (chat_id, role, content, sources) VALUES ($1, 'assistant', $2, $3)`,
                [chatId, cachedAnswer, JSON.stringify({ source: "cache" })]
            );
            return res.status(200).json({ answer: cachedAnswer, chatId, cached: true });
        }
        console.log("Cache Miss -> Tìm trong Pinecone...");

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });
        const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 4);
        const relevantDocs = results.filter(r => r[1] > 0.35);

        let context = "";
        let sources = [];

        if (relevantDocs.length > 0) {
            context = relevantDocs.map(r => r[0].pageContent).join("\n\n");
            sources = relevantDocs.map(r => r[0].metadata);
        } else {
            context = "Không tìm thấy thông tin cụ thể trong tài liệu.";
        }
        const model = new ChatAnthropic({
            modelName: MODEL_NAME,
            apiKey: process.env.ANTHROPIC_API_KEY,
            temperature: 0.3,
            maxTokens: 1024
        });

        const template = `Bạn là một trợ lý AI hỗ trợ sinh viên, nhiệt tình và am hiểu quy chế của TNUT - Thai Nguyen University of Technology (Trường Đại học Kỹ thuật Công nghiệp - Đại học Thái Nguyên)
        Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp trong thẻ <context>.
            
            <context>
            {context}
            </context>
            
            Câu hỏi của sinh viên: "{question}"
            
            Yêu cầu trả lời:
            1. Chỉ sử dụng thông tin trong <context> để trả lời. Không bịa đặt.
            2. Nếu thông tin không liên quan đến việc học tập của sinh viên trường TNUT thì trả lời "Mình chỉ hỗ trợ tư vấn nội quy, quy chế cho sinh viên TNUT, ..." 
            3. Trình bày câu trả lời rõ ràng, đẹp mắt bằng Markdown:
               - Sử dụng **in đậm** cho các ý chính.
               - Sử dụng gạch đầu dòng (-) cho các danh sách.
               - Chia đoạn văn hợp lý, không viết dính liền một khối.
            4. Giọng văn thân thiện, ngắn gọn, súc tích (đừng dài dòng lê thê).
            5. Đưa ra lưu ý hoặc lời khuyên liên quan tới câu hỏi cho người hỏi.
            Câu trả lời:`;

        const chain = PromptTemplate.fromTemplate(template).pipe(model).pipe(new StringOutputParser());
        const answer = await chain.invoke({ context, question });

        await Promise.all([
            pool.query(`INSERT INTO messages (chat_id, role, content, sources) VALUES ($1, 'assistant', $2, $3)`,
                [chatId, answer, JSON.stringify(sources)]),
            saveToSemanticCache(question, answer, queryVector)
        ]);

        res.status(200).json({ answer, chatId, sources });

    } catch (error) {
        console.error("Lỗi Server:", error);
        res.status(500).json({ error: "Lỗi hệ thống. Vui lòng thử lại sau." });
    }
};