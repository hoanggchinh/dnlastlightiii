const { pool } = require('../utils/db');
const { sanitizeQuestion } = require('../utils/sanitize');
const { findInSemanticCache, saveToSemanticCache } = require('../utils/semanticCache');

// LangChain imports
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { ChatAnthropic } = require("@langchain/anthropic");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");

// Config
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "rag-do-an";
const MODEL_NAME = "claude-3-5-haiku-20241022";

module.exports = async (req, res) => {
    // Chỉ nhận method POST
    if (req.method !== 'POST') return res.status(405).json({ error: "Method not allowed" });

    try {
        let { question, userId = 1, chatId } = req.body;

        // 1. Sanitize (Làm sạch input)
        question = sanitizeQuestion(question);
        if (!question) return res.status(400).json({ error: "Câu hỏi không hợp lệ" });

        // 2. EMBEDDING (CHỈ LÀM 1 LẦN DUY NHẤT TẠI ĐÂY) 💎
        // Tiết kiệm thời gian và tiền bạc, dùng vector này cho cả Cache và Pinecone
        console.log("🧠 Đang tạo Vector...");
        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: process.env.GEMINI_API_KEY,
        });
        const queryVector = await embeddings.embedQuery(question);

        // 3. CHECK SEMANTIC CACHE (Dùng vector vừa tạo)
        const cachedAnswer = await findInSemanticCache(queryVector);

        // --- QUẢN LÝ DB: Tạo Chat Session nếu chưa có ---
        if (!chatId) {
            const newChat = await pool.query(
                `INSERT INTO chats (user_id, title) VALUES ($1, $2) RETURNING id`,
                [userId, question.substring(0, 50)]
            );
            chatId = newChat.rows[0].id;
        }

        // Lưu câu hỏi user vào lịch sử
        await pool.query(`INSERT INTO messages (chat_id, role, content) VALUES ($1, 'user', $2)`, [chatId, question]);

        // === TRƯỜNG HỢP 1: CÓ CACHE ===
        if (cachedAnswer) {
            // Vẫn lưu câu trả lời từ cache vào lịch sử chat để hiển thị lại
            await pool.query(
                `INSERT INTO messages (chat_id, role, content, sources) VALUES ($1, 'assistant', $2, $3)`,
                [chatId, cachedAnswer, JSON.stringify({ source: "cache" })]
            );
            return res.status(200).json({ answer: cachedAnswer, chatId, cached: true });
        }

        // === TRƯỜNG HỢP 2: KHÔNG CÓ CACHE -> RAG ===
        console.log("🔍 Cache Miss -> Tìm trong Pinecone...");

        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });

        // QUAN TRỌNG: Tìm bằng VECTOR có sẵn, không embed lại! 🚀
        const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 4);

        // Lọc kết quả (chỉ lấy độ chính xác > 0.35 để tránh rác)
        const relevantDocs = results.filter(r => r[1] > 0.35);

        let context = "";
        let sources = [];

        if (relevantDocs.length > 0) {
            context = relevantDocs.map(r => r[0].pageContent).join("\n\n");
            sources = relevantDocs.map(r => r[0].metadata);
        } else {
            context = "Không tìm thấy thông tin cụ thể trong tài liệu.";
        }

        // Gọi Claude
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

        // Lưu DB & Cache
        await Promise.all([
            // Lưu lịch sử chat
            pool.query(`INSERT INTO messages (chat_id, role, content, sources) VALUES ($1, 'assistant', $2, $3)`,
                [chatId, answer, JSON.stringify(sources)]),
            // Lưu Semantic Cache cho lần sau
            saveToSemanticCache(question, answer, queryVector)
        ]);

        res.status(200).json({ answer, chatId, sources });

    } catch (error) {
        console.error("❌ Lỗi Server:", error);
        res.status(500).json({ error: "Lỗi hệ thống. Vui lòng thử lại sau." });
    }
};

