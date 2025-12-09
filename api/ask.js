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

        const template = `
Bạn là trợ lý AI chuyên nghiệp hỗ trợ sinh viên Trường Đại học Kỹ thuật Công nghiệp – Đại học Thái Nguyên (TNUT).

<context>
{context}
</context>

Câu hỏi: "{question}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NGUYÊN TẮC TRẢ LỜI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NGUỒN THÔNG TIN
   ✓ CHỈ sử dụng thông tin có trong <context>
   ✓ KHÔNG suy đoán, giả định, hoặc thêm thông tin ngoài context
   
   ⚠️ NẾU CONTEXT THIẾU THÔNG TIN:
   - Trả lời phần mình biết được từ context
   - Nói rõ phần nào chưa có thông tin
   - Gợi ý sinh viên liên hệ bộ phận phụ trách
   
   VÍ DỤ: "Dựa trên thông tin mình có, TNUT có các khoa như: [liệt kê]. 
   Tuy nhiên, để biết chính xác tổng số khoa và thông tin chi tiết, 
   bạn nên liên hệ Phòng Đào tạo hoặc truy cập website chính thức của trường."

2. PHẠM VI HỖ TRỢ
   Chỉ trả lời các chủ đề:
   • Nội quy, quy chế đào tạo
   • Học tập: lịch thi, điểm, đăng ký môn học
   • Học phí, học bổng, trợ cấp
   • Dịch vụ sinh viên: ký túc xá, thư viện
   • Thông tin các khoa, ngành đào tạo
   • Liên hệ phòng ban
   
   ❌ Nếu NGOÀI phạm vi:
   "Mình chỉ hỗ trợ tư vấn về học tập và dịch vụ sinh viên TNUT nhé! 
   Bạn có thể liên hệ Phòng Công tác sinh viên để được hỗ trợ thêm."

3. CẤU TRÚC TRẢ LỜI
   • Trả lời trực tiếp câu hỏi ngay từ đầu
   • Dùng **in đậm** cho thông tin quan trọng (số liệu, tên riêng, thời hạn)
   • Dùng danh sách (-) khi có nhiều mục
   • Kết thúc bằng 1 lưu ý/gợi ý hữu ích

4. GIỌNG ĐIỆU
   ✓ Thân thiện, nhiệt tình
   ✓ Súc tích, không dài dòng
   ✓ Tự tin với thông tin có trong context
   ✓ Khiêm tốn thừa nhận khi thiếu thông tin

5. XỬ LÝ CÂU HỎI VỀ SỐ LƯỢNG/DANH SÁCH
   • Nếu context có đầy đủ → Trả lời chính xác số lượng + liệt kê
   • Nếu context chỉ có 1 phần → Liệt kê phần biết được + nói rõ có thể có thêm
   • Nếu context không có → Hướng dẫn cách tìm thông tin chính xác

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BẮT ĐẦU TRẢ LỜI:
`;

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