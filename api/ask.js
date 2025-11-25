// Import thư viện
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { ChatAnthropic } = require("@langchain/anthropic");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");


const PINECONE_INDEX_NAME = "rag-do-an";
const MODEL_NAME = "claude-4-5-haiku-20241022"; // Model mới nhất, rẻ, thông minh
const MAX_CONTEXT_LENGTH = 6000; // Giới hạn ký tự context (khoảng 1500 tokens) để tiết kiệm tiền
const TOP_K = 4; // Chỉ lấy 4 đoạn liên quan nhất (thay vì 5-10 gây nhiễu và tốn tiền)

module.exports = async (req, res) => {

    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method not allowed" });
    }

    try {
        const { question } = req.body;

        // Validate input
        if (!question || typeof question !== 'string') {
            return res.status(400).json({ error: "Câu hỏi không hợp lệ." });
        }

        // Lấy API Key
        const googleApiKey = process.env.GEMINI_API_KEY;
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const anthropicApiKey = process.env.ANTHROPIC_API_KEY;

        if (!googleApiKey || !pineconeApiKey || !anthropicApiKey) {
            return res.status(500).json({ error: "Missing API key" });
        }

        // --- 1. KẾT NỐI DB & TÌM KIẾM ---

        // Khởi tạo Embedding (Phải khớp model với file ingest.py)
        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: googleApiKey,
        });

        // Kết nối Pinecone
        const pinecone = new Pinecone({ apiKey: pineconeApiKey });
        const index = pinecone.Index(PINECONE_INDEX_NAME);

        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: index,
        });

        // Tìm kiếm vector (Similarity Search)
        // Kỹ thuật: Chỉ lấy Top 4 để tiết kiệm token nhưng vẫn đủ thông tin
        const results = await vectorStore.similaritySearch(question, TOP_K);

        // --- 2. XỬ LÝ CONTEXT (TIẾT KIỆM TOKEN) ---

        let contextData = "";
        let currentLength = 0;

        for (const doc of results) {
            // Xử lý text: Xóa bớt xuống dòng thừa, khoảng trắng thừa để tiết kiệm token
            const cleanContent = doc.pageContent.replace(/\n+/g, " ").replace(/\s+/g, " ").trim();

            // Kiểm tra giới hạn độ dài
            if (currentLength + cleanContent.length < MAX_CONTEXT_LENGTH) {
                // Thêm thẻ XML <doc> để Claude hiểu rõ đây là tài liệu tham khảo
                contextData += `<doc>\n${cleanContent}\n</doc>\n`;
                currentLength += cleanContent.length;
            } else {
                break; // Đã đủ thông tin, dừng lại để không tốn thêm tiền
            }
        }

        if (!contextData) {
            return res.status(200).json({ answer: "Xin lỗi, tôi không tìm thấy thông tin nào trong tài liệu liên quan đến câu hỏi của bạn." });
        }

        // --- 3. PROMPT ENGINEERING (NÂNG CẤP) ---

        // Model Config
        const model = new ChatAnthropic({
            modelName: MODEL_NAME,
            apiKey: anthropicApiKey,
            temperature: 0.3, // Thấp để trả lời chính xác theo tài liệu
            maxTokens: 1024,  // Tăng lên để tránh bị cắt giữa chừng (quan trọng!)
        });

        // Template thông minh: Sử dụng kỹ thuật "Role-playing" và "Format instruction"
        const template = `
Bạn là một trợ lý AI hỗ trợ sinh viên, nhiệt tình và am hiểu quy chế.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp trong thẻ <context>.

<context>
{context}
</context>

Câu hỏi của sinh viên: "{question}"

Yêu cầu trả lời:
1. CHỈ sử dụng thông tin trong <context> để trả lời. Không bịa đặt.
2. Nếu không có thông tin, hãy nói "Tài liệu không đề cập đến vấn đề này".
3. Trình bày câu trả lời rõ ràng, đẹp mắt bằng Markdown:
   - Sử dụng **in đậm** cho các ý chính.
   - Sử dụng gạch đầu dòng (-) cho các danh sách.
   - Chia đoạn văn hợp lý, không viết dính liền một khối.
4. Giọng văn thân thiện, ngắn gọn, súc tích (đừng dài dòng lê thê).

Câu trả lời:`;

        const prompt = PromptTemplate.fromTemplate(template);

        // --- 4. GỌI AI & TRẢ VỀ KẾT QUẢ ---

        const chain = prompt.pipe(model).pipe(new StringOutputParser());

        const response = await chain.invoke({
            context: contextData,
            question: question
        });

        // Trả về JSON
        return res.status(200).json({
            answer: response,
            // (Optional) Trả về sources để debug xem nó lấy tin từ đâu
            // sources: results.map(r => r.metadata.source || "unknown")
        });

    } catch (error) {
        console.error("Lỗi xử lý:", error);
        return res.status(500).json({ error: "Đã có lỗi xảy ra khi xử lý câu hỏi." });
    }
};