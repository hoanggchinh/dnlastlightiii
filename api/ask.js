// Import các thư viện cần thiết
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { PromptTemplate } = require("@langchain/core/prompts");

// Cấu hình: Tên Index Pinecone của bạn
const PINECONE_INDEX_NAME = "rag-do-an";

// Hàm handler chính của Vercel
module.exports = async (req, res) => {
    // Chỉ cho phép phương thức POST
    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method not allowed" });
    }

    try {
        // 1. Lấy câu hỏi từ body của request
        const { question } = req.body;
        if (!question || typeof question !== 'string') {
            return res.status(400).json({ error: "Question is required and must be a string" });
        }

        // 2. Lấy API keys từ Biến Môi Trường của Vercel
        const googleApiKey = process.env.GEMINI_API_KEY;
        const pineconeApiKey = process.env.PINECONE_API_KEY;

        if (!googleApiKey || !pineconeApiKey) {
            console.error("Missing API keys:", {
                hasGemini: !!googleApiKey,
                hasPinecone: !!pineconeApiKey
            });
            return res.status(500).json({
                error: "API keys not configured (GEMINI_API_KEY or PINECONE_API_KEY missing)"
            });
        }

        console.log("Initializing services...");

        // 3. Khởi tạo các dịch vụ
        const pinecone = new Pinecone({ apiKey: pineconeApiKey });
        const pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: googleApiKey,
        });

        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex,
        });

        const model = new ChatGoogleGenerativeAI({
            model: "gemini-2.5-flash",
            apiKey: googleApiKey,
            temperature: 0.3,
        });

        // 4. Tạo Prompt Template
        const promptTemplate = PromptTemplate.fromTemplate(`
Bạn là một trợ lý AI hữu ích của trường đại học.
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên dựa trên các tài liệu nội bộ của trường.
Chỉ sử dụng thông tin từ "NGỮ CẢNH" được cung cấp.
Nếu "NGỮ CẢNH" không chứa thông tin để trả lời, hãy nói: "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu."
Không được bịa đặt thông tin.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI (bằng tiếng Việt):
        `);

        // 5. Tìm kiếm và tạo câu trả lời
        console.log("Searching for relevant documents...");

        // Tìm kiếm 4 documents liên quan nhất
        const retriever = vectorStore.asRetriever(4);
        const relevantDocs = await retriever.invoke(question);

        console.log(`Found ${relevantDocs.length} relevant documents`);

        // Kết hợp nội dung các documents thành context
        const context = relevantDocs
            .map((doc) => doc.pageContent)
            .join("\n\n");

        // Tạo prompt với context và question
        const prompt = await promptTemplate.format({ context, question });

        console.log("Generating answer...");

        // Gọi model để tạo câu trả lời
        const response = await model.invoke(prompt);

        // Lấy nội dung text từ response
        const answer = typeof response === 'string'
            ? response
            : response.content || response.text || String(response);

        console.log("Answer generated successfully");

        // Gửi câu trả lời về cho frontend
        res.status(200).json({
            answer: answer,
            // Optional: trả về metadata để debug (có thể bỏ)
            metadata: {
                documentsFound: relevantDocs.length
            }
        });

    } catch (error) {
        console.error("Error processing request:", error);

        // Trả về lỗi chi tiết hơn
        const errorMessage = error.message || "Unknown error occurred";
        const errorType = error.constructor.name;

        res.status(500).json({
            error: "An error occurred while processing your request",
            details: errorMessage,
            type: errorType
        });
    }
};