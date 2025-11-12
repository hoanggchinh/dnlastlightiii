// Import các thư viện cần thiết
// quan trọng: đây là cú pháp "require" của Node.js, không phải "import"
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { PromptTemplate } = require("@langchain/core/prompts");
const { RunnableSequence } = require("@langchain/core/runnables");
const { StringOutputParser } = require("@langchain/core/output_parsers");

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
        if (!question) {
            return res.status(400).json({ error: "Question is required" });
        }

        // 2. Lấy API keys từ Biến Môi Trường của Vercel
        const googleApiKey = process.env.GEMINI_API_KEY;
        const pineconeApiKey = process.env.PINECONE_API_KEY;

        if (!googleApiKey || !pineconeApiKey) {
            return res.status(500).json({ error: "API keys not configured" });
        }

        // 3. Khởi tạo các dịch vụ
        // Khởi tạo Pinecone
        const pinecone = new Pinecone({ apiKey: pineconeApiKey });
        const pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);

        // Khởi tạo model Embedding
        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "models/text-embedding-004",
            apiKey: googleApiKey,
        });

        // Khởi tạo Vector Store
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex,
        });

        // Khởi tạo model Chat (Gemini)
        const model = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            apiKey: googleApiKey,
            temperature: 0.3, // Giảm độ "sáng tạo"
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

        // 5. Tạo chuỗi xử lý (RAG Chain)

        // Bước 1: Tìm kiếm ngữ cảnh (Retrieval)
        // Tạo một "retriever" để tìm 4 chunk liên quan nhất
        const retriever = vectorStore.asRetriever(4);

        // Hàm này sẽ kết hợp các document (chunk) tìm được thành 1 chuỗi văn bản
        const formatContext = (docs) => docs.map((doc) => doc.pageContent).join("\n\n");

        // Bước 2: Tạo chuỗi (Chain)
        const ragChain = RunnableSequence.from([
            {
                // Lấy context: Dùng retriever tìm doc, sau đó format lại
                context: async (input) => {
                    const docs = await retriever.invoke(input.question);
                    return formatContext(docs);
                },
                // Giữ nguyên câu hỏi
                question: (input) => input.question,
            },
            promptTemplate, // Gửi context và question vào prompt
            model,          // Gửi prompt cho model
            new StringOutputParser(), // Lấy kết quả dạng text
        ]);

        // 6. Thực thi: Tìm kiếm và tạo câu trả lời
        console.log("Đang tìm kiếm tài liệu liên quan...");

        // Bước 1: Tìm kiếm documents
        const relevantDocs = await retriever.invoke(question);
        const context = formatContext(relevantDocs);

        console.log("Đã tìm thấy", relevantDocs.length, "tài liệu. Đang tạo câu trả lời...");

        // Bước 2: Tạo câu trả lời
        const prompt = await promptTemplate.format({ context, question });
        const answer = await model.invoke(prompt);

        console.log("Đã có câu trả lời.");

        // Gửi câu trả lời về cho frontend
        res.status(200).json({ answer: answer.content });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "An error occurred: " + error.message });
    }
};