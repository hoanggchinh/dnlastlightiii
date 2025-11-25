// Import thư viện
const { Pinecone } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("@langchain/pinecone");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { ChatAnthropic } = require("@langchain/anthropic");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");


const PINECONE_INDEX_NAME = "rag-do-an";
const MODEL_NAME = "claude-3-5-haiku-20241022"; // Model mới nhất, rẻ, thông minh
const MAX_CONTEXT_LENGTH = 6000; // Giới hạn ký tự context (khoảng 1500 tokens) để tiết kiệm tiền
const TOP_K = 4; // Chỉ lấy 4 đoạn liên quan nhất (thay vì 5-10 gây nhiễu và tốn tiền)

module.exports = async (req, res) => {

    if (req.method !== 'POST') {
        return res.status(405).json({ error: "Method not allowed" });
    }

    try {
        // --- LOG 1: KIỂM TRA INPUT ---
        console.log('1. Bắt đầu xử lý request...');
        const { question } = req.body;
        console.log(`2. Câu hỏi nhận được: "${question}"`);

        // Validate input
        if (!question || typeof question !== 'string') {
            console.error("Lỗi: Câu hỏi rỗng hoặc không hợp lệ.");
            return res.status(400).json({ error: "Câu hỏi không hợp lệ." });
        }

        // --- 1. LẤY API KEY VÀ KHỞI TẠO CÁC THÀNH PHẦN ---
        // Lấy API Key từ Environment Variables
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const googleApiKey = process.env.GEMINI_API_KEY;
        const anthropicApiKey = process.env.ANTHROPIC_API_KEY;

        if (!pineconeApiKey || !googleApiKey || !anthropicApiKey) {
            console.error("Lỗi: Thiếu một hoặc nhiều API Key trong Environment Variables.");
            return res.status(500).json({ error: "Lỗi cấu hình server (Thiếu API Key)." });
        }

        const pinecone = new Pinecone({ apiKey: pineconeApiKey });
        const pineconeIndex = pinecone.Index(PINECONE_INDEX_NAME);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: googleApiKey,
            model: "text-embedding-004",
        });


        const model = new ChatAnthropic({
            apiKey: anthropicApiKey,
            model: MODEL_NAME,
            maxTokens: 1024,
            temperature: 0.3
        });


        console.log('3. Khởi tạo Pinecone, Embeddings và Model thành công.');



        const vectorStore = new PineconeStore(embeddings, { pineconeIndex });

        const results = await vectorStore.similaritySearch(question, TOP_K);


        console.log(`4. Đã tìm thấy ${results.length} tài liệu liên quan.`);



        let contextData = results
            .map(doc => doc.pageContent)
            .join("\n---\n")
            .substring(0, MAX_CONTEXT_LENGTH);


        console.log(`5. Kích thước Context Data được truyền vào AI: ${contextData.length} ký tự.`);



        const template = `Bạn là một trợ lý AI hỗ trợ sinh viên, nhiệt tình và am hiểu quy chế của Trường Đại học Kỹ thuật Công nghiệp - Đại học Thái Nguyên
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

        const chain = prompt.pipe(model).pipe(new StringOutputParser());

        const response = await chain.invoke({
            context: contextData,
            question: question
        });
        console.log(`6. AI đã trả lời thành công. Kích thước câu trả lời: ${response.length} ký tự.`);

        return res.status(200).json({
            answer: response,
        });
    } catch (error) {
        console.error('7. LỖI SERVER XẢY RA: ', error);
        if (error.message && (error.message.includes('API_KEY') || error.message.includes('Auth'))) {
             return res.status(500).json({ error: "Lỗi xác thực API. Vui lòng kiểm tra lại API Key trên Vercel." });
        }
        return res.status(500).json({ error: "Lỗi Server nội bộ không xác định." });
    }
};