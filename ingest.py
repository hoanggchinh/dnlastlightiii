import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Tên Index mà bạn đã tạo trên Pinecone
PINECONE_INDEX_NAME = "rag-do-an"
# Thư mục chứa tài liệu
DATA_DIR = "data/"

def main():
    print("Bắt đầu quá trình nạp dữ liệu...")

    # 1. Tải API keys từ file .env
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("Lỗi: Không tìm thấy GOOGLE_API_KEY hoặc PINECONE_API_KEY trong file .env")
        return

    # 2. Tải tài liệu từ thư mục data/
    print(f"Đang tải tài liệu từ thư mục: {DATA_DIR}...")
    # Dùng DirectoryLoader để tự động đọc nhiều loại file (pdf, txt, docx)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.*", show_progress=True, use_multithreading=True)
    documents = loader.load()
    print(f"Đã tải thành công {len(documents)} tài liệu.")

    if not documents:
        print("Không tìm thấy tài liệu nào trong thư mục data/. Vui lòng kiểm tra lại.")
        return

    # 3. Chia nhỏ tài liệu (Split)
    print("Đang chia nhỏ tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia tài liệu thành {len(chunks)} đoạn (chunks).")

    # 4. Khởi tạo mô hình Embedding (Google)
    print("Đang khởi tạo mô hình Embedding (Gemini)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    # 5. Đẩy dữ liệu lên Pinecone (Ingest)
    print(f"Đang đẩy dữ liệu lên Pinecone (Index: {PINECONE_INDEX_NAME})...")
    # LangChain sẽ tự động:
    # - Biến mỗi chunk thành vector bằng model embedding
    # - Đẩy vector và nội dung text (metadata) lên Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
        # Lưu ý: Pinecone API key được tự động đọc từ biến môi trường PINECONE_API_KEY
    )

    print("Hoàn thành! Dữ liệu đã được nạp thành công lên Pinecone.")

if __name__ == "__main__":
    main()