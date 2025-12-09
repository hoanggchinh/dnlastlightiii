import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

PINECONE_INDEX_NAME = "rag-do-an"
DATA_DIR = "data/"
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DIMENSION = 768


def load_documents_from_directory(directory):
    """
    Load tất cả file TXT, PDF, DOCX từ thư mục
    """
    all_documents = [ ]

    print(f"📂 Đang quét thư mục: {directory}")

    # 1. Load TXT files
    print("📄 Loading .txt files...")
    try:
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True
        )
        txt_docs = txt_loader.load()
        all_documents.extend(txt_docs)
        print(f"✅ Loaded {len(txt_docs)} TXT files")
    except Exception as e:
        print(f"⚠️ Error loading TXT: {e}")

    # 2. Load PDF files
    print("📕 Loading .pdf files...")
    try:
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        pdf_docs = pdf_loader.load()
        all_documents.extend(pdf_docs)
        print(f"✅ Loaded {len(pdf_docs)} PDF files")
    except Exception as e:
        print(f"⚠️ Error loading PDF: {e}")

    # 3. Load DOCX files
    print("📘 Loading .docx files...")
    try:
        docx_loader = DirectoryLoader(
            directory,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
            use_multithreading=True
        )
        docx_docs = docx_loader.load()
        all_documents.extend(docx_docs)
        print(f"✅ Loaded {len(docx_docs)} DOCX files")
    except Exception as e:
        print(f"⚠️ Error loading DOCX: {e}")

    return all_documents


def main():
    print("=" * 60)
    print("🚀 RAG INGESTION PIPELINE - TNUT CHATBOT")
    print("=" * 60)

    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("❌ Missing API keys in .env file")
        return

    # ============================================================
    # BƯỚC 1: XÓA TOÀN BỘ DỮ LIỆU CŨ TRONG INDEX
    # ============================================================
    print(f"\n🗑️  Đang xóa toàn bộ dữ liệu cũ trong index '{PINECONE_INDEX_NAME}'...")

    try:
        pc = Pinecone(api_key=pinecone_api_key)

        # Kiểm tra index có tồn tại không
        existing_indexes = pc.list_indexes().names()

        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"❌ Index '{PINECONE_INDEX_NAME}' không tồn tại!")
            print("Vui lòng tạo index trước hoặc kiểm tra tên index.")
            return

        # Xóa toàn bộ vectors trong index
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)

        print(f"✅ Đã xóa toàn bộ dữ liệu cũ!")
        time.sleep(3)  # Chờ Pinecone xử lý xong

    except Exception as e:
        print(f"❌ Lỗi khi xóa dữ liệu: {e}")
        return

    # ============================================================
    # BƯỚC 2: LOAD DOCUMENTS
    # ============================================================
    documents = load_documents_from_directory(DATA_DIR)

    if not documents:
        print("❌ Không tìm thấy file nào!")
        return

    print(f"\n✅ Tổng số documents: {len(documents)}")

    # ============================================================
    # BƯỚC 3: SPLIT THÀNH CHUNKS
    # ============================================================
    print("\n🔪 Đang chia nhỏ documents thành chunks...")

    separators = [
        r"(?<=\n)Chương\s+[IVX0-9]+",
        r"(?<=\n)Phần\s+[IVX0-9]+",
        r"(?<=\n)[IVX]+\.\s",
        r"(?<=\n)Điều\s+\d+",
        r"(?<=\n)[A-Z]\.\s",
        r"(?<=\n)\d+(\.\d+)+(\.)?\s",
        r"(?<=\n)\d+\.\s",
        r"(?<=\n)[a-z]\)\s",
        r"(?<=\n)-\s",
        "\n\n",
        "\n",
        " ",
        ""
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=separators,
        is_separator_regex=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã tạo {len(chunks)} chunks")

    # Preview chunk đầu tiên
    if chunks:
        print("\n" + "=" * 60)
        print("📋 PREVIEW CHUNK ĐẦU TIÊN:")
        print("=" * 60)
        print(f"Content: {chunks [ 0 ].page_content [ :300 ]}...")
        print(f"\nMetadata: {chunks [ 0 ].metadata}")
        print("=" * 60)

    # ============================================================
    # BƯỚC 4: TẠO EMBEDDINGS VÀ UPLOAD LÊN PINECONE
    # ============================================================
    print(f"\n🔮 Khởi tạo embedding model: {EMBEDDING_MODEL}...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key
    )

    print(f"\n📌 Kết nối tới Pinecone index: '{PINECONE_INDEX_NAME}'...")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Upload chunks theo batch
    print(f"\n⬆️  Đang upload {len(chunks)} chunks lên Pinecone...")
    print("=" * 60)

    batch_size = 10
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    for i in range(0, total_chunks, batch_size):
        batch = chunks [ i:i + batch_size ]
        batch_num = i // batch_size + 1

        print(f"📦 Batch {batch_num}/{total_batches} (chunks {i + 1}-{min(i + batch_size, total_chunks)})...", end=" ")

        try:
            vectorstore.add_documents(batch)
            print("✅")
            time.sleep(2)  # Rate limiting để tránh bị chặn

        except Exception as e:
            print(f"\n❌ Lỗi tại batch {batch_num}: {e}")
            print("Tiếp tục với batch tiếp theo...")
            continue

    # ============================================================
    # HOÀN THÀNH
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH INGESTION!")
    print("=" * 60)
    print(f"📊 Tổng kết:")
    print(f"   • Tổng số documents: {len(documents)}")
    print(f"   • Tổng số chunks: {len(chunks)}")
    print(f"   • Index name: {PINECONE_INDEX_NAME}")
    print(f"   • Embedding model: {EMBEDDING_MODEL}")
    print(f"   • Dữ liệu cũ: ĐÃ XÓA")
    print(f"   • Dữ liệu mới: ĐÃ UPLOAD")
    print("=" * 60)


if __name__ == "__main__":
    main()