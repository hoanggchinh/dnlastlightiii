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
from langchain_core.documents import Document

PINECONE_INDEX_NAME = "rag-do-an"
DATA_DIR = "data/"
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DIMENSION = 768


def load_documents_from_directory(directory):

    all_documents = [ ]

    print(f"📂 Đang quét thư mục: {directory}")

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
        print(f"Loaded {len(txt_docs)} TXT files")
    except Exception as e:
        print(f"Error loading TXT: {e}")


    print("Loading .pdf files...")
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
        print(f"Loaded {len(pdf_docs)} PDF files")
    except Exception as e:
        print(f"Error loading PDF: {e}")

    print("Loading .docx files...")
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
        print(f"Loaded {len(docx_docs)} DOCX files")
    except Exception as e:
        print(f"Error loading DOCX: {e}")

    return all_documents


def smart_chunk_documents(documents):


    separators = [

        r"(?<=\n)━{10,}",
        r"(?<=\n)={10,}",

        r"(?<=\n)#{1,3}\s+",
        r"(?<=\n)Chương\s+[IVX0-9]+",
        r"(?<=\n)Phần\s+[IVX0-9]+",
        r"(?<=\n)Điều\s+\d+",

        r"(?<=\n)[0-9]+\.\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆ]",
        "\n\n",
        r"(?<=\n)[-•●○]\s+",
        "\n",
        r"[.!?]\s+",
        " "
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500,
        separators=separators,
        is_separator_regex=True,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    cleaned_chunks = [ ]
    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) > 50:
            cleaned_chunks.append(chunk)

    return cleaned_chunks


def enhance_chunk_with_context(chunks):

    enhanced_chunks = [ ]

    for i, chunk in enumerate(chunks):

        title = ""
        for line in chunk.page_content.split('\n') [ :3 ]:
            line_upper = line.strip()
            if len(line_upper) > 5 and (
                    line_upper.isupper() or
                    line.startswith('#') or
                    '━' in line or
                    '=' in line
            ):
                title = line_upper.replace('#', '').replace('━', '').replace('=', '').strip()
                break

        metadata = {
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_id": i,
        }

        if title:
            metadata [ "title" ] = title

        if "page" in chunk.metadata:
            metadata [ "page" ] = chunk.metadata [ "page" ]

        if i > 0:
            prev_snippet = chunks [ i - 1 ].page_content [ -100: ].strip()
            metadata [ "previous_context" ] = prev_snippet

        enhanced_doc = Document(
            page_content=chunk.page_content,
            metadata=metadata
        )
        enhanced_chunks.append(enhanced_doc)

    return enhanced_chunks


def main():
    print("=" * 60)
    print("RAG INGESTION PIPELINE - TNUT CHATBOT (ENHANCED)")
    print("=" * 60)

    load_dotenv()
    google_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("Missing API keys in .env file")
        return


    print(f"\nĐang xóa dữ liệu cũ trong index '{PINECONE_INDEX_NAME}'...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        existing_indexes = pc.list_indexes().names()

        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Index '{PINECONE_INDEX_NAME}' không tồn tại!")
            return

        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        print(f"Đã xóa toàn bộ dữ liệu cũ!")
        time.sleep(3)
    except Exception as e:
        print(f"Lỗi khi xóa dữ liệu: {e}")
        return

    documents = load_documents_from_directory(DATA_DIR)

    if not documents:
        print("Không tìm thấy file nào!")
        return

    print(f"\nTổng số documents: {len(documents)}")

    print("\n Đang chia nhỏ documents bằng thuật toán thông minh...")
    chunks = smart_chunk_documents(documents)
    print(f"Đã tạo {len(chunks)} chunks")

    print("\n Đang thêm context metadata...")
    enhanced_chunks = enhance_chunk_with_context(chunks)
    print(f"Đã enhance {len(enhanced_chunks)} chunks")

    if enhanced_chunks:
        print("\n" + "=" * 60)
        print("📋 PREVIEW 3 CHUNKS ĐẦU TIÊN:")
        print("=" * 60)
        for i in range(min(3, len(enhanced_chunks))):
            chunk = enhanced_chunks [ i ]
            print(f"\n--- CHUNK {i + 1} ---")
            print(f"Content: {chunk.page_content [ :200 ]}...")
            print(f"Metadata: {chunk.metadata}")
        print("=" * 60)

    print(f"\nKhởi tạo embedding model: {EMBEDDING_MODEL}...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key
    )

    print(f"\n Kết nối tới Pinecone index: '{PINECONE_INDEX_NAME}'...")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # UPLOAD
    print(f"\n Đang upload {len(enhanced_chunks)} chunks lên Pinecone...")
    print("=" * 60)

    batch_size = 10
    total_chunks = len(enhanced_chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    for i in range(0, total_chunks, batch_size):
        batch = enhanced_chunks [ i:i + batch_size ]
        batch_num = i // batch_size + 1

        print(f"Batch {batch_num}/{total_batches} (chunks {i + 1}-{min(i + batch_size, total_chunks)})...", end=" ")

        try:
            vectorstore.add_documents(batch)
            print("")
            time.sleep(2)
        except Exception as e:
            print(f"\n Lỗi tại batch {batch_num}: {e}")
            continue

    # SUMMARY
    print("\n" + "=" * 60)
    print(" HOÀN THÀNH INGEST!")
    print("=" * 60)
    print(f" Tổng kết:")
    print(f"   • Tổng số documents: {len(documents)}")
    print(f"   • Tổng số chunks: {len(enhanced_chunks)}")
    print(f"   • Chunk size: 2500 chars")
    print(f"   • Chunk overlap: 500 chars")
    print(f"   • Index name: {PINECONE_INDEX_NAME}")
    print(f"   • Embedding model: {EMBEDDING_MODEL}")
    print("=" * 60)


if __name__ == "__main__":
    main()