import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

PINECONE_INDEX_NAME = "rag-do-an"
DATA_DIR = "data/"
EMBEDDING_MODEL = "models/text-embedding-004"
VECTOR_DIMENSION = 768


def main():
    print("Ingesting...")


    load_dotenv()
    google_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key or not pinecone_api_key:
        print("missing api key in .env")
        return

    print(f"reading {DATA_DIR}...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        print("found nothin")
        return

    print(f"done load {len(documents)} file")

    print("chunking")

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
    print(f" split {len(chunks)} chunks")


    if len(chunks) > 0:
        print(
            f"\n--- preview ---\n{chunks [ 0 ].page_content [ :200 ]}...\n------------------------------------------\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key
    )


    print(f" Đang đẩy dữ liệu lên Pinecone Index: '{PINECONE_INDEX_NAME}'...")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    batch_size = 10
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):

        batch = chunks [ i: i + batch_size ]
        print(f" Đang nạp gói {i // batch_size + 1} (từ chunk {i} đến {min(i + batch_size, total_chunks)})...")
        try:
            vectorstore.add_documents(batch)
            time.sleep(3)

        except Exception as e:
            print(f"❌ Lỗi khi nạp gói tại vị trí {i}: {e}")
            break

    print("Done")
main()