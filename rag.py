from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SQLiteVSS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import os


class TextChunk:
    def __init__(self, text: str, metadata: dict):
        self.text: str = text
        self.metadata: dict = metadata

    def __repr__(self):
        return f"TextChunk(text='{self.text[:50]}...', metadata={self.metadata})"


def is_handwritten(file_path: str) -> bool:
    return False


def transcribe_handwritten(file_path: str) -> Document:
    text_content = f"Mock OCR for {file_path}"
    return Document(page_content=text_content, metadata={"source": file_path, "type": "handwritten_ocr"})


def load_remarkable_documents(directory_path: str) -> list[Document]:
    all_docs: list[Document] = []
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}. Please create it and add files.")
        return all_docs

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                if is_handwritten(file_path):
                    print(f"Processing handwritten file (mock): {file_path}")
                    docs = transcribe_handwritten(file_path)
                    all_docs.extend(docs)
                elif file_path.lower().endswith(".pdf"):
                    print(f"Processing PDF file: {file_path}")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()  # Returns a list of Document objects
                    for d in docs:
                        d.metadata["source"] = "/".join(d.metadata["source"].split("/")[1:])

                        d.metadata.pop("producer", None)
                        d.metadata.pop("creator", None)
                        d.metadata.pop("creationdate", None)
                    all_docs.extend(docs)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return all_docs


def split_into_chunks(docs: list[Document]) -> list[TextChunk]:
    print("Splitting docs into chunks")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    return [TextChunk(doc.page_content, doc.metadata) for doc in docs]


DB_FILE_PATH = "vss.db"
DB_TABLE_NAME = "notebook"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def create_embeddings(chunks: list[TextChunk]):
    print("Creating embeddings")

    if os.path.exists(DB_FILE_PATH):
        os.remove(DB_FILE_PATH)

    db = SQLiteVSS.from_texts(
        texts=[c.text for c in chunks],
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
        table=DB_TABLE_NAME,
        db_file=DB_FILE_PATH,
    )


def get_db():
    # create the open-source embedding function
    connection = SQLiteVSS.create_connection(db_file=DB_FILE_PATH)

    db1 = SQLiteVSS(table=DB_TABLE_NAME, embedding=embeddings, connection=connection)
    return db1


if __name__ == "__main__":
    docs = load_remarkable_documents("my_mock_backup")
    chunks = split_into_chunks(docs)
    for c in chunks:
        print(c.metadata)
        print("==========================")
    create_embeddings(chunks)
