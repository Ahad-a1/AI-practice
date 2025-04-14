
import os
import getpass
from dotenv import load_dotenv

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get Mistral AI API key from env or prompt
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        api_key = getpass.getpass("🔐 Enter API key for Mistral AI: ")

    # Load PDF path from env
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        print("⚠️ PDF_PATH not found in environment.")
        exit(1)

    # Initialize Mistral AI embedding model (semantic embedding)
    embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")

    # Load documents from PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        print("⚠️ No content loaded from PDF.")
        exit(1)

    # Chunk the documents semantically
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    # Create vector store (semantic search backend)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    # Perform semantic similarity search
    query = "What is Math"
    results = vector_store.similarity_search(query)

    # Display the best semantic match
    if results:
        print("\n🔍 Top semantic match:")
        print(results[0].page_content)
    else:
        print("No results found for your query.")
