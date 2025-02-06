from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from utils.func import create_vector_database

doc_path = ".\data\India-Metro-Systems-Report.pdf"

## Initiate the LLM model
model = "deepseek-r1:1.5b"

if doc_path: 
    loader = PyMuPDFLoader(file_path=doc_path)
    documents = loader.load()
    print(f"Number of documents: {len(documents)} in this PDF file: {doc_path}")

## print the content 
content = documents[10]
print(content)

## Chunking the PDF documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents)
print("Chunking of documents completed")
print(f"Documents splitted into {len(chunks)} chunks")

## Add these chunks to Vector Database
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_db = create_vector_database(chunks, embedding_model, name="vector-db")