from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from func import create_vector_database

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import time

start_time = time.time()

## Load env variables : langsmith
load_dotenv()  # take environment variables from `.env`.


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

print("Creating vector database")
vector_db = create_vector_database(chunks, embedding_model, name="vector-db")
print("Vector database successfully created")


query = "How many metro stations are there in India?"
# results = vector_db.similarity_search(query, k=5)
# print(type(vector_db))
# print(len(results))

## Load the model
model = "deepseek-r1:1.5b" 
llm = ChatOllama(model=model)

## Create the multiquery retriever prompt
query_prompt = PromptTemplate(
    input_variables = ['question'],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

## Initiate the MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever = vector_db.as_retriever(),
    llm = llm,
    prompt = query_prompt
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "How many metro stations are there in India?"
response = chain.invoke(query)
print(response)

end_time = time.time() - start_time
print(f"Time taken: {end_time:.2f} seconds")
