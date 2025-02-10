import randomname
import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from glob import glob
import logging 
import os

DOC_PATH = "data\India-Metro-Systems-Report.pdf"
MODEL_NAME = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_COLLECTION_NAME = "vector-db"
EMBEDDING_MODEL = "nomic-embed-text"

logging.basicConfig(
    format = "%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def document_loader(doc_path):

    try:
        loader = PyMuPDFLoader(file_path=doc_path)
        documents = loader.load()
        logging.info(f"Number of documents: {len(documents)} in this PDF file: {doc_path}")

        return documents
    except Exception as e:
        logging.error(f"There was some issue in reading the documents \nMessage:{e}")
        return None
    

def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    logging.info(f'The documents splitter into {len(chunks)} chunks')

def create_vector_database(chunks: list, embeddings, name="vector-db"):

    os.makedirs("db", exist_ok=True)

    random_suffix = randomname.get_name()

    persistent_directory = f"db/chroma-{name}-({random_suffix})"

    ## If already there, delete and create a new one

    for folder in glob(f"db/chroma-{name}*"):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.mkdir(persistent_directory)

    vector_db = Chroma.from_documents(
        documents = chunks,
        collection_name = name,
        embedding = embeddings,
        persist_directory = persistent_directory
    )

    return vector_db    

def load_vector_db(path):
    if os.path.exists(path):

        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

        vector_db = Chroma(
            persist_directory=path,
            collection_name=VECTOR_STORE_COLLECTION_NAME,
            embedding_function = embedding
        )

        logging.info(f"Successfully loaded the existing vector database: {VECTOR_STORE_COLLECTION_NAME}")
        return vector_db

    else:
        logging.info("Vector Database Not Found, creating a new vector database")
        return None


def create_retriever(vector_db, llm):

    '''
    Given a query, use an LLM to write a set of queries.
    Retrieve docs for each query. Return the unique union of all retrieved docs
    '''

    QUERY_PROMPT = PromptTemplate(
        input_variables = ['question'],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        
    )