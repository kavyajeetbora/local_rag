from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma


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

vector_db = Chroma(persist_directory = r"db\chroma-vector-db-(rude-luminosity)")
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=query_prompt
)


