from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma

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
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(persist_directory = r"db\chroma-vector-db-(rude-luminosity)", embedding_function=embedding_model)
query = "How many metro stations are there in India?"

results = vector_db.similarity_search(query, k=5)
print(len(results))


# retriever = MultiQueryRetriever.from_llm(
#     retriever = vector_db.as_retriever(search_kwargs={"k": 10}),
#     llm = llm
# )

# query = "How many metro stations are there in India?"


# ## Rag prompt
# template = """
# Answer the question based ONLY on the following context: 
# {context}
# Question:
# {question}
# """

# query = "How many metro stations are there in India?"

# prompt = ChatPromptTemplate.from_template(template)
# print(prompt)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

