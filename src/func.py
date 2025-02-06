import randomname
import os
import shutil
from langchain_community.vectorstores import Chroma
from glob import glob

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