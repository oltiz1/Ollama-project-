from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df1 = pd.read_csv("file.csv")
df2 = pd.read_csv("food-prize-september-2023.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db_2"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df2.iterrows():
        document = Document(
            page_content=str(row["Period"]) + " " + str(row["Data_value"]),
            metadata={
                "product": row["Series_reference"],
                "price": row["Data_value"],
                "period": row["Period"]
            },
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
        