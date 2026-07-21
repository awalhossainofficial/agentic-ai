from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Hello world, Dhaka is the capital of Bangladesh.",
    "Pabna is a district in the Rajshahi Division of Bangladesh."
    "Dhaka is located in the central part of Bangladesh."
]

embeddings = OpenAIEmbeddings(
     model="text-embedding-3-large",
     dimensions=32,
)

result = embeddings.embed_documents(docs)

print(str(result))