from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Hello world, Dhaka is the capital of Bangladesh.",
    "Pabna is a distr     ict in the Rajshahi Division of Bangladesh.",
    "Dhaka is located in the central part of Bangladesh."
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"},
)

result = embeddings.embed_documents(docs)

print(result)