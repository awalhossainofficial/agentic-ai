from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = OpenAIEmbeddings(
     model="text-embedding-3-large",
     dimensions=32,
)

docs = [
    "Hello world, Dhaka is the capital of Bangladesh.",
    "Pabna is a district in the Rajshahi Division of Bangladesh.",
    "Dhaka is located in the central part of Bangladesh.",
    "Barisal is a district in the Barisal Division of Bangladesh.",
    "Chittagong is a district in the Chittagong Division of Bangladesh.",
    "Khulna is a district in the Khulna Division of Bangladesh.",
    "Mymensingh is a district in the Mymensingh Division of Bangladesh.",
    "Rajshahi is a district in the Rajshahi Division of Bangladesh.",
    "Rangpur is a district in the Rangpur Division of Bangladesh.",
    "Sylhet is a district in the Sylhet Division of Bangladesh.",
    "In Barishal, the main river is the Kirtankhola River.",
    "In Chittagong, the main river is the Karnaphuli River.",
    "In Dhaka, the main river is the Buriganga River.",
    "In Khulna, the main river is the Rupsha River.",
    "In Mymensingh, the main river is the Brahmaputra River.",
    "In Pabna, Padma River used to be the biggest.",
    "In Rajshahi, the main river is the Padma River.",
    "In Rangpur, the main river is the Teesta River.",
    "In Sylhet, the main river is the Surma River."
]

query = "Which one is the biggest river in Bangladesh?"

doc_embeddings = embeddings.embed_documents(docs)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]


print(scores)
print(f"Most similar document: {docs[index]}")