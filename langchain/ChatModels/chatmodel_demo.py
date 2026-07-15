from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model_name="gpt-4o-mini")

res = chat.invoke("what is the capital of bangladesh?")

print(res.content)