from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt-4o-mini")

result = llm.invoke("who is the prime minister of the bangladesh? just give me the answer, no explanation")

print(result)