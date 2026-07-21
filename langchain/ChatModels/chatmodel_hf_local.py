from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# os.environ['HF_HOME'] = 'D:/huggign'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs=dict(
        temperature = 0.7,
        max_new_tokens=500
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Do you know anything about pabna? ")

print(result.content)