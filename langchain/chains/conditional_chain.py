from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='zai-org/GLM-5.2'
)

model = ChatHuggingFace(llm=llm, max_tokens=2048)

parser = StrOutputParser()

class FeedBack(BaseModel):
    sentiment: Literal['pos', 'neg'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'pos', prompt2 | model | parser),
    (lambda x: x['sentiment'].sentiment == 'neg', prompt3 | model | parser),
    RunnableLambda(lambda x: "Couldn't find sentiment")
)

chain = RunnablePassthrough.assign(sentiment = classifier_chain) | branch_chain


print(chain.invoke({'feedback': 'This is wrost phone'}))

