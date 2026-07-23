from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    # MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    lines = [line.strip() for line in f if line.strip()]

for i, line in enumerate(lines):
    role='human' if i % 2 == 0 else 'ai'
    chat_history.append((role,line))

# print(chat_history)

prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query':'Where is my refund'
})

print(prompt)