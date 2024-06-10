from flask import Flask, request, jsonify
import os
import secrets
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

groq_api_key = os.getenv('GROQ_KEY')
model_name = 'llama3-70b-8192'
port = int(os.getenv('PORT', 5000))

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a storyteller. You speak only English!"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

conversation = LLMChain(
    llm=groq_chat,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

def get_chat_response(user_input):
    response = conversation.predict(human_input=user_input)
    return response

@app.route('/generate_story', methods=['POST'])
def generate_story():
    data = request.form
    custom_input = data.get('customInput')
    story_type = data.get('storyType')
    reader_age = data.get('readerAge')
    writing_style = data.get('writingStyle')

    user_input = (
        f"Please generate a story. Story type is {story_type}. Reader age: {reader_age}. "
        f"And writing style should be {writing_style}. The story idea is: {custom_input}. "
        "Move instantly to the story part, skip any unnecessary introduction parts. "
        "Please return the result in the following JSON format: { \"title\": \"<title>\", \"story\": \"<story>\" } "
        "where 'title' is the created title for the story. Make this story long, at least 2000 characters, but completed."
    )
    response = get_chat_response(user_input)
    print(response)

    return jsonify(story=response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
