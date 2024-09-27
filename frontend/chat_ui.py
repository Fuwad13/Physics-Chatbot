import requests
import streamlit as st
from langchain_ollama import OllamaLLM
import shelve

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

MODEL_LIST = ["mistral:latest", "gemma2:latest", "llama3.1:latest"] 
EMBEDDING_MODEL_LIST = ["nomic-embed-text", "chroma/all-minilm-l6-v2-f32", "mxbai-embed-large"]

if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(model='mistral:latest')

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


def change_model():
    st.session_state.llm = OllamaLLM(model=st.session_state.model_selection)
    st.session_state.messages = []
    save_chat_history([])

def change_embedding():
    print(st.session_state.embedding_selection)

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

st.title("Physics Chatbot")
st.write("This is a chatbot that can answer questions related to your physics textbook. You can ask questions about physics and the chatbot will try to answer them.")

with st.sidebar:
    model_option = st.selectbox("Select a LLM", MODEL_LIST, index=0, key="model_selection")
    st.write(f"Selected : **{model_option}**")
    embedding_option = st.selectbox("Select an Embedding", EMBEDDING_MODEL_LIST, index=0, key="embedding_selection")
    st.write(f"Selected : **{embedding_option}**")
    if st.button("Change Model & Embedding"):
        change_model()
        change_embedding()
    
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        # full_response = st.session_state.llm.invoke(prompt)
        full_response = requests.get(f"http://localhost:8080/ask_question?question={prompt}").json()
        full_response = full_response["response"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)


