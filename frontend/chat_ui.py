import time
import requests
import streamlit as st
import shelve

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

MODEL_LIST = ["llama3.2:latest", "llama3.1:latest", "mistral:latest", "gemma2:latest"] 
EMBEDDING_MODEL_LIST = ["nomic-embed-text", "chroma/all-minilm-l6-v2-f32", "mxbai-embed-large"]

API_BASE_URL = "http://localhost:8080"

def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

def load_current_llm_model():
    with shelve.open("current_model") as db:
        st.session_state.llm = db.get("llm", "llama3.2:latest")

def load_current_embedding_model():
    with shelve.open("current_model") as db:
        st.session_state.embedding = db.get("embedding", "nomic-embed-text")

def save_current_model_names(llm_, embedding_):
    with shelve.open("current_model") as db:
        db["llm"] = llm_
        db["embedding"] = embedding_


def change_model():
    st.session_state.llm = st.session_state.model_selection

def change_embedding():
    st.session_state.embedding = st.session_state.embedding_selection

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "llm" not in st.session_state:
    load_current_llm_model()

if "embedding" not in st.session_state:
    load_current_embedding_model()

# math_render = """
# <script type="text/javascript" async
#   src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
# </script>
# """

math_render = """
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script>
function renderMath() {
  MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
}
</script>
"""

# hide the deploy button
st.markdown(
        r"""
        <style>
        .stAppDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )



st.title("Physics Chatbot")
st.write("This is a chatbot that can answer questions related to your physics textbook. You can ask questions about physics and the chatbot will try to answer them.")

with st.sidebar:
    model_option = st.selectbox("Select a LLM", MODEL_LIST, index=MODEL_LIST.index(st.session_state.llm), key="model_selection")
    st.write(f"Selected : **{model_option}**")
    embedding_option = st.selectbox("Select an Embedding", EMBEDDING_MODEL_LIST, index=EMBEDDING_MODEL_LIST.index(st.session_state.embedding), key="embedding_selection")
    st.write(f"Selected : **{embedding_option}**")
    if st.button("Change Model & Embedding"):
        change_model()
        change_embedding()
    
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        requests.post(f"{API_BASE_URL}/reset_session_context")
        

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(math_render + message["content"], unsafe_allow_html=True)


if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        payload = {
            "question": prompt,
            "llm_model": st.session_state.llm,
            "embedding_model": st.session_state.embedding
        }
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_BASE_URL}/ask_question_enhanced_v2", json=payload)
            response = response.json()
            model_response = response["model_response"]
        # message_placeholder.markdown(response)
        type_speed = 0.01
        model_response_latex = math_render + model_response
        for i in range(len(model_response_latex)):
            message_placeholder.markdown(model_response_latex[:i+1], unsafe_allow_html=True)
            time.sleep(type_speed)
        st.components.v1.html(
            """
            <script>
            renderMath();
            </script>
            """,
            height=0,  # Set height to 0 since it's just a trigger
        )
        # st.success(f"db query time: {response['db_query_time']:.4f} seconds, model invoke time: {response['model_invoke_time']:.4f} seconds", icon="⏱️")
        # st.caption(f"db query time: {response['db_query_time']:.4f} seconds, model invoke time: {response['model_invoke_time']:.4f} seconds")
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs")
        if selected is not None:
            st.markdown(f"You selected: {sentiment_mapping[selected]}")    
    st.session_state.messages.append({"role": "assistant", "content": model_response})
    this_interaction = f"USER : {prompt}\nASSISTANT : {model_response}"
    requests.post(f"{API_BASE_URL}/save_session_context", json={"conversation": this_interaction})

save_chat_history(st.session_state.messages)
save_current_model_names(st.session_state.llm, st.session_state.embedding)



