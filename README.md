# Physics-Chatbot

Physics Chatbot is a chatbot that can answer questions about physics ( specifically, topics from the Class 9-10 NCTB Physics Textbook ).
It uses RAG to retrieve relvent information/document from a vector database and uses locally run LLM model to answer questions.

Technologies used: <br> - LLM Host: Ollama 
<br> - Language Models: llama3.2, gemma2, mistral etc.
<br> - Vector Database: Chroma
<br> - Embedding Models: all-MiniLM-L6-v2, nomic-embed-text etc.
<br> - Streamlit, FastAPI, Langchain etc.

## Setup
### Prerequisites
install python version <= 3.12 (recommended)

create a virtual environment and activate it

`python3.12 -m venv venv`

install pip packages from requirements.txt

`source venv/bin/activate`

`pip install -r requirements.txt`

### install  Ollama
https://ollama.com/

install these llm models: llama3.2:latest, gemma2:latest, mistral:latest

and these embedding models: nomic-embed-text, all-MiniLM-L6-v2, mxbai-embed-large

### Run the Chatbot

run main.py to run the chatbot

`python main.py`

