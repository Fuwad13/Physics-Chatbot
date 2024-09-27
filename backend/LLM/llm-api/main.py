import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

PROMPT_TEMPLATE = """
I'll give you some information/context about a topic. Answer the question based on the given information.
{context}


Answer the following question based on the information above:
{question} 
"""

class Question(BaseModel):
    text: str

app = FastAPI()

@app.get("/")
async def root():
    return "running successfully"

@app.get("/ask_question")
async def ask_question(question: str):
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    # TODO : add ability to select different embeddings/ llm models
    collection = Chroma(client=chroma_client, collection_name='physics_fsc_net', embedding_function=OllamaEmbeddings(model='nomic-embed-text'), create_collection_if_not_exists=False)
    model = OllamaLLM(model='mistral:latest')
    results = collection.similarity_search(query=question, k = 5)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt  = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=question)
    response = model.invoke(prompt)
    return {"response": response}