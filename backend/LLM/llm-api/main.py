import time
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

collection_name = {
    "nomic-embed-text": "physics_fsc_net",
    "chroma/all-minilm-l6-v2-f32": "physics_fsc_net",
    "mxbai-embed-large": "physics_fsc_net"
}


class AskQuestion(BaseModel):
    question: str
    model_name: str
    embedding_model_name: str

app = FastAPI()

@app.get("/")
async def root():
    return "running successfully"

@app.post("/ask_question")
async def ask_question(req: AskQuestion):
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    # TODO : add ability to select different embeddings/ llm models
    # collection = Chroma(client=chroma_client, collection_name='physics_fsc_net', embedding_function=OllamaEmbeddings(model='nomic-embed-text'), create_collection_if_not_exists=False)
    # model = OllamaLLM(model='mistral:latest')
    question = req.question
    model_name = req.model_name
    embedding_model_name = req.embedding_model_name
    model = OllamaLLM(model=model_name)
    collection  = Chroma(client=chroma_client, collection_name=collection_name[embedding_model_name], embedding_function=OllamaEmbeddings(model=embedding_model_name), create_collection_if_not_exists=False)
    dbq_start = time.time()
    results = collection.similarity_search(query=question, k = 5)
    dbq_end = time.time()
    print(f"Using collection: {collection_name[embedding_model_name]}, embedding model: {embedding_model_name}, llm model: {model_name}")
    print(f"Database query time: {(dbq_end - dbq_start):.6f} seconds")
    context = "\n\n".join([doc.page_content for doc in results])
    prompt  = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=question)
    model_invoke_start = time.time()
    response = model.invoke(prompt)
    model_invoke_end = time.time()
    print(f"Model invoke time: {(model_invoke_end - model_invoke_start):.6f} seconds")
    return {"model_response": response}