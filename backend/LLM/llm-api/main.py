import time
import asyncio
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# PROMPT_TEMPLATE = """
# I'll give you some information/context about a topic. Answer the question based on the given information.

# {context}

# Answer the following question based on the information above:
# {question} 
# """

PROMPT_TEMPLATE = """
You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation. Follow these instructions carefully:

1. **For Physics Questions**:
   If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook. Your answer **must be based on this retrieved content** and no external knowledge should be used unless explicitly instructed.
   render mathematical formulas if needed

   - **Retrieved Context**:
   {context}

   Based on the above context, provide a clear and accurate answer to the user's query. If the context doesn't fully answer the question, mention that you don't have enough information to provide a complete answer.

2. **For General Conversation**:
   If the user asks a question or makes a statement unrelated to physics, generate a conversational and engaging response based on your general knowledge. Avoid using any context that was retrieved.

3. **Uncertainty Handling**:
   If the retrieved context or your general knowledge does not contain sufficient information, kindly inform the user instead of guessing or generating an incorrect answer.

### User's Query:
{input}

Respond appropriately based on the instructions above.
"""

PROMPT_TEMPLATE_TEST = """
You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation. Follow these instructions carefully:

1. **For Physics Questions**:
   If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook. Your answer **must be based on this retrieved content** and no external knowledge should be used unless explicitly instructed.
   Render mathematical formulas if needed.

   - **Retrieved Context**:
   {context}

   Based on the above context, provide a clear and accurate answer to the user's query. If the context doesn't fully answer the question, mention that you don't have enough information to provide a complete answer.

2. **For General Conversation**:
   If the user asks a question or makes a statement unrelated to physics, generate a conversational and engaging response based on your general knowledge. Avoid using any context that was retrieved.

3. **Uncertainty Handling**:
   If the retrieved context or your general knowledge does not contain sufficient information, kindly inform the user instead of guessing or generating an incorrect answer.

### User's Query:
{input}

Respond appropriately based on the instructions above.
"""

collection_name = {
    "nomic-embed-text": "physics_fsc_net",
    "chroma/all-minilm-l6-v2-f32": "physics_fsc_allmini",
    "mxbai-embed-large": "physics_fsc_mel"
}


chat_history = []


class AskQuestion(BaseModel):
    question: str
    llm_model: str
    embedding_model: str

app = FastAPI()

@app.get("/")
async def root():
    return "running successfully"

@app.post("/ask_question")
async def ask_question(req: AskQuestion):
    """
    Ask a question to the chatbot
    params:
    - question: str: The question to ask
    - llm_model: str: The LLM model to use
    - embedding_model: str: The embedding model to use
    returns a dict containing the model response, db query time and model invoke time
    """
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    question = req.question
    model_name = req.llm_model
    embedding_model_name = req.embedding_model
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
    return {"model_response": response, "db_query_time": (dbq_end - dbq_start), "model_invoke_time": (model_invoke_end - model_invoke_start)}

@app.post("/ask_question_v2")
async def ask_question_v2(req: AskQuestion):
    """
    Ask a question to the chatbot and get context aware response
    params:
    - question: str: The question to ask
    - llm_model: str: The LLM model to use
    - embedding_model: str: The embedding model to use
    returns a dict containing the model response, db query time and model invoke time
    """
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    question = req.question
    model_name = req.llm_model
    embedding_model_name = req.embedding_model
    model = OllamaLLM(model=model_name)
    collection  = Chroma(client=chroma_client, collection_name=collection_name[embedding_model_name], embedding_function=OllamaEmbeddings(model=embedding_model_name), create_collection_if_not_exists=False)
    retriever = collection.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},
    )
    retriever_promt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                PROMPT_TEMPLATE_TEST,
            )
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_promt
    )
    # raw_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    full_chat_history = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in chat_history])
    PROMPT_TEMPLATE_TEST_TEMP = PROMPT_TEMPLATE_TEST
    PROMPT_TEMPLATE_TEST_TEMP = f"Here is the chat history between you(assistant) and me(human)\n### Chat history:\n{full_chat_history}\n{PROMPT_TEMPLATE_TEST_TEMP}"
    raw_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_TEST_TEMP)

    document_chain = create_stuff_documents_chain(model, raw_prompt)

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    result = retrieval_chain.invoke({"input": question})

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))
    print(chat_history)

    return {"model_response": result["answer"]}


@app.post("/ask_question_v2")
async def ask_question_v2(req: AskQuestion):
    """
    Ask a question to the chatbot and get context aware response
    params:
    - question: str: The question to ask
    - llm_model: str: The LLM model to use
    - embedding_model: str: The embedding model to use
    returns a dict containing the model response, db query time and model invoke time
    """
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    question = req.question
    model_name = req.llm_model
    embedding_model_name = req.embedding_model
    model = OllamaLLM(model=model_name)
    collection  = Chroma(client=chroma_client, collection_name=collection_name[embedding_model_name], embedding_function=OllamaEmbeddings(model=embedding_model_name), create_collection_if_not_exists=False)
    retriever = collection.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},
    )
    retriever_promt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                PROMPT_TEMPLATE_TEST,
            )
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_promt
    )
    # raw_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    full_chat_history = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in chat_history])
    full_chat_history = model.invoke(f"Here is a chat history between you(assistant) and me(human)\n### Chat history:\n{full_chat_history}\nCan you summarize it and include the key information and keep the human questions as is?")
    PROMPT_TEMPLATE_TEST_TEMP = PROMPT_TEMPLATE_TEST
    PROMPT_TEMPLATE_TEST_TEMP = f"Here is the chat history between you(assistant) and me(human)\n### Chat history:\n{full_chat_history}\n{PROMPT_TEMPLATE_TEST_TEMP}"
    raw_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_TEST_TEMP)

    document_chain = create_stuff_documents_chain(model, raw_prompt)

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    result = retrieval_chain.invoke({"input": question})

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))
    print(chat_history)

    return {"model_response": result["answer"]}

