import shelve
import time
import asyncio
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_core.documents import Document
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

def create_session_context_db():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    model = OllamaLLM(model="llama3.2:latest")
    try:
        chroma_client.delete_collection("session_context_db")
        print("Deleted session context db")
    except:
        pass
    collection  = Chroma(client=chroma_client, collection_name="session_context_db", 
                         embedding_function=OllamaEmbeddings(model="nomic-embed-text"), create_collection_if_not_exists=True)
    collection.reset_collection()

create_session_context_db()

class AskQuestion(BaseModel):
    question: str
    llm_model: str
    embedding_model: str

class ConversationContext(BaseModel):
    conversation: str

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
    # print(f"Using collection: {collection_name[embedding_model_name]}, embedding model: {embedding_model_name}, llm model: {model_name}")
    # print(f"Database query time: {(dbq_end - dbq_start):.6f} seconds")
    # print("====================================")
    # print(results)
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
                PROMPT_TEMPLATE,
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
    PROMPT_TEMPLATE_TEMP = PROMPT_TEMPLATE
    PROMPT_TEMPLATE_TEMP = f"Here is the chat history between you(assistant) and me(human)\n### Chat history:\n{full_chat_history}\n{PROMPT_TEMPLATE_TEMP}"
    raw_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_TEMP)

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


@app.post("/ask_question_enhanced")
async def ask_question_enhanced(req: AskQuestion):
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
    temp = f"Classify the following user input as a physics related question or conversation context or general conversation. You should give me only the classification result \
        from this list: ['physics', 'context', 'general', 'uncertain'] based on the user input. user input: {question}"
    class_result = model.invoke(temp)
    # print(question)
    # print(f"Classification result: {class_result}")
    session_context = Chroma(client=chroma_client, collection_name="session_context_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"), create_collection_if_not_exists=False)
    ssdb_results = session_context.similarity_search(query=question, k = 10)
    # ssdb_results.sort(key=lambda x: x.id, reverse=False)
    relevant_context = "\n\n".join([doc.page_content for doc in ssdb_results])
    if class_result.count("general") or class_result.count("uncertain"):
        prompt = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {relevant_context}. You must ignore the contexts for this prompt for most cases. Generate a brief general response\
                  to the following user input. Also ask them if they need help about\
              physics related question. user input: {question}"
        result = model.invoke(prompt)
        return {"model_response": result}
    elif class_result=="context":
        prompt = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {relevant_context}. Generate a brief response\
                  to the following user input.Use information from the context if needed. user input: {question}"
        result = model.invoke(prompt)
        return {"model_response": result}
    else:
        results = collection.similarity_search(query=question, k = 5)
        context = "\n\n".join([doc.page_content for doc in results])
        # prompt  = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
        #     . Here is some contexts from the conversation between you and\
        #     the user, context: {relevant_context}.Use information from the context if needed.\
        #          If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook\
        #             . Your answer **must be based on this retrieved content** and no external knowledge should be used unless explicitly instructed.\
        #                 render mathematical formulas with latex if needed. Retrieved context: {context}\n user input: {question}"
        prompt  = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {relevant_context}.Use information from the context if needed.\
                 If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook\
                    . You can use this information to get document help to answer the question.\
                        render mathematical formulas with latex if needed. Retrieved documents: {context}\n user input: {question}"
        response = model.invoke(prompt)
        return {"model_response": response}
    
@app.post("/ask_question_enhanced_v2")
async def ask_question_enhanced_v2(req: AskQuestion):
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
    temp = f"Classify the following user input as a physics related question or conversation context or general conversation. You should give me only the classification result \
        from this list: ['physics', 'context', 'general', 'uncertain'] based on the user input. user input: {question}"
    class_result = model.invoke(temp)
    # print(question)
    # print(f"Classification result: {class_result}")
    session_context = Chroma(client=chroma_client, collection_name="session_context_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"), create_collection_if_not_exists=False)
    ssdb_results = session_context.similarity_search(query=question, k = 10)
    # ssdb_results.sort(key=lambda x: x.id, reverse=False)
    relevant_context = "\n\n".join([doc.page_content for doc in ssdb_results])
    # raw_context
    ch_path = "E:\\Physics-Chatbot\\frontend\\chat_history"
    with shelve.open(ch_path) as ch:
        messages = ch.get("messages", [])

    raw_context="";
    for message in messages:
        if message["role"]=="user":
            raw_context += f"User: {message['content']}\n"
        else:
            raw_context += f"Assistant: {message['content']}\n"

    if class_result.count("general") or class_result.count("uncertain"):
        prompt = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {raw_context}. You must ignore the contexts for this prompt for most cases. Generate a brief general response\
                  to the following user input. Also ask them if they need help about\
              physics related question. user input: {question}"
        result = model.invoke(prompt)
        return {"model_response": result}
    elif class_result=="context":
        prompt = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {raw_context}. Generate a brief response\
                  to the following user input.Use information from the context if needed. user input: {question}"
        result = model.invoke(prompt)
        return {"model_response": result}
    else:
        results = collection.similarity_search(query=question, k = 5)
        context = "\n\n".join([doc.page_content for doc in results])
        # prompt  = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
        #     . Here is some contexts from the conversation between you and\
        #     the user, context: {relevant_context}.Use information from the context if needed.\
        #          If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook\
        #             . Your answer **must be based on this retrieved content** and no external knowledge should be used unless explicitly instructed.\
        #                 render mathematical formulas with latex if needed. Retrieved context: {context}\n user input: {question}"
        prompt  = f"You are a helpful assistant specialized in answering physics-related questions and engaging in general conversation\
            . Here is some contexts from the conversation between you and\
            the user, context: {raw_context}.Use information from the context if needed.\
                 If the user asks a physics-related question, you will be provided with relevant information retrieved from a physics textbook\
                    . You can use this information to get document help to answer the question.\
                        render mathematical formulas with latex if needed. Retrieved documents: {context}\n user input: {question}"
        response = model.invoke(prompt)
        return {"model_response": response}

@app.post("/save_session_context")
async def save_session_context(convo : ConversationContext):
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection  = Chroma(client=chroma_client, collection_name="session_context_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"), create_collection_if_not_exists=False)
    cnt = len(collection.get()['documents'])
    doc = Document(page_content=convo.conversation, metadata={}, id=f"session_context_{cnt}")
    collection.add_documents(documents=[doc], ids=[f"session_context_{cnt}"])
    return {"status": "saved"}

@app.post("/reset_session_context")
async def reset_session_context():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection  = Chroma(client=chroma_client, collection_name="session_context_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"), create_collection_if_not_exists=False)
    collection.reset_collection()
    return {"status": "reset"}
