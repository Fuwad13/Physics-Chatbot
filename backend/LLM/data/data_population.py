import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


TEXTBOOK_PATH = "E:\\Physics-Chatbot\\backend\\LLM\\data\\Physics Classes 9-10 (English Version) - National Curriculum and Textbook Board of Bangladesh - PDF Room.pdf"


def main():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    documents = load_textbook()
    chunks = fixed_size_chunk(documents)
    

    # assigning unique id to each chunk
    cur_ind = 0
    prev_page = -1

    for chunk in chunks:
        page_no = chunk.metadata.get('page')
        if page_no != prev_page:
            cur_ind = 0
        else:
            cur_ind += 1
        cur_id = f"physics:{page_no}:{cur_ind}"
        chunk.metadata['id'] = cur_id
        # print(cur_id)
        prev_page = page_no
    
    docs = [doc.page_content for doc in chunks]
    metadatas = [ mdt.metadata for mdt in chunks]
    ids = [id.metadata.get('id') for id in chunks]

    batch_size = 100
    
    # WARNING: run these commented code only once, otherwise it will create duplicate data in the database

    # fixed size chunking, embedding: nomic-embed-text
    collection = Chroma(client=chroma_client, collection_name='physics_fsc_net', embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    for i in range(0, len(docs), batch_size):
        collection.add_documents(documents=chunks[i:i+batch_size], ids=ids[i:i+batch_size])


    # # fixed size chunking, embedding: chroma/all-minilm-l6-v2-f32
    collection = Chroma(client=chroma_client, collection_name='physics_fsc_allmini', embedding_function=OllamaEmbeddings(model='chroma/all-minilm-l6-v2-f32'))
    for i in range(0, len(docs), batch_size):
        collection.add_documents(documents=chunks[i:i+batch_size], ids=ids[i:i+batch_size])

    # # fixed size chunking, embedding: mxbai-embed-large
    collection = Chroma(client=chroma_client, collection_name='physics_fsc_mel', embedding_function=OllamaEmbeddings(model='mxbai-embed-large'))
    for i in range(0, len(docs), batch_size):
        collection.add_documents(documents=chunks[i:i+batch_size], ids=ids[i:i+batch_size])
    

def load_textbook():
    """
    Returns a list of `Document` objects, each representing a page of the textbook.
    """
    loader = PyPDFLoader(TEXTBOOK_PATH)
    
    return loader.load()

def fixed_size_chunk(documents: list[Document]):
    """
    Returns a list of Documents of fixed size chunks of the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)





def semantic_chunking(documents):
    """
    Returns a list of Documents of semantically chunked documents.
    """
    
    ...

if __name__ == "__main__":
    main()
