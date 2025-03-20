import os
import shutil
import argparse

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document


CHROMA_PATH = "chroma" #Directory  where db be stored
DATA_PATH = "data" # Directory containging PDF filfes

def main():
    '''organizes RAG Process'''

    # Check if db needs to be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset database")
    args = parser.parse_args()

    # Clear Data if reset flag is set
    if args.reset:
        print("Resetting database")
        clear_database()

    # create - update data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_chroma(chunks)

def load_documents():
    '''load all PDF documents from data directory
    - Output: list[Document] containing PDF contents'''
    # automatically laods all pdf from dir
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    '''splits large documents into smaller, chunks for better processing
    - Input: list[Document] - documents to split
    - Output: list[Document] - smaller chunks of documents'''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # size of each chunk in characters
        chunk_overlap=80, # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def get_embeddings():
    '''initialize embeddings model for vector encoding
    - Output: OllamaEmbeddings object'''
    embeddings = OllamaEmbeddings(model="deepseek-r1", # model we use
                                  base_url="http://localhost:11434")
    return embeddings

def add_chroma(chunks: list[Document]):
    '''add document chunks to chroma & avoid duplicate
    - Input: list[Document] - documents to add'''

    # initialize Chroma vector base
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())

    # generate unique ids for chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # get existing document ids from db
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing chunks in DB: {len(existing_ids)}")

    # only add into db if it doenst exust
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # add new chunks if any were found
    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)

    else:
        print(f"No new chunks found")

def calculate_chunk_ids(chunks):
    '''Tag each chunk with unique string ID based on source path, page number, chunk index:
        - Input: list[Document] - documents chunks without IDs
        - Output: list[Document] - same chunks with added ID in metadata
            example: "data/blablabla.pdf:8:3" '''


    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # extract data
        source = chunk.metadata['source']
        page = chunk.metadata['page']
        current_page_id = f"{source}_{page}"

        # reset chunk index for new pages
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # create unique ID for chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id # adds id to metadata
    return chunks

def clear_database():
    '''Delete entire Chroma Database dir. if exists'''
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == '__main__':
    main()
