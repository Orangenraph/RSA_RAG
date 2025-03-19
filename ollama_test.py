import os
import shutil
import argparse

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document


CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    # Check if db needs to be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset database")
    args = parser.parse_args()
    if args.reset:
        print("Resetting database")
        clear_database()

    # create - update data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_chroma(chunks)




def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH) # https://python.langchain.com/docs/how_to/document_loader_pdf/
    return document_loader.load()

def split_documents(documents: list[Document]):
    '''pdf is too big, splits into smaller chunks'''
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def get_embeddings():
    '''key for data base'''
    #embeddings = BedrockEmbeddings(credentials_profile_name="default",region_name="eu-west")
    embeddings = OllamaEmbeddings(model="deepseek-r1",
                                  base_url="http://localhost:11434")
    return embeddings

def add_chroma(chunks: list[Document]):
    '''builds vector chroma database'''
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())

    # Calc. Page Ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing chunks in DB: {len(existing_ids)}")

    # only add into db if it doenst exust
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)

    else:
        print(f"No new chunks found")

def calculate_chunk_ids(chunks):
    '''tag every item with string id:
        source path, pagenumber, chunk number
        example: "data/blablabla.pdf:8:3"   '''

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata['source']
        page = chunk.metadata['page']
        current_page_id = f"{source}_{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == '__main__':
    main()