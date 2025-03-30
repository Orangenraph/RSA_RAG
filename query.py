import argparse


from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# helper functions
from embeddings import get_embeddings
from config import CHROMA_PATH, text_prompt
from helpers import save_response_to_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    '''performs RAG query and generates rules in the format 'if <condition> then <do>'.'''
    # load embeddings and db
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if results := db.similarity_search_with_score(query_text, k=5): # top k most relevant chunks to our question
        print(f"Similarity score: {results[0][1]}")

    context_text = "\n\n-------\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(text_prompt)
    prompt = prompt_template.format(context=context_text, query=query_text)

    # sends prompt to LLM
    model = OllamaLLM(model="deepseek-r1",
                      temperature=0.2,  # Temp 0-1 for creativity,
                      num_ctx=4096) # Context length num_ctx numbers of tokens ,

    response_text = model.invoke(prompt)

    # extracs  sources from metadata
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"RAG Response:\n\n{response_text}\n"
    print(formatted_response)
    print("Sources:")
    for source in sources:
        print(source)

    save_response_to_json(
        query_text=query_text,
        response_text=response_text,
        sources=sources,
        similarity_score=results[0][1])

    return formatted_response

if __name__ == '__main__':
    main()