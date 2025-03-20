from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    '''initialize embeddings model for vector encoding
    - Output: OllamaEmbeddings object'''
    embeddings = OllamaEmbeddings(model="deepseek-r1", # LOCAL model we use
                                  base_url="http://localhost:11434")
    return embeddings