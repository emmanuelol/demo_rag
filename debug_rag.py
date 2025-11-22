import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import re
import os


model_embedding ='bge-m3'                                       
model_chat = 'llama3.1:8b'

embeddings_directory = '/datasets/embeddings/'
pdf_bytes = '/datasets/pdf/'

print("Loading PDF documents...")
loader = PyPDFDirectoryLoader(pdf_bytes)
data = loader.load()
print(f"Loaded {len(data)} documents")

# Initialize embeddings

embeddings = OllamaEmbeddings(model=model_embedding)

# Spanish-optimized text splitter
#text_splitter = RecursiveCharacterTextSplitter(
#            chunk_size=500,
#            chunk_overlap=150,
#            separators=["\n\n---", "\n---", "## ", "# ", "\n\n", "\n", ". ", "! ", "? "]
 #       )

text_splitter = CharacterTextSplitter(
     separator="\n\n---",
     chunk_size=300,
     chunk_overlap=50,
     length_function=len,
     is_separator_regex=False,
 )

# Split documents
print("Splitting documents...")
docs = text_splitter.split_documents(data)
print(f"Created {len(docs)} chunks")

# Print sample chunks to verify splitting
print("\nSample chunks:")
for i, doc in enumerate(docs[:3]):
    print(f"Chunk {i+1}: {doc.page_content[:100]}...")

# Create vector store
print("Creating vector store...")
vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=embeddings_directory
        )
            
retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

def ollama_llm(question, context, model_embedding):

    formatted_prompt = f"""Eres un asistente especializado en recetas de café. Responde en español.
        CONTEXTO DE RECETAS:
        {context}

        PREGUNTA: {question}

        INSTRUCCIONES:
        - Responde únicamente en español
        - Si la receta está en el contexto, proporciona todos los ingredientes y pasos exactamente como aparecen
        - Si no encuentras la receta, di "No tengo información sobre esa receta en mis documentos"
        - Mantén el formato original de medidas y pasos
        - No inventes ingredientes o pasos"""

    response = ollama.chat(
        model=model_chat,
        messages=[{"role": "user", "content": formatted_prompt}],
        options={
            "temperature": 0.1,  # Moved temperature here
            "top_k": 10,
            "top_p": 0.9
        }
    )
    response_content = response["message"]["content"]

    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    return final_answer


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#question ='dame la receta de un matcha latte'
question ='dame la receta de un chai latte'


retrieved_docs = retriever.invoke(question)
print(f"Retrieved {len(retrieved_docs)} documents")
formatted_content = combine_docs(retrieved_docs)

print("\n=== RETRIEVED CONTEXT ===")
print(formatted_content[:1000] + "..." if len(formatted_content) > 1000 else formatted_content)


print("\n=== ANSWER ===")
answer = ollama_llm(question, formatted_content, model_chat)
print(answer)