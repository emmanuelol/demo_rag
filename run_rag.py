# Libraries
import gradio as gr
#from langchain_community.document_loaders import PyMuPDFLoader,PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

#from lanngchain_community.emmbiddings import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from ollama import ChatResponse, chat, Client
#import ollama
import re
import os
import yaml
from time import sleep


def process_pdf(pdf_bytes, model_embedding, persist_directory,chunk_size,chunk_overlap):
    if pdf_bytes is None:
        return None, None, None

    base_url = 'http://ollama:11434'
    #loader = PyMuPDFLoader(pdf_bytes)

    loader = PyPDFDirectoryLoader(path = pdf_bytes)
    data = loader.load()
    #print(loader)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model=model_embedding,base_url=base_url)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever

def load_embeddings(model_embedding, persist_directory, chunk_size, chunk_overlap):
    base_url = 'http://ollama:11434'
    print(model_embedding)
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function= OllamaEmbeddings(model=model_embedding,base_url=base_url))
    
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=chunk_size, 
         chunk_overlap=chunk_overlap
    )
   
    retriever = vectordb.as_retriever()

    
    return text_splitter, vectordb, retriever

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context, model_embedding):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    client = Client(
        host='http://ollama:11434'
        )
    
    response = client.chat(
        model=model_embedding,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]

    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    return final_answer


def rag_chain(question, text_splitter, retriever, model_embedding):
    
    retrieved_docs = retriever.invoke(question)
    
    formatted_content = combine_docs(retrieved_docs)
    
    return ollama_llm(question, formatted_content, model_embedding)



def ask_question(pdf_bytes, question, create_embeddings, embeddings_directory, model_embedding, chunk_size, chunk_overlap):
    if create_embeddings is True:
        print('create_embeddings')
        text_splitter, vectorstore, retriever = process_pdf(pdf_bytes,
                                                            model_embedding,
                                                            embeddings_directory,
                                                            chunk_size,
                                                            chunk_overlap
                                                           )
    else:
        print('preloaded')
        text_splitter, vectorstore, retriever = load_embeddings(model_embedding,
                                                                embeddings_directory,
                                                                chunk_size,
                                                                chunk_overlap
                                                               )

    if embeddings_directory is None:
        return None  # No PDF uploaded
    print(retriever)
    result = rag_chain(question, 
                       text_splitter,  
                       retriever, 
                       model_embedding)
    
    #return {result}
    return result

def main():

    # Load parameters
    #with open('config.yaml','r') as file:
    #    config = yaml.safe_load(file)

#    model_embedding = config['models']['embedding']
#    print(model_embedding)
#    embeddings_directory = config['path']['emmbeddings']
 #   print(embeddings_directory)
#    pdf_bytes = config['path']['pdfs']
 #   print(pdf_bytes)
#    create_embeddings = config['parameters']['create_embeddings']
  #  print(create_embeddings)
#    chunk_size = config['parameters']['chunk_size']
   # print(chunk_size)
#    chunk_overlap = config['parameters']['chunk_overlap']
    #print(chunk_overlap)
#    question ='please provide me the structure of a docker-compose.yaml file'
   # print(question)
    #results = ask_question(pdf_bytes ,question,create_embeddings, embeddings_directory, model_embedding, chunk_size, chunk_overlap)
    #print(results)
    ###
    # path_to_pdf,question,

    print('loading')
    #sleep(5*60)

    interface = gr.Interface(
        fn=ask_question,
        inputs=[
            gr.File(label="Path to upload PDF directory (optional)"), # path to pdf
            gr.Textbox(label="Ask a question"), # question
            gr.Checkbox(value=False,label='create embeddings', info='create embeddings'), # check to create embeddings
            #gr.File(label='Path to embeddings'), # path of the embeddings
            gr.Textbox(value='/datasets/deepseek-r1',label='Path to embeddings'), # path of the embeddings
            gr.Dropdown(['deepseek-r1:1.5b','deepseel-r1:7b'],label='model name'), ## model_name
            gr.Slider(1,1000,value=500,label='chunk size'), ## chunk_size
            gr.Slider(1,200,value=100,label='chunk overlap'), ## chuck_overlap

        ],
        outputs="textbox",
        title="Ask questions about Machine Learning, data science and, MLOps based on Packt PDFs",
        description="Use DeepSeek-R1 to answer your questions about the uploaded PDF documents.",
    )

    interface.launch(share=True)
    #interface.launch(inbrowser=True)
'''
if __name__ == '__main__':
    interface.launch()

'''
if __name__ == '__main__':
    main()
