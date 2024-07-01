import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tika import parser
from llama_index.core import VectorStoreIndex
from PyPDF2 import PdfReader

writer=open("gitaasitis.txt","r")

file=open('bagavat gita.txt', 'r')
content = file.read()
gita=writer.read()
content=content+gita
text_splitter=RecursiveCharacterTextSplitter(separators=["\n\n","\n"],chunk_size=20,chunk_overlap=0)
splits=text_splitter.split_text(content)

embeddings=OllamaEmbeddings(model="mistral")
print("embeddings setup completed..")
#emb=HuggingFaceEmbeddings()
vec_store=FAISS.from_texts(splits,embeddings)
print("FAISS setup compleated.")
#vec_store=chroma.from_texts(splits,embeddings)
retriever=vec_store.as_retriever()
print("retriver setup compleated.")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(qus,context):
    formatted=f"Question: {qus} \n\ncontext: {context}"
    print("formatted qus ans: ",formatted)
    stream=ollama.chat(model="mistral",messages=[{"role":"user","content":formatted}])
    print("ollama ans: ",stream["messages"]["content"])

def rag(question):
    re_docs=retriever.invoke(question)
    print("retriver invokes..")
    for_context=format_docs(re_docs)
    print("documents formatted..ready to read qus..")
    return ollama_llm(question,for_context)
rag("who is arjuna..?")
