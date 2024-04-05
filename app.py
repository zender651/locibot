

from langchain.llms import Ollama
from fastapi import FastAPI,UploadFile,File
import os
import shutil
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory

from langchain.llms import Ollama

from langchain.chains import ConversationalRetrievalChain
import sys


llm = Ollama(model="llama2")

from langchain.prompts.prompt import PromptTemplate


template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.
Please be concise.
<</SYS>>


Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %}
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)

app=FastAPI()

@app.post("/query")
def predict(message: str, history: str):
    response = conversation.predict(input=message)

    return response


dir="files/"

@app.post("/FileInfo")
def fileapi(file:UploadFile,query:str):
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    path=os.path.join(dir,file.filename)
    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)    
    DB_FAISS_PATH = "vectorstore/db_faiss"
    loader = CSVLoader(file_path=path, encoding="utf-8", csv_args={'delimiter': ','})
     
    data=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    docsearch = FAISS.from_documents(text_chunks, embeddings)

    docsearch.save_local(DB_FAISS_PATH)
    
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    chat_history = []
    result = qa({"question":query, "chat_history":chat_history})
    return result

