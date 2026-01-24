import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN ---
MODEL_NAME = "mistral"  # Aquí puedes cambiar a "llama3" fácilmente
DATA_PATH = "data/logs"
DB_PATH = "db_chroma"

# 1. Cargar documentos (Logs de errores previos o históricos de bugs)
def load_logs():
    print("Cargando logs históricos...")
    # Asumimos que tienes una carpeta 'data/logs' con archivos .txt o .log
    loader = DirectoryLoader(DATA_PATH, glob="./*.log", loader_cls=TextLoader)
    documents = loader.load()
    
    # Dividir el texto en chunks (importante para logs largos)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# 2. Crear o cargar la Base de Datos Vectorial
def get_vectorstore(chunks):
    # Usamos embeddings de HuggingFace que corren localmente
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(DB_PATH):
        print("Cargando base de datos persistente...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("Creando nueva base de datos vectorial...")
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_PATH
        )
    return vectorstore

# 3. Configurar el Prompt (Cerebro del QA Engineer)
template = """
Eres un QA Automation Engineer experto en debugging. Utiliza los siguientes fragmentos de logs históricos 
y soluciones previas para analizar el nuevo error que te proporciona el usuario.
Si no encuentras la solución en el contexto, di que no estás seguro, pero sugiere pasos de investigación.

CONTEXTO DE ERRORES PREVIOS:
{context}

NUEVO ERROR A ANALIZAR:
{question}

SOLUCIÓN SUGERIDA Y ANÁLISIS:
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# 4. Ejecución del RAG
def main():
    # Asegúrate de crear la carpeta data/logs y meter algún archivo .log de ejemplo
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Por favor, añade archivos .log en {DATA_PATH} y reinicia.")
        return

    chunks = load_logs()
    vectorstore = get_vectorstore(chunks)
    
    # Configuramos Ollama con el modelo elegido
    llm = Ollama(model=MODEL_NAME)
    
    # Creamos la cadena de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("\n--- SISTEMA DE DEBUGGING IA LISTO ---")
    while True:
        query = input("\nIntroduce el log de error o stack trace (o 'salir'): ")
        if query.lower() == 'salir':
            break
        
        print("\nAnalizando con IA...")
        response = qa_chain.invoke(query)
        print("\nREPORTE DE LA IA:")
        print(response["result"])

if __name__ == "__main__":
    main()