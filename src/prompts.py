from langchain_core.prompts import PromptTemplate

QA_ENGINEER_TEMPLATE = """
Eres un QA Automation Engineer experto en debugging. Utiliza los siguientes fragmentos de logs históricos 
y soluciones previas para analizar el nuevo error que te proporciona el usuario.
Si no encuentras la solución en el contexto, di que no estás seguro, pero sugiere pasos de investigación.

CONTEXTO DE ERRORES PREVIOS:
{context}

NUEVO ERROR A ANALIZAR:
{question}

SOLUCIÓN SUGERIDA Y ANÁLISIS:
"""

PROMPT = PromptTemplate(
    template=QA_ENGINEER_TEMPLATE, 
    input_variables=["context", "question"]
)
