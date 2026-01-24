# Smart Error Debugger

Analizador de logs y buscador de errores diseñado para equipos de QA. Este proyecto utiliza técnicas de RAG (Retrieval-Augmented Generation) para contrastar nuevos errores con históricos de la base de datos y ofrecer soluciones basadas en experiencias previas.

## Stack Tecnológico

El proyecto está construido sobre un stack moderno orientado a IA local y eficiencia en el procesamiento de texto:

- **LLM**: DeepSeek-R1 (8B) corriendo sobre **Ollama**.
- **Orquestación**: **LangChain** para la gestión de cadenas y recuperación de contexto.
- **Base de Datos Vectorial**: **ChromaDB** para el almacenamiento persistente de fragmentos de logs.
- **Embeddings**: Modelo **all-MiniLM-L6-v2** de HuggingFace (ejecución 100% local).
- **Ingesta**: Procesamiento personalizado para archivos `.log` y reportes estructurados `.json`.

## Estructura del código

Para garantizar la escalabilidad desde el MVP a producción, el código se ha organizado de forma modular:

- `main.py`: Punto de entrada y CLI interactiva.
- `src/config.py`: Centraliza rutas, nombres de modelos e hiperparámetros.
- `src/loader.py`: Gestiona la carga de archivos. Incluye lógica específica para extraer campos clave de JSONs de error (error_message, stack_trace, fix).
- `src/vector_store.py`: Encapsula la persistencia y recuperación en ChromaDB.
- `src/model.py`: Configura el cerebro del sistema y la cadena de razonamiento de DeepSeek.
- `src/prompts.py`: Define el "persona" del asistente como un ingeniero experto en QA.

## Instalación y Configuración

1. **Modelos Locales**: Asegúrate de tener Ollama instalado y el modelo descargado:
   ```bash
   ollama pull deepseek-r1:8b
   ```

2. **Entorno de Python**: Instala las dependencias necesarias:
   ```bash
   pip3 install -r requirements.txt
   ```

## Modo de uso

1. **Histórico**: Copia tus archivos de logs o reportes de errores previos en la carpeta `data/logs/`. El sistema acepta formatos `.log` y `.json`.
2. **Ejecución**: Inicia el sistema con:
   ```bash
   python3 main.py
   ```
3. **Consulta**: Introduce una traza de error o un log sospechoso. El sistema buscará en el histórico y DeepSeek generará un análisis de causa raíz y posibles soluciones.

## Notas sobre el procesamiento JSON

A diferencia de un cargador de texto simple, este sistema procesa los archivos JSON de forma inteligente. Si el archivo contiene campos como `previous_fix` o `error_message`, el cargador prioriza esta información para que el modelo pueda aprender de soluciones que ya funcionaron en el pasado.
