"""
Configurações centralizadas do sistema.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- TELEGRAM ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN não definido no arquivo .env")

# ---------------- MODELS ----------------
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------- LLM SETTINGS ----------------
LLM_REQUEST_TIMEOUT = 60.0  # segundos
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_WINDOW = 4096
LLM_NUM_CTX = 4096

# ---------------- RETRIEVAL SETTINGS ----------------
SIMILARITY_TOP_K = 20
SIMILARITY_CUTOFF = 0.2

# ---------------- QUERY SETTINGS ----------------
QUERY_TIMEOUT = 360  # segundos
MAX_RESPONSE_LENGTH = 4000  # caracteres (limite do Telegram: 4096)

# ---------------- STORAGE ----------------
STORAGE_DIR = "./storage"
CHROMA_COLLECTION_NAME = "institucional_db"

# ---------------- LOGGING ----------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ---------------- PROMPT SETTINGS ----------------
MAX_CONTEXT_LENGTH = 8000  # Caracteres máximos de contexto
ENABLE_HALLUCINATION_DETECTION = True