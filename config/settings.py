import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- TELEGRAM ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN não definido no arquivo .env")

# ---------------- GOOGLE GEMINI ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY não definido no arquivo .env")

# ---------------- MODELS ----------------
LLM_MODEL = "models/gemini-2.5-flash"  # Nome genérico da variável (antes era OLLAMA_MODEL)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------- LLM SETTINGS ----------------
LLM_TEMPERATURE = 0.1

# ---------------- RETRIEVAL SETTINGS ----------------
SIMILARITY_TOP_K = 20
SIMILARITY_CUTOFF = 0.3

# ---------------- QUERY SETTINGS ----------------
QUERY_TIMEOUT = 90  # segundos
MAX_RESPONSE_LENGTH = 4000  # caracteres

# ---------------- STORAGE ----------------
STORAGE_DIR = "./storage"
CHROMA_COLLECTION_NAME = "institucional_db"

# ---------------- LOGGING ----------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"