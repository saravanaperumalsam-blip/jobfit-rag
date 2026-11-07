"""Central settings. Values can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_CSV = DATA_DIR / "jobs.csv"
CV_PATH = DATA_DIR / "sample_cv.txt"
DB_DIR = ROOT / "chroma_db"
RESULTS_DIR = ROOT / "results"

# Retrieval
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hybrid")  # "hybrid", "minilm" or "tfidf"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "40"))      # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "10"))  # words shared between chunks
TOP_K = int(os.getenv("TOP_K", "8"))

# LLM (any OpenAI-compatible API). Leave LLM_PROVIDER=none to run without an LLM.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()  # groq | openai | gemini | ollama | none
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

PROVIDER_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
}

# Backup models, tried in order when the main model hits a rate limit or is unavailable.
# On Groq each model has its own daily token quota, so a backup keeps answers working.
# Override with a comma-separated list, or set LLM_FALLBACK_MODELS=none to disable.
FALLBACK_DEFAULTS = {
    "groq": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
}
_fallback = os.getenv("LLM_FALLBACK_MODELS", "")
LLM_FALLBACK_MODELS = ([] if _fallback.lower() == "none" else
                       [m.strip() for m in _fallback.split(",") if m.strip()] or
                       FALLBACK_DEFAULTS.get(LLM_PROVIDER, []))
