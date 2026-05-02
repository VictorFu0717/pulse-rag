"""Pluse RAG — entry point.

Quick start
-----------
    cp .env.example .env                # fill in API keys
    python scripts/build_index.py       # build FAISS indices (first time only)
    python main.py                      # start server on port 8002

Configuration: config/config.yaml
System prompt: config/system_prompt.txt
"""
import os

from dotenv import load_dotenv

load_dotenv()

for _key in ("TAVILY_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "OPENAI_API_KEY"):
    _val = os.getenv(_key, "")
    if _val:
        os.environ[_key] = _val

from core.config import load_settings  # noqa: E402
from core.server import create_app     # noqa: E402

settings = load_settings()
app = create_app(settings)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)
