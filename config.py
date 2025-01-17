from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):

    chromadb_dir: Path = './src/backend/LLM/chroma server'
    chromadb_port: int = 8000
    ollama_cmd: str = 'ollama serve'
    streamlit_file: Path = './src/frontend/chat_ui.py'
    streamlit_port: int = 8501
    fastapi_dir: Path = './src/backend/LLM/llm-api'
    fastapi_port: int = 8080


config = Settings()