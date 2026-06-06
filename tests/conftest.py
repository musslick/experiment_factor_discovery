"""
Load .env before any test module is imported so ANTHROPIC_API_KEY is available.
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
