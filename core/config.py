import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDING_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
    AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

settings = Settings()
