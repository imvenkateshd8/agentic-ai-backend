from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from core.config import settings

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_DEPLOYMENT,
        api_version=settings.AZURE_API_VERSION,
        azure_endpoint=settings.AZURE_ENDPOINT,
        api_key=settings.AZURE_API_KEY,
    )

def get_embedding_model():
    return AzureOpenAIEmbeddings(
        model=settings.AZURE_EMBEDDING_MODEL_DEPLOYMENT,
        api_version=settings.AZURE_EMBEDDING_API_VERSION,
        azure_deployment=settings.AZURE_EMBEDDING_MODEL_DEPLOYMENT,
        azure_endpoint=settings.AZURE_ENDPOINT,
        api_key=settings.AZURE_API_KEY
    )
    
