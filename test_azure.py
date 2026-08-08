import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Read from .env
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")
version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

print(f"Endpoint: {endpoint}")
print(f"Key loaded: {'Yes' if key else 'NO'}")
print(f"Key prefix: {key[:12] if key else 'NONE'}...")
print(f"Deployment: {deployment}")
print(f"API Version: {version}")

# Try calling the model
try:
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=version,
        azure_deployment=deployment,
        temperature=0.3,
    )
    
    response = llm.invoke("Hello, Azure OpenAI! Can you respond to this test message?")
    print(f"\n✅ Azure OpenAI is working!")
    print(f"Response: {response.content}")
    
except Exception as e:
    print(f"\n❌ Azure OpenAI failed: {type(e).__name__}")
    print(f"Error: {e}")