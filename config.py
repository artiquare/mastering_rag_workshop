import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
PERSIST_DIRECTORY = os.getenv('VECTOR_DATABASE_LOCATION')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MID_LLM_MODEL_HF = os.getenv("MID_LLM_MODEL")
S_LLM_MODEL_HF = os.getenv("S_LLM_MODEL")
BERT_MODEL = os.getenv("BERT_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")