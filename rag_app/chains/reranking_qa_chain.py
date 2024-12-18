# retrieval chain
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pickle
from config import PERSIST_DIRECTORY, EMBEDDING_MODEL, S_LLM_MODEL_HF, RERANKER_MODEL

embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
    )

vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
chroma_retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k':5, 'fetch_k':10})

with open(f'{PERSIST_DIRECTORY}/keywords_index.pkl', "rb") as f:
    bm25_retriever = pickle.load(f)

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever]
)

# Rerank the ensemble results
reanker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=reanker_model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

# Load the model from the Hugging Face Hub
llm = HuggingFaceEndpoint(repo_id=S_LLM_MODEL_HF, 
                          temperature=0.1, 
                          max_new_tokens=1024,
                          repetition_penalty=1.2,
                          return_full_text=False
    )

global qar
template = """
You are an RFP analysis agent, tasked with analyzing the given specifications and extracting insights about electrical transformers.\
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
# Create a PromptTemplate object with specified input variables and the defined template
prompt = PromptTemplate.from_template(
    template=template,
)
prompt.format(context="context", history="history", question="question")

# Create a memory buffer to manage conversation history
memory = ConversationBufferMemory(memory_key="history", input_key="question")

# Initialize the RetrievalQA object with the specified model, # retriever, and additional configurations
qar = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
    "memory": memory,
    "prompt": prompt
}
    )