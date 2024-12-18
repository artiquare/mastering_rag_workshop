# retrieval chain
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import PERSIST_DIRECTORY, EMBEDDING_MODEL, S_LLM_MODEL_HF

# Load the model from the Hugging Face Hub
llm = HuggingFaceEndpoint(repo_id=S_LLM_MODEL_HF, 
                          temperature=0.1, 
                          max_new_tokens=1024,
                          repetition_penalty=1.2,
                          return_full_text=False
    )

embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
    )

vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
chroma_retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k':5, 'fetch_k':10})

global qm
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

# Initialize the RetrievalQA object with the specified model, retriever, and additional configurations
qm = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=chroma_retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
    "memory": memory,
    "prompt": prompt
}
    )