# Section-level splits, based on MD Headers
from langchain_text_splitters import MarkdownHeaderTextSplitter
# Char-level splits
from langchain_text_splitters import RecursiveCharacterTextSplitter
# vectorization functions
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from config import PERSIST_DIRECTORY, EMBEDDING_MODEL
import time
import pickle

# get the MD file
markdown_path = "docs/2014_CRM_Spec. Power Transformer_220kV_100MVA_GTC 643-2014_Rev 3.md"
print(f'reading file {markdown_path} ...')
with open(markdown_path, 'r') as file:
    full_markdown_text = file.read()

# the headers we will use to split the document into sections
headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
    ("###", "Topic"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

md_header_splits = markdown_splitter.split_text(full_markdown_text)

print(f'split file into sections using headers, created a total of {len(md_header_splits)} ...')


chunk_size = 500
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# char-level Splits 
splits = text_splitter.split_documents(md_header_splits)
print(f'split sections into a total of {len(splits)} chunks ...')

# create vector store
print(f'Loading chunks into chroma vector store ...')
st = time.time()
persist_directory=PERSIST_DIRECTORY
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db_chroma = Chroma.from_documents(splits, embeddings, persist_directory=persist_directory)

#  and bm25 index
bm25_retriever = BM25Retriever.from_documents(md_header_splits)
with open(f'{PERSIST_DIRECTORY}/keywords_index.pkl', "wb") as f:
    pickle.dump(bm25_retriever, f)

et = time.time() - st
print(f'Time taken: {et} seconds.')