# RAG Workshop: Transformer Specification Analysis

This repository contains the code and resources used in the workshop for building and refining RAG pipelines to analyze transformer specifications. The implementation demonstrates multiple retrieval and generation strategies, progressing from basic similarity search to advanced reranking.

---

## 📂 **Project Structure**
- **`docs/`**: Contains the Markdown (`.md`) file created from the original PDF specifications using `docling`.
- **`rag_app/chains/`**: Includes the implementations for various RAG chains:
  - `basic_qa_chain.py`: Basic similarity search-based chain.
  - `mmr_qa_chain.py`: Chain with Maximal Marginal Relevance (MMR).
  - `ensemble_qa_chain.py`: Chain combining BM25 and embedding-based retrieval.
  - `reranking_qa_chain.py`: Advanced chain incorporating MMR, ensemble retrieval, and reranking.
- **`app_gui_<chain_name>.py`**: Gradio-based GUIs for each chain, e.g., `app_gui_basic.py` for the basic chain.
- **`chunking_embedding.py`**: Preprocessing script to chunk and embed the Markdown file into Chroma and BM25.

---

## 🔧 **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. **Install Dependencies**:
   Use the provided `requirements.txt` file to install the necessary Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the Data**:
   Run `chunking_embedding.py` to chunk the Markdown file and embed it into Chroma and BM25:
   ```bash
   python chunking_embedding.py
   ```

4. **Run the GUIs**:
   Each RAG chain has its own Gradio interface. Launch a specific chain by running its corresponding script, e.g., for the basic chain:
   ```bash
   python app_gui_basic.py
   ```

   The application will open on a specific port (e.g., `localhost:7861`).

---

## 🚀 **Chains Overview**

1. **Basic Chain**:
   - Implements simple similarity search-based retrieval.
   - Found in `rag_app/chains/basic_qa_chain.py`.

2. **MMR Chain**:
   - Adds Maximal Marginal Relevance (MMR) for diverse retrieval.
   - Found in `rag_app/chains/mmr_qa_chain.py`.

3. **Ensemble Chain**:
   - Combines BM25 and embedding-based retrieval for hybrid results.
   - Found in `rag_app/chains/ensemble_qa_chain.py`.

4. **Reranker Chain**:
   - Uses MMR and ensemble retrieval, with reranking for improved precision.
   - Found in `rag_app/chains/reranking_qa_chain.py`.

---

## ✨ **Key Features**
- **Flexible Retrieval Strategies**: Experiment with similarity search, MMR, hybrid retrieval, and reranking.
- **Gradio Integration**: User-friendly GUIs for each chain.
- **Scalable Preprocessing**: Efficient document embedding with Chroma and BM25.

---

## 📝 **Usage Example**
1. Process the Markdown file:
   ```bash
   python chunking_embedding.py
   ```
2. Start the RAG GUI:
   ```bash
   python app_gui_mm.py
   ```
3. Query the system to explore transformer specifications interactively.

## 🛠 Additional Details

### Parallel Execution for Side-by-Side Comparison
Each Gradio app GUI is configured to run on a unique port, allowing you to run multiple chains in parallel for easier comparison:
- `app_gui_basic.py`: Runs on port **7861**.
- `app_gui_mmr.py`: Runs on port **7862**.
- `app_gui_ensemble.py`: Runs on port **7863**.
- `app_gui_reranker.py`: Runs on port **7864**.

To run all GUIs simultaneously:
```bash
python app_gui_basic.py &
python app_gui_mmr.py &
python app_gui_ensemble.py &
python app_gui_reranker.py &
```

You can then access each GUI at `localhost:<port>`.

---

### 🔔 Known Warnings
During execution, you might encounter the following warnings. These are due to our choice to maintain compatibility with older versions for simplicity:
1. **LangChainDeprecationWarning**: 
   - Warning from `ConversationBufferMemory` in LangChain.
   - Migration guide: [LangChain Migration Guide](https://python.langchain.com/docs/versions/migrating_memory/).
2. **Gradio UserWarning**: 
   - Warning about `type='tuples'` in the Gradio chatbot component.
   - This is related to upcoming Gradio updates that recommend using `type='messages'`.

These warnings are non-critical and do not affect the functionality of the pipelines. Updating to newer versions may introduce additional complexity not covered in this workshop.
