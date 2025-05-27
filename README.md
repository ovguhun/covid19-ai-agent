# AI Research Agent: COVID-19 & Smoking Linkages

## 🚀 Introduction

This project, introduces an AI research agent specialized in analyzing the CORD-19 (COVID-19 Open Research Dataset). The primary goal of this agent is to answer questions and provide insights into the relationship between COVID-19 and smoking (including cigarettes, e-cigarettes, vaping, and tobacco use).

The agent leverages Large Language Models (LLMs) and a Retrieval Augmented Generation (RAG) pipeline built with LlamaIndex to process and query a filtered subset of the CORD-19 dated. It features a conversational interface built with Gradio, allowing users to interact with the research findings in an intuitive way.

## ✨ Features

* **Specialized Knowledge Base:** Utilizes a filtered subset of the CORD-19 dataset, focusing specifically on abstracts relevant to smoking and COVID-19.
* **Retrieval Augmented Generation (RAG):** Employs LlamaIndex to build a vector index from relevant abstracts, retrieve pertinent context for user queries, and synthesize answers.
* **Advanced LLM:** Powered by the `unsloth/llama-3-8b-Instruct-bnb-4bit` model, a quantized version of Llama 3 8B, for intelligent and nuanced responses.
* **Conversational Interface:** Uses a `CondenseQuestionChatEngine` to understand chat history and handle follow-up questions or vague queries more effectively.
* **Robust Prompt Engineering:** Features custom prompts for both question condensing and final answer generation to ensure factual grounding and appropriate handling of off-topic or casual inputs.
* **Interactive Web UI:** A user-friendly interface built with Gradio for easy interaction with the AI assistant.
* **Optimized for Performance:** Runs on CUDA-enabled GPUs (tested on A100) and uses techniques like model quantization and efficient data handling.
* **Persistent Index Storage:** Supports saving the vector index to Google Drive to avoid time-consuming rebuilds.

## 🛠️ Technologies Used

* **Core AI/RAG:** LlamaIndex, Hugging Face Transformers
* **Language Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **UI:** Gradio
* **Data Handling:** Pandas, PyArrow
* **Development Environment:** Python 3 on Google Colab (with A100 GPU)
* **Supporting Libraries:** PyTorch, `bitsandbytes`, `accelerate`, `tqdm`

## 📋 Project Workflow (Phases in the Notebook)

The development process, as detailed in the accompanying `.ipynb` notebook, followed these main phases:

1.  **Phase 1: Project Setup & Environment Configuration:**
    * Setting up the Google Colab GPU environment.
    * Installing all necessary Python libraries.
    * Verifying PyTorch and CUDA compatibility.
    * Setting up Hugging Face Hub authentication.
    * Integrating Google Drive for persistent storage.
2.  **Phase 2: Data Acquisition & Preprocessing:**
    * Downloading the CORD-19 abstracts dataset (`pritamdeka/cord-19-abstract`).
    * Inspecting and cleaning the data (handling missing values, ensuring correct types).
    * Filtering abstracts using keywords related to smoking, COVID-19, etc., to create a specialized corpus.
3.  **Phase 3: Vector Database Creation & Management:**
    * Configuring LlamaIndex global settings for the embedding model.
    * Chunking the filtered abstracts into manageable pieces (150 words per chunk).
    * Creating a `VectorStoreIndex` from these chunks, generating embeddings on the GPU.
    * Persisting (saving) the index to Google Drive and implementing logic to load it.
4.  **Phase 4: AI Agent Development:**
    * Configuring the LLM (`unsloth/llama-3-8b-Instruct-bnb-4bit` with `context_window=4096`, `max_new_tokens=512`).
    * Creating a robust base RAG query engine with a custom prompt template designed for factual synthesis and graceful handling of off-topic queries.
    * Layering a `CondenseQuestionChatEngine` on top, with a custom condense prompt, to enable better conversational understanding and handling of follow-ups or vague inputs.
5.  **Phase 5: Application Development (Gradio GUI):**
    * Structuring the AI system initialization (embedding, index, LLM, engines) into a single, cacheable function (`initialize_ai_system`).
    * Developing the Gradio `ChatInterface` with streaming responses and pre-filters for common casual inputs.

## ⚙️ Setup & How to Run

This project is designed to be run in a Google Colab notebook environment.

1.  **Open the Notebook:** Upload and open the provided `.ipynb` file in Google Colab.
2.  **Select GPU Runtime:**
    * In the Colab menu, navigate to `Runtime` -> `Change runtime type`.
    * Select `Python 3` and choose an `A100 GPU` (or `T4 GPU`/`V100 GPU` if A100 is unavailable, though performance will vary).
3.  **Hugging Face Token:**
    * You will need a Hugging Face Hub account and an access token with at least `read` permissions.
    * When prompted by the `huggingface-cli login` or `notebook_login()` cell (Step 1.4 in the notebook), provide your token.
4.  **Google Drive (Recommended for Index Persistence):**
    * The notebook is configured to save/load the `VectorStoreIndex` to/from Google Drive to `/content/drive/MyDrive/CORD19_Smoking_Chatbot_Index`.
    * Ensure you authorize Colab to access your Google Drive when prompted (Step 1.5). If you change this path, update it in the relevant cells (Step 3.3 for saving, and Step 5.1 `initialize_ai_system` function for loading).
    * If Google Drive is not used, the index will be built in Colab's temporary storage and will need to be rebuilt each session.
5.  **Run Cells in Order:** Execute the notebook cells sequentially from top to bottom.
    * **Library Installation (Step 1.2):** Installs all dependencies.
    * **Data Processing & Indexing (Phases 2 & 3):** These cells download data and build the vector index. The index creation (Step 3.3) will take several minutes.
    * **AI System Initialization (Step 5.1 - defines `initialize_ai_system`):** This cell prepares the function that sets up the entire AI agent.
    * **Launch Gradio UI (Step 5.2):** This cell will first call `initialize_ai_system()`. The first time this runs in a session, it will load the LLM (which can take several minutes) and prepare the engine. Subsequent calls in the same session will use the cached engine. After initialization, it will launch the Gradio interface and provide a public URL.
6.  **Access the Chatbot:** Click the public Gradio URL generated in the output of the final cell to interact with the AI Research Assistant.

## 📊 Dataset Used

* **Primary Dataset:** [pritamdeka/cord-19-abstract](https://huggingface.co/datasets/pritamdeka/cord-19-abstract) from Hugging Face.
    * This dataset provides a collection of abstracts from the CORD-19 research papers.
    * We filter this further using keywords: `["smoking", "cigarette", "nicotine", "vaping", "tobacco", "e-cigarette", "smoker"]`.

## 🧠 Model Used

* **Large Language Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
    * A 4-bit quantized version of Meta's Llama 3 8B Instruct model, optimized by Unsloth for faster inference and reduced memory footprint.
    * Key parameters used: `context_window=4096`, `max_new_tokens=512`, `temperature=0.7`.
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
    * Used for generating vector embeddings from the text chunks.

## 💡 Key Parameters & Configurations

* **Retriever `similarity_top_k`:** 5 (retrieves the top 5 most relevant text chunks for a query).
* **Text Chunk Size:** 150 words.
* **Prompts:** Custom prompt templates are used for both the `CondenseQuestionChatEngine` (to refine user queries based on chat history) and the base `RetrieverQueryEngine` (to guide the LLM in synthesizing answers based strictly on the provided context and handling off-topic/casual inputs).

## 🔮 Potential Future Work

* **Evaluation of Retrieval Quality:** Implement metrics to evaluate the relevance of retrieved documents.
* **More Advanced Chunking:** Explore sentence-aware chunking or proposition-based indexing.
* **Hybrid Search:** Combine vector search with traditional keyword search for potentially improved retrieval.
* **Fact-Checking/Citation for LLM Output:** More explicitly link parts of the generated answer to specific source sentences within the retrieved chunks.
* **UI Enhancements:** Add features like displaying source document links or allowing users to give feedback on answers.

---

*Author: Mehmet Ovgu Hun*

*Course: Advanced Natural Language Processing, M.Sc. Data Science, University of Debrecen, 2025*
