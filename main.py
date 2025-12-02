import streamlit as st
import numpy as np
import os
import io

# --- LangChain and PDF Loader Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Core RAG Imports ---
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity 

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_COLLECTION_NAME = 'rag_document_index'
# Using common delimiters for better recursive splitting
CHUNK_DELIMITERS = ["\n\n", "\n", " ", ""]
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 100 
TOP_K_CHUNKS = 2 

# --- Core RAG Logic ---

class DocumentRAGProcessor:
    """
    Handles document loading (LangChain), chunking (LangChain), embedding (SBERT), 
    vector storage (ChromaDB), and retrieval (Scikit-learn Cosine Similarity).
    """

    def __init__(self):
        """Initializes the Sentence Transformer model and ChromaDB client."""
        
        # 1. Initialize the Embedding Model (Explicitly setting device='cpu' to address the meta tensor error)
        @st.cache_resource
        def load_model():
            st.info(f"Loading Sentence Transformer: {EMBEDDING_MODEL_NAME} on CPU...", icon="🧠")
            try:
                model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
                st.success(f"Embedding Model ({EMBEDDING_MODEL_NAME}) Loaded Successfully on CPU!")
                return model
            except Exception as e:
                st.error(f"FATAL: The Embedding Model failed to load. Error: {e}")
                st.error("Please ensure your PyTorch, sentence-transformers, and dependencies are up to date.")
                return None

        self.model = load_model()

        # 2. Initialize ChromaDB client and collection
        self.chunk_texts = [] 
        self.is_indexed = False
        self.chroma_client = None
        self.collection = None

        if self.model:
            # Custom adapter to bridge the cached SBERT model instance to ChromaDB
            class CachedModelChromaAdapter(embedding_functions.EmbeddingFunction):
                def __init__(self, model_instance):
                    self.model_instance = model_instance
                def __call__(self, texts):
                    # Encode texts using the cached SentenceTransformer model instance
                    return self.model_instance.encode(texts).tolist()
            
            self.embedding_function = CachedModelChromaAdapter(self.model)
            
            # ChromaDB uses a simple in-memory client
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
        

    def load_document(self, uploaded_file: io.BytesIO):
        """Loads, chunks using LangChain, and indexes the document into ChromaDB."""
        if not self.model or not self.collection:
            st.error("Model or Vector Store failed to initialize. Cannot process document.")
            return
            
        self.is_indexed = False
        self.chunk_texts = []
        
        # 1. Save uploaded file temporarily to disk for LangChain to read
        temp_file_path = f"./temp_doc_{uploaded_file.name}"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving file temporarily: {e}")
            return

        st.info("Loading document using LangChain's loader...")
        
        # 2. Load the document using the appropriate LangChain loader
        try:
            if uploaded_file.name.endswith('.pdf'):
                # PyPDFLoader is reliable for general PDF text extraction
                loader = PyPDFLoader(temp_file_path)
            elif uploaded_file.name.endswith(('.txt', '.md')):
                # TextLoader for text-based files
                loader = TextLoader(temp_file_path)
            else:
                st.warning("Unsupported file type. Please upload a PDF, TXT, or MD file.")
                os.remove(temp_file_path)
                return

            documents = loader.load()

        except Exception as e:
            st.error(f"Error loading document with LangChain: {e}")
            os.remove(temp_file_path)
            return

        # 3. Chunking using LangChain's RecursiveTextSplitter
        st.info("Chunking document using RecursiveCharacterTextSplitter...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=CHUNK_DELIMITERS
            )
            # This creates high-quality, semantically coherent chunks
            split_documents = text_splitter.split_documents(documents)
            self.chunk_texts = [doc.page_content for doc in split_documents]

        except Exception as e:
            st.error(f"Error splitting text: {e}")
            os.remove(temp_file_path)
            return
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        # 4. Add to ChromaDB
        if self.chunk_texts:
            with st.spinner(f"Indexing {len(self.chunk_texts)} chunks into ChromaDB..."):
                
                # Clear existing data in the collection
                self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )

                # Generate IDs
                ids = [f"doc_{i}" for i in range(len(self.chunk_texts))]
                
                # Add documents (embeddings are generated internally by Chroma)
                self.collection.add(
                    documents=self.chunk_texts,
                    ids=ids
                )

            self.is_indexed = True
            st.success(f"Document processed successfully! {len(self.chunk_texts)} chunks indexed in ChromaDB.")
        else:
            st.error("Could not create document chunks.")


    def get_answer(self, query: str) -> tuple[str, list[tuple[str, float, str]]]:
        """
        Queries ChromaDB, then re-ranks the results using Scikit-learn's Cosine Similarity.
        """
        if not self.is_indexed or not self.collection:
            return "Please upload and process a document first, and ensure the model is loaded.", []

        # 1. Query ChromaDB to retrieve more candidates (e.g., K*4) for better re-ranking
        N_CANDIDATES = TOP_K_CHUNKS * 4 
        
        results = self.collection.query(
            query_texts=[query],
            n_results=N_CANDIDATES,
            include=['documents'] 
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant chunks found.", []
            
        retrieved_chunks = results['documents'][0]
        
        # 2. Embed the query and the retrieved chunks using the SBERT model
        query_embedding = self.model.encode([query])
        candidate_embeddings = self.model.encode(retrieved_chunks)
        
        # 3. Calculate Cosine Similarity (using sklearn)
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # 4. Combine chunks and scores, then sort by similarity
        chunk_score_pairs = list(zip(retrieved_chunks, similarities))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        final_top_k = chunk_score_pairs[:TOP_K_CHUNKS]

        # 5. Format results
        retrieved_info = []
        for i, (text, similarity) in enumerate(final_top_k):
            retrieved_info.append((f"Chunk {i+1}", similarity, text))

        # The answer is the concatenated retrieved context.
        context = "\n\n---\n\n".join([chunk for _, _, chunk in retrieved_info])
        response = f"**Retrieval Context (Top {TOP_K_CHUNKS} relevant chunks, re-ranked by Cosine Similarity):**\n\n{context}"
             
        return response, retrieved_info

# --- Streamlit App UI ---

def main():
    """Streamlit application entry point."""
    st.set_page_config(page_title="LangChain RAG App", layout="wide")

    st.title("📄 LangChain-Powered RAG Application Chat With Your Document")
    st.markdown("Now using **LangChain's Without Api")
    st.markdown("---")

    # Initialize RAG Processor in session state
    if 'rag_processor' not in st.session_state:
        st.session_state.rag_processor = DocumentRAGProcessor()
    
    rag_processor = st.session_state.rag_processor
    
    if not rag_processor.model:
        st.error("Application setup failed. The embedding model failed to load. Please check your dependencies.")
        return

    # Sidebar for Configuration
    with st.sidebar:
        st.header("1. Document Input")
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        process_button = st.button("Process Document & Build Index", type="primary", use_container_width=True)

        if process_button and uploaded_file is not None:
            rag_processor.load_document(uploaded_file)
            st.session_state.document_ready = rag_processor.is_indexed
        elif process_button and uploaded_file is None:
            st.error("Please upload a file before processing.")
            st.session_state.document_ready = False
        
        st.session_state.document_ready = rag_processor.is_indexed

        if st.session_state.document_ready and rag_processor.collection:
             st.sidebar.success(f"ChromaDB Index Ready with {rag_processor.collection.count()} chunks.")
        else:
             st.sidebar.warning("No document indexed yet.")

        st.markdown("---")
        st.header("Configuration")
        st.markdown(f"""
        - **Document Loader:** `LangChain PyPDFLoader/TextLoader`
        - **Text Splitter:** `LangChain RecursiveCharacterTextSplitter`
        - **Embedding Model:** `{EMBEDDING_MODEL_NAME}`
        - **Vector Store:** `ChromaDB (In-Memory)`
        - **Chunk Size:** `{CHUNK_SIZE}`
        - **Chunk Overlap:** `{CHUNK_OVERLAP}`
        - **Retrieved Chunks (K):** `{TOP_K_CHUNKS}`
        """)


    # Main Area for Q&A
    st.header("2. Ask a Question")

    if not st.session_state.document_ready:
        st.warning("Please upload and process a document first to enable the Q&A feature.")

    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., Explain the concept of Machine Learning.",
        disabled=not st.session_state.document_ready,
        key="user_query"
    )

    if st.button("Get Answer (Pure Retrieval)", disabled=not st.session_state.document_ready):
        if user_query:
            with st.spinner("Searching and re-ranking knowledge base..."):
                response, retrieved_info = rag_processor.get_answer(user_query)

            st.subheader("🤖 Answer / Generated Response")
            st.markdown(response)

            st.subheader("🔍 Retrieval Details")
            for i, (id_placeholder, similarity, text) in enumerate(retrieved_info):
                with st.expander(f"Chunk {i+1} (Cosine Similarity Score: {similarity:.4f})"):
                    st.code(text, language='text')

        else:
            st.error("Please enter a question.")


if __name__ == "__main__":
    main()