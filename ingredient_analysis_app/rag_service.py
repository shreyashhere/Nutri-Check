import os
import logging
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF
import numpy as np
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, docs_path=None, persist_directory=None):
        # Use absolute paths based on Django project root
        if docs_path is None:
            docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
        
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        
        # Use sentence transformers directly
        try:
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformers: {e}")
            raise Exception("Could not initialize embeddings. Please check your installation.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Simple vector store using numpy
        self.documents = []
        self.embeddings_list = []
        
        logger.info(f"RAG Service initialized with docs_path: {self.docs_path}")
        logger.info(f"RAG Service initialized with persist_directory: {self.persist_directory}")
        self.initialize_knowledge_base()
    
    def load_pdf_documents(self):
        """Load PDF documents from the docs folder"""
        documents = []
        pdf_files = [f for f in os.listdir(self.docs_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.docs_path, pdf_file)
                logger.info(f"Loading PDF: {pdf_file}")
                
                # Use PyMuPDF for better text extraction
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                # Create document with metadata
                document = Document(
                    page_content=text,
                    metadata={"source": pdf_file, "type": "pdf"}
                )
                documents.append(document)
                logger.info(f"Successfully loaded {pdf_file}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                continue
        
        return documents
    
    def load_text_documents(self):
        """Load text documents from the docs folder"""
        documents = []
        text_files = [f for f in os.listdir(self.docs_path) if f.endswith('.txt')]
        
        for text_file in text_files:
            try:
                text_path = os.path.join(self.docs_path, text_file)
                logger.info(f"Loading text file: {text_file}")
                
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                document = Document(
                    page_content=text,
                    metadata={"source": text_file, "type": "text"}
                )
                documents.append(document)
                logger.info(f"Successfully loaded {text_file}")
                
            except Exception as e:
                logger.error(f"Error loading {text_file}: {str(e)}")
                continue
        
        return documents
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with documents"""
        try:
            # Check if vector store already exists
            vector_store_file = os.path.join(self.persist_directory, "vector_store.pkl")
            if os.path.exists(vector_store_file):
                logger.info("Loading existing vector store...")
                with open(vector_store_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings_list = data['embeddings']
                logger.info(f"Vector store loaded successfully with {len(self.documents)} documents")
                return
            
            # Load documents
            logger.info("Loading documents...")
            documents = []
            documents.extend(self.load_pdf_documents())
            documents.extend(self.load_text_documents())
            
            if not documents:
                logger.warning("No documents found in docs folder")
                return
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks")
            
            # Create embeddings
            logger.info("Creating embeddings...")
            self.documents = split_docs
            texts = [doc.page_content for doc in split_docs]
            self.embeddings_list = self.embeddings.encode(texts)
            
            # Save vector store
            os.makedirs(self.persist_directory, exist_ok=True)
            with open(vector_store_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings_list
                }, f)
            
            logger.info("Vector store created and persisted successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def retrieve_relevant_info(self, ingredients, user_profile, k=5):
        """Retrieve relevant information based on ingredients and user profile"""
        try:
            if not self.documents or len(self.embeddings_list) == 0:
                logger.warning("Vector store not initialized")
                return []
            
            # Create a comprehensive query
            query = f"""
            Ingredients: {ingredients}
            User allergies: {user_profile.get('allergies', 'None')}
            User medical conditions: {user_profile.get('diseases', 'None')}
            
            Find information about:
            1. Health effects of these ingredients
            2. Allergic reactions and contraindications
            3. Medical research and studies
            4. Safety recommendations
            5. Alternative ingredients
            """
            
            # Encode query as 1D vector
            query_embedding = self.embeddings.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings_list, query_embedding)
            
            # Get top k documents
            top_indices = np.argsort(similarities)[-k:][::-1]
            docs = [self.documents[i] for i in top_indices]
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant info: {str(e)}")
            return []

    
    def get_ingredient_specific_info(self, ingredient_name):
        """Get specific information about a particular ingredient"""
        try:
            if not self.documents or len(self.embeddings_list) == 0:
                return []
            
            query = f"Find detailed information about {ingredient_name} including health effects, safety, research studies, and medical recommendations"
            
            # Encode query as 1D vector
            query_embedding = self.embeddings.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings_list, query_embedding)
            
            # Get top 3 documents
            top_indices = np.argsort(similarities)[-3:][::-1]
            docs = [self.documents[i] for i in top_indices]
            
            return docs
            
        except Exception as e:
            logger.error(f"Error getting ingredient specific info: {str(e)}")
            return []

    
    def refresh_knowledge_base(self):
        """Refresh the knowledge base with new documents"""
        try:
            # Remove existing vector store
            vector_store_file = os.path.join(self.persist_directory, "vector_store.pkl")
            if os.path.exists(vector_store_file):
                os.remove(vector_store_file)
            
            # Clear current data
            self.documents = []
            self.embeddings_list = []
            
            # Reinitialize
            self.initialize_knowledge_base()
            logger.info("Knowledge base refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {str(e)}")
            raise
