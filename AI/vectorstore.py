from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS, Pinecone, Weaviate, Milvus, Chroma
from langchain.chains import LLMChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class VectorStore(ABC):
   """Abstract base class for vector stores"""
   
   @abstractmethod
   def create(self, documents: List[Document], embeddings: Any) -> Any:
       """Create vector store from documents"""
       pass

   @abstractmethod 
   def save(self, path: str):
       """Save vector store"""
       pass

   @abstractmethod
   def load(self, path: str) -> Any:
       """Load vector store"""
       pass

class FAISSVectorStore(VectorStore):
   """FAISS vector store implementation"""
   
   def create(self, documents: List[Document], embeddings: Any) -> FAISS:
       return FAISS.from_documents(documents, embeddings)

   def save(self, store: FAISS, path: str):
       store.save_local(path)

   def load(self, path: str, embeddings: Any) -> FAISS:
       return FAISS.load_local(path, embeddings)

class PineconeVectorStore(VectorStore):
   """Pinecone vector store implementation"""
   
   def create(self, documents: List[Document], embeddings: Any) -> Pinecone:
       # Add Pinecone initialization
       # index = pinecone.Index("index-name") 
       return Pinecone.from_documents(documents, embeddings) #, index=index)
       
   def save(self, store: Pinecone, path: str):
       # Pinecone is cloud-based, no need to save
       pass

   def load(self, path: str, embeddings: Any) -> Pinecone:
       # Initialize connection to existing Pinecone index
       pass

class WeaviateVectorStore(VectorStore):
   """Weaviate vector store implementation"""
   
   def create(self, documents: List[Document], embeddings: Any) -> Weaviate:
       # Add Weaviate client initialization
       # client = weaviate.Client(...)
       return Weaviate.from_documents(documents, embeddings) #, client)

   def save(self, store: Weaviate, path: str):
       # Cloud-based, no need to save
       pass

   def load(self, path: str, embeddings: Any) -> Weaviate:
       # Initialize connection to existing Weaviate instance
       pass

class VectorStoreCreator:
   def __init__(self, folder_path: str, store_type: str = "faiss"):
       """
       Initialize the VectorStoreCreator.
       
       Args:
           folder_path (str): Path to folder containing JSON files
           store_type (str): Type of vector store to use ('faiss', 'pinecone', 'weaviate', etc)
       """
       self.folder_path = folder_path
       self.hf_embeddings = HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2"
       )
       self.vector_store = None
       self.store_type = store_type
       
       # Vector store factory
       self.store_implementations = {
           "faiss": FAISSVectorStore(),
           "pinecone": PineconeVectorStore(),
           "weaviate": WeaviateVectorStore(),
       }

   def create(self) -> Union[FAISS, Pinecone, Weaviate]:
       """
       Create and return a vector store from JSON files.
       """
       store_impl = self.store_implementations.get(self.store_type)
       if not store_impl:
           raise ValueError(f"Unsupported vector store type: {self.store_type}")

       print("getting documents")

       documents = self._process_json_files()
       chunks = self._get_chunks(documents)

       print("Creating store")

       self.vector_store = store_impl.create(chunks, self.hf_embeddings)

       return self.vector_store

   def _process_json_files(self) -> List[Document]:
       """Process JSON files - same as before"""
       documents = []
       json_files = [f for f in os.listdir(self.folder_path) if f.endswith('.json')]
       
       if not json_files:
           raise ValueError(f"No JSON files found in {self.folder_path}")

       print(f"Found {len(json_files)} JSON files")

       for file_name in json_files:
           file_path = os.path.join(self.folder_path, file_name)
           try:
               with open(file_path, 'r') as file:
                   data = json.load(file)
                   json_content= data['study'] + data['indication'] + data['content']
                   content_str = json.dumps(json_content, indent=2)
                   doc = Document(
                       page_content=content_str,
                       #TO DO MODIFY THIS
                       metadata={
                           "source": file_name,
                           "file_path": file_path,
                           "type": "json"
                       }
                   )
                   documents.append(doc)
                   
           except json.JSONDecodeError as e:
               print(f"Error parsing JSON file {file_name}: {str(e)}")
           except Exception as e:
               print(f"Error processing file {file_name}: {str(e)}")

       return documents

   def _get_chunks(self, documents: List[Document]) -> List[Document]:
       """Split documents into chunks - same as before"""
       text_splitter = CharacterTextSplitter(
           chunk_size=1000,
           chunk_overlap=200,
           separator="\n"
       )
       
       chunked_documents = []
       for doc in documents:
           chunks = text_splitter.split_text(doc.page_content)
           chunked_documents.extend([
               Document(
                   page_content=chunk,
                   metadata=doc.metadata
               ) for chunk in chunks
           ])
       
       print(f"Created {len(chunked_documents)} chunks")
       return chunked_documents

   def save(self, save_path: str):
       #TODO add naming etc

       """Save the vector store"""
       if self.vector_store is None:
           raise ValueError("Vector store has not been created yet. Call create() first.")
           
       store_impl = self.store_implementations.get(self.store_type)
       store_impl.save(self.vector_store, save_path)
       print("completed saving vector store")

   def load(self, load_path: str):
       """Load a vector store"""
       #TO DO SAVE AND LOAD ERROR to be fixed for some reason may be update packages 
       store_impl = self.store_implementations.get(self.store_type)
       self.vector_store = store_impl.load(load_path, self.hf_embeddings)
       return self.vector_store


if __name__ == "__main__":

    test_folder='backend/AI/reports/'
    vector_folder='backend/AI/vectorstore/'
    vector_store_creator = VectorStoreCreator(test_folder, store_type="faiss")
    vector_store=vector_store_creator.create()
    vector_store_creator.save(vector_folder)

    

    