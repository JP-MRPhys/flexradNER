from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import Ollama 
from templates import Templates
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

#import quickumls
from langchain.vectorstores import FAISS
from neo4j import GraphDatabase
from langchain.llms import OpenAI
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np




class LLM:

    def __init__(self, model_name='llama3.1', retriever=None, ontology_path=None) -> None:
        self.llm = Ollama(model=model_name)
        self.templates = Templates()
        self.retriever = retriever
        self.ontology_path = ontology_path

        # Set up BioBERT for keyword extraction and explanation
        self.biobert_model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1")
        self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
               # Initialize a pipeline for Named Entity Recognition (NER)
        self.ner_pipeline = pipeline("ner", model=self.biobert_model, tokenizer=self.biobert_tokenizer, grouped_entities=True)
        
        # Set up QuickUMLS for ontology-based keyword extraction
        #self.umls_matcher = quickumls.QuickUMLS(quickumls_fp=ontology_path)

        # Chains
        self.summarize_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_summary)
        self.keyword_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_keywords)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_explanation)
        self.keyword_explanation_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_keywords_explanation)

        if retriever is not None:
            self.rag_QA_chain = self.create_RAG_QA(self.retriever, self.templates.prompt_RAG_QA)

        print("Model and ontology initialized successfully.")

    def create_RAG_QA(self, retriever, prompt):
        """Create a retrieval-based question-answering system."""
        if retriever is not None:
            rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
        else:
            rag_chain = NotImplementedError
        return rag_chain
    

    def extract_keywords_with_biobert(self, text):
        """
        Extracts medical keywords from the provided text using BioBERT's NER pipeline.
        
        :param text: Medical report as input.
        :return: A list of extracted keywords with their associated categories.
        """
        # Apply the NER pipeline to extract entities
        ner_results = self.ner_pipeline(text)
        keywords = []
        for entity in ner_results:
            # Extract the entity's word and its type
            word = entity['word']
            entity_type = entity.get('entity_group', 'Unknown')
            score = entity.get('score', 0.0)
            
            # Filter out low-confidence entities (threshold can be adjusted)
            if score > 0.5:  # You can tune this threshold
                keywords.append({
                    "term": word,
                    "category": entity_type,
                    "confidence": score
                })

        return keywords

    def extract_keywords_with_ontology(self, text):
        """Extract keywords using QuickUMLS."""
        matches = self.umls_matcher.match(text, best_match=True, ignore_syntax=False)
        extracted_keywords = [{"term": match["ngram"], "cui": match["cui"], "similarity": match["similarity"]} for match in matches]

        return {"keywords": extracted_keywords}

    def get_keywords_explanation(self, text):
        """Extract keywords and their explanations from the provided text."""
        biobert_keywords = self.extract_keywords_with_biobert(text)
        ontology_keywords = self.extract_keywords_with_ontology(text)
        
        # Combine and explain
        combined_keywords = biobert_keywords["keywords"] + [k["term"] for k in ontology_keywords["keywords"]]
        response = self.keyword_explanation_chain.run({"text": text, "keywords": combined_keywords})

        return response

    def get_summary(self, text: str):
        """Get a summary of the provided text."""
        return self.summarize_chain.invoke({"text": text})

    def get_keywords(self, text: str):
        """Extract keywords and their explanations from the provided text."""
        biobert_keywords = self.extract_keywords_with_biobert(text)
        ontology_keywords = self.extract_keywords_with_ontology(text)
        return {"biobert_keywords": biobert_keywords, "ontology_keywords": ontology_keywords}

    def get_explanation(self, text: str):
        """Extract explanations from the provided keywords."""
        return self.explanation_chain.run({"text": text})

    def rag_query(self, context, query):
        """Query the RAG pipeline."""
        response = self.rag_QA_chain({"context": context, "query": query})
        return {"answer": response["result"]}

    def chat(self, question: str):
        """Engage in a chat using the QA system."""
        return self.qa_system.run(question)



class TextEmbeddingModel:
    """
    Handles text embeddings using Hugging Face models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the text embedding model.

        :param model_name: Hugging Face model name for text embeddings.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        """
        Generate an embedding for a given text.

        :param text: The input text.
        :return: A numpy array representing the text embedding.
        """
        return self.model.encode(text, convert_to_numpy=True)


class ImageEmbeddingModel:
    """
    Handles image embeddings using Hugging Face CLIP models.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the image embedding model.

        :param model_name: Hugging Face model name for image embeddings.
        """
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def embed(self, image: Image.Image):
        """
        Generate an embedding for a given image.

        :param image: The input image (PIL format).
        :return: A numpy array representing the image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().numpy()


class RAGPipeline:
    """
    RAG Pipeline with separate text and image embeddings.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        text_embedding_model: TextEmbeddingModel,
        image_embedding_model: ImageEmbeddingModel,
        llm: OpenAI,
    ):
        """
        Initialize the RAG pipeline.

        :param neo4j_uri: URI for the Neo4j database.
        :param neo4j_user: Username for the Neo4j database.
        :param neo4j_password: Password for the Neo4j database.
        :param text_embedding_model: Instance of TextEmbeddingModel for text embeddings.
        :param image_embedding_model: Instance of ImageEmbeddingModel for image embeddings.
        :param llm: Instance of an LLM (e.g., OpenAI or Hugging Face model).
        """
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model
        self.vectorstore = FAISS(self.text_embedding_model.embed, "flat")  # FAISS for vectorstore
        self.llm = llm

    def close(self):
        """Close the Neo4j connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def add_text_to_graph(self, entity_type: str, entity_name: str, text: str):
        """
        Add text knowledge to the Neo4j graph and the vectorstore.

        :param entity_type: Type of the entity (e.g., 'Disease', 'Symptom').
        :param entity_name: Name of the entity.
        :param text: Textual description to store.
        """
        # Add to Neo4j
        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.text = $text
        """
        with self.neo4j_driver.session() as session:
            session.run(query, entity_name=entity_name, text=text)

        # Add to vectorstore
        embedding = self.text_embedding_model.embed(text)
        self.vectorstore.add_texts([text], metadatas=[{"entity_type": entity_type, "entity_name": entity_name}])

    def add_image_to_graph(self, entity_type: str, entity_name: str, image: Image.Image):
        """
        Add image knowledge to the Neo4j graph and the vectorstore.

        :param entity_type: Type of the entity (e.g., 'MedicalImage', 'SymptomImage').
        :param entity_name: Name of the entity.
        :param image: PIL image to store.
        """
        # Generate image embedding
        embedding = self.image_embedding_model.embed(image)

        # Add to Neo4j
        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.image_embedding = $embedding
        """
        with self.neo4j_driver.session() as session:
            session.run(query, entity_name=entity_name, embedding=embedding.tolist())

        # Add to vectorstore
        self.vectorstore.add_texts(["Image for " + entity_name], metadatas=[{"entity_type": entity_type}])

    def retrieve_context(self, query: str, top_k: int = 5):
        """
        Retrieve context from the vectorstore based on a query.

        :param query: User query.
        :param top_k: Number of top documents to retrieve.
        :return: List of relevant documents.
        """
        return self.vectorstore.similarity_search(query, k=top_k)

    def generate_response(self, query: str, top_k: int = 5):
        """
        Generate a response using context retrieved from the knowledge graph.

        :param query: User's query.
        :param top_k: Number of top documents to use as context.
        :return: Response string.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_context(query, top_k=top_k)

        # Combine documents into a single prompt
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response with context
        response = self.llm.generate_response(f"Context:\n{context}\n\nQuery: {query}\nAnswer:")
        return response

if __name__ == "__main__":


    # Initialize text and image embedding models
    text_embedding_model = TextEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    image_embedding_model = ImageEmbeddingModel(model_name="openai/clip-vit-base-patch32")

    # Initialize the pipeline
    rag_pipeline = RAGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        text_embedding_model=text_embedding_model,
        image_embedding_model=image_embedding_model,
        llm=OpenAI(openai_api_key="your_openai_api_key")
    )

    try:
        # Add text and images to the graph
        rag_pipeline.add_text_to_graph("Disease", "Diabetes", "Diabetes is a chronic condition...")
        rag_pipeline.add_image_to_graph("MedicalImage", "Retinal Scan", Image.open("retinal_scan.jpg")) #TODO image is DICOM Image

        # Generate a response
        query = "What is diabetes and how is it related to retinal scans?"
        response = rag_pipeline.generate_response(query=query, top_k=5)
        print("Response:", response)

    finally:
        rag_pipeline.close()





















if __name__ == "__main__":
     
    model=LLM()