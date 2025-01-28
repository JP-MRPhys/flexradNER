from py2neo import Graph
import re
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel, pipeline
import speech_recognition as sr
import pytesseract
from PIL import Image
import numpy as np
from typing import List, Dict


# Initialize the MedicalKnowledgeGraph
graph = MedicalKnowledgeGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
    embedding_model="huggingface"  # Use "openai" for OpenAI embeddings
)

# Initialize the LLM
llm = LLM

# Initialize the RAG Pipeline
rag_pipeline = RAGPipeline(knowledge_graph=graph, llm=llm)

try:
    # Add data to the knowledge graph
    graph.add_medical_text(
        entity_type="Disease",
        entity_name="Diabetes",
        text="Diabetes is a chronic condition associated with high blood sugar levels."
    )

    # User query
    user_query = "What are the symptoms of diabetes?"

    # Generate a RAG response
    response = rag_pipeline.generate_response(query=user_query, threshold=0.8)
    print("Generated Response:\n", response)

finally:
    graph.close()




class MedicalKnowledgeGraph:
    """
    A class to create and manage a medical knowledge graph.
    Supports text, speech-to-text, and image embeddings for integration with other knowledge graphs.
    """

    def __init__(self, uri: str, user: str, password: str, embedding_model: str = "openai"):
        """
        Initialize the MedicalKnowledgeGraph instance.

        :param uri: Neo4j URI (e.g., "bolt://localhost:7687").
        :param user: Neo4j username.
        :param password: Neo4j password.
        :param embedding_model: The embedding model to use (e.g., 'openai', 'huggingface').
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Initialize the embedding model
        if embedding_model == "openai":
            import openai
            self.embedding_function = lambda text: openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
        elif embedding_model == "huggingface":
            self.text_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_function = lambda text: np.mean(self.text_embedder(text), axis=1).tolist()

        # Initialize CLIP model for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def add_medical_text(self, entity_type: str, entity_name: str, text: str):
        """
        Add medical text data to the knowledge graph with embeddings.

        :param entity_type: The type of the entity (e.g., 'Disease', 'Symptom', 'Treatment').
        :param entity_name: The name of the entity.
        :param text: The text data to add.
        """
        embedding = self.embedding_function(text)
        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.text = $text, e.embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(query, entity_name=entity_name, text=text, embedding=embedding)

    def add_speech_to_graph(self, audio_file: str, entity_type: str, entity_name: str):
        """
        Convert speech to text, generate embeddings, and add it to the knowledge graph.

        :param audio_file: Path to the audio file.
        :param entity_type: The type of the entity.
        :param entity_name: The name of the entity.
        """
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            self.add_medical_text(entity_type, entity_name, text)

    def add_image_to_graph(self, image_path: str, entity_type: str, entity_name: str):
        """
        Generate embeddings from an image and add it to the knowledge graph.

        :param image_path: Path to the image file.
        :param entity_type: The type of the entity.
        :param entity_name: The name of the entity.
        """
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)
        embedding = outputs[0].detach().numpy().tolist()

        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.image_embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(query, entity_name=entity_name, embedding=embedding)

    def integrate_graphs(self, other_graph_data: List[Dict]):
        """
        Integrate external knowledge graph data into the current graph.

        :param other_graph_data: A list of dictionaries representing nodes and relationships.
        """
        with self.driver.session() as session:
            for data in other_graph_data:
                query = """
                MERGE (e:Entity {id: $id})
                SET e += $properties
                """
                session.run(query, id=data["id"], properties=data.get("properties", {}))

    def query_similar_entities(self, embedding: List[float], threshold: float = 0.8) -> List[Dict]:
        """
        Query for similar entities in the graph based on embedding similarity.

        :param embedding: The embedding to compare against.
        :param threshold: The similarity threshold.
        :return: A list of similar entities.
        """
        query = """
        MATCH (e)
        WHERE gds.similarity.cosine(e.embedding, $embedding) >= $threshold
        RETURN e.name AS name, e.text AS text, gds.similarity.cosine(e.embedding, $embedding) AS similarity
        """
        with self.driver.session() as session:
            results = session.run(query, embedding=embedding, threshold=threshold)
            return [record.data() for record in results]





class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline that integrates the MedicalKnowledgeGraph
    with a language model to generate responses based on a knowledge graph.
    """

    def __init__(self, knowledge_graph: MedicalKnowledgeGraph, llm):
        """
        Initialize the RAG pipeline.

        :param knowledge_graph: An instance of the MedicalKnowledgeGraph class.
        :param llm: A language model class with a `generate_response(prompt)` method.
        """
        self.knowledge_graph = knowledge_graph
        self.llm = llm

    def retrieve_context(self, query: str, threshold: float = 0.8) -> List[Dict]:
        """
        Retrieve context from the knowledge graph using embedding similarity.

        :param query: The input query string.
        :param threshold: Similarity threshold for retrieving relevant entities.
        :return: List of relevant entities as dictionaries.
        """
        # Generate embedding for the input query
        query_embedding = self.knowledge_graph.embedding_function(query)

        # Query the knowledge graph for similar entities
        similar_entities = self.knowledge_graph.query_similar_entities(
            embedding=query_embedding, threshold=threshold
        )
        return similar_entities

    def format_context(self, entities: List[Dict]) -> str:
        """
        Format the retrieved context into a string for the language model.

        :param entities: List of entities retrieved from the knowledge graph.
        :return: Formatted context string.
        """
        context = "\n".join(
            [f"- {entity['name']}: {entity.get('text', 'No description')}" for entity in entities]
        )
        return context

    def generate_response(self, query: str, threshold: float = 0.8) -> str:
        """
        Generate a response by combining context retrieval and LLM generation.

        :param query: The user's query.
        :param threshold: Similarity threshold for retrieving relevant entities.
        :return: Generated response string.
        """
        # Step 1: Retrieve context from the knowledge graph
        similar_entities = self.retrieve_context(query, threshold)

        # Step 2: Format the context for the language model
        context = self.format_context(similar_entities)

        # Step 3: Create the prompt and generate a response using the LLM
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.generate_response(prompt)
        return response













class MedicalNER:
    def __init__(self, llm, kg_url, username, password):
        """
        Initialize the Medical NER system.
        
        :param llm: Instance of the LLM class for explanations.
        :param kg_url: URL of the Neo4j RTX-KG2 database.
        :param username: Username for Neo4j.
        :param password: Password for Neo4j.
        """
        self.llm = llm
        self.graph = Graph(kg_url, auth=(username, password))

    def query_kg(self, term):
        """
        Query RTX-KG2 to check if the term exists in the knowledge graph.
        
        :param term: Medical term to look up.
        :return: A list of matching entities with their categories.
        """
        query = f"""
        MATCH (n)
        WHERE n.name =~ '(?i).*{re.escape(term)}.*'
        RETURN n.name AS entity, n.category AS category
        LIMIT 10
        """
        results = self.graph.run(query).data()
        return results

    def extract_entities(self, medical_text):
        """
        Perform Named Entity Recognition (NER) on the input medical text using RTX-KG2.
        
        :param medical_text: Input text (e.g., a medical report).
        :return: List of extracted entities with their categories and confidence scores.
        """
        # Tokenize the text into terms (basic splitting, can be improved with NLP libraries)
        terms = re.findall(r'\b\w+\b', medical_text)

        # Query KG for each term and collect entities
        entities = []
        for term in terms:
            matches = self.query_kg(term)
            for match in matches:
                entities.append({
                    "term": match["entity"],
                    "category": match["category"],
                    "confidence": 0.9  # Placeholder confidence score, adjust as needed
                })
        return entities

    def explain_entities(self, entities):
        """
        Use LLM to generate explanations for the extracted entities.
        
        :param entities: List of extracted entities.
        :return: Entities with explanations.
        """
        explanations = self.llm.generate_batch_explanation(entities)
        return explanations

    def process_report(self, medical_report):
        """
        Process a medical report to extract and explain entities.
        
        :param medical_report: Input text (e.g., a medical report).
        :return: List of entities with explanations.
        """
        # Step 1: Extract entities
        extracted_entities = self.extract_entities(medical_report)

        # Step 2: Generate explanations for entities
        entities_with_explanations = self.explain_entities(extracted_entities)

        return entities_with_explanations
    


class MedicalKnowledgeGraph:
    """
    A class to create and manage a medical knowledge graph.
    Supports text, speech-to-text, and image embeddings for integration with other knowledge graphs.
    """

    def __init__(self, uri: str, user: str, password: str, embedding_model: str = "openai"):
        """
        Initialize the MedicalKnowledgeGraph instance.

        :param uri: Neo4j URI (e.g., "bolt://localhost:7687").
        :param user: Neo4j username.
        :param password: Neo4j password.
        :param embedding_model: The embedding model to use (e.g., 'openai', 'huggingface').
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Initialize the embedding model
        if embedding_model == "openai":
            import openai
            self.embedding_function = lambda text: openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
        elif embedding_model == "huggingface":
            self.text_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_function = lambda text: np.mean(self.text_embedder(text), axis=1).tolist()

        # Initialize CLIP model for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def add_medical_text(self, entity_type: str, entity_name: str, text: str):
        """
        Add medical text data to the knowledge graph with embeddings.

        :param entity_type: The type of the entity (e.g., 'Disease', 'Symptom', 'Treatment').
        :param entity_name: The name of the entity.
        :param text: The text data to add.
        """
        embedding = self.embedding_function(text)
        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.text = $text, e.embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(query, entity_name=entity_name, text=text, embedding=embedding)

    def add_speech_to_graph(self, audio_file: str, entity_type: str, entity_name: str):
        """
        Convert speech to text, generate embeddings, and add it to the knowledge graph.

        :param audio_file: Path to the audio file.
        :param entity_type: The type of the entity.
        :param entity_name: The name of the entity.
        """
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            self.add_medical_text(entity_type, entity_name, text)

    def add_image_to_graph(self, image_path: str, entity_type: str, entity_name: str):
        """
        Generate embeddings from an image and add it to the knowledge graph.

        :param image_path: Path to the image file.
        :param entity_type: The type of the entity.
        :param entity_name: The name of the entity.
        """
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)
        embedding = outputs[0].detach().numpy().tolist()

        query = f"""
        MERGE (e:{entity_type} {{name: $entity_name}})
        SET e.image_embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(query, entity_name=entity_name, embedding=embedding)

    def integrate_graphs(self, other_graph_data: List[Dict]):
        """
        Integrate external knowledge graph data into the current graph.

        :param other_graph_data: A list of dictionaries representing nodes and relationships.
        """
        with self.driver.session() as session:
            for data in other_graph_data:
                query = """
                MERGE (e:Entity {id: $id})
                SET e += $properties
                """
                session.run(query, id=data["id"], properties=data.get("properties", {}))

    def query_similar_entities(self, embedding: List[float], threshold: float = 0.8) -> List[Dict]:
        """
        Query for similar entities in the graph based on embedding similarity.

        :param embedding: The embedding to compare against.
        :param threshold: The similarity threshold.
        :return: A list of similar entities.
        """
        query = """
        MATCH (e)
        WHERE gds.similarity.cosine(e.embedding, $embedding) >= $threshold
        RETURN e.name AS name, e.text AS text, gds.similarity.cosine(e.embedding, $embedding) AS similarity
        """
        with self.driver.session() as session:
            results = session.run(query, embedding=embedding, threshold=threshold)
            return [record.data() for record in results]












# Usage example
if __name__ == "__main__":
    # Initialize LLM
    llm = LLM(model_name="llama3.1")

    # Initialize MedicalNER with RTX-KG2 Neo4j database credentials
    ner = MedicalNER(
        llm=llm,
        kg_url="bolt://localhost:7687",  # Replace with your Neo4j instance URL
        username="neo4j",
        password="password"
    )

    # Example medical report
    medical_report = """
    MRI Brain Report: No evidence of mass effect, hemorrhage, or infarction. 
    Normal appearance of gray-white differentiation. No significant findings in ventricles or CSF spaces.
    """

    # Process the report
    results = ner.process_report(medical_report)

    # Print the results
    for result in results:
        print(f"Term: {result['term']}, Category: {result['category']}, Explanation: {result['explanation']}")
