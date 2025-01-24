# core/rtx_rag.py
import numpy as np
from typing import List, Dict, Any, Optional
import requests
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class RTXKG2RAG:
    """Core RTX-KG2 RAG system that other components can build upon"""
    
    def __init__(self, base_url: str = "https://arax.ncats.io/api/rtx/v1"):
        self.base_url = base_url
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        self.knowledge_cache = {}
        
    def query_kg(self, query: str) -> Dict[str, Any]:
        """Query the RTX-KG2 knowledge graph"""
        try:
            cache_key = query.lower().strip()
            if cache_key in self.knowledge_cache:
                return self.knowledge_cache[cache_key]

            endpoint = f"{self.base_url}/query"
            payload = {
                "message": {
                    "query_graph": {
                        "nodes": {},
                        "edges": {}
                    },
                    "query_options": {
                        "max_results": 50
                    }
                }
            }
            
            payload["message"]["query"] = query
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            results = response.json()
            self.knowledge_cache[cache_key] = results
            return results
            
        except Exception as e:
            self.logger.error(f"RTX-KG2 query error: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            return self.encoder.encode(texts, show_progress_bar=False)
        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            raise

    def process_kg_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and standardize KG results"""
        processed = []
        try:
            for result in results.get("message", {}).get("results", []):
                item = {
                    "text": result.get("text", ""),
                    "score": result.get("confidence_score", 0.0),
                    "source": result.get("source", ""),
                    "type": result.get("type", ""),
                    "relationships": self._extract_relationships(result),
                    "timestamp": datetime.now().isoformat()
                }
                processed.append(item)
            return processed
        except Exception as e:
            self.logger.error(f"Results processing error: {str(e)}")
            return []

    def _extract_relationships(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract relationship information from a result"""
        relationships = []
        try:
            edges = result.get("edges", [])
            for edge in edges:
                rel = {
                    "type": edge.get("predicate", ""),
                    "source": edge.get("subject", ""),
                    "target": edge.get("object", ""),
                    "confidence": str(edge.get("confidence_score", 0))
                }
                relationships.append(rel)
        except Exception:
            pass
        return relationships

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge for a query"""
        try:
            # Query KG
            kg_results = self.query_kg(query)
            processed_results = self.process_kg_results(kg_results)
            
            # Generate embeddings
            query_embedding = self.get_embeddings([query])
            result_texts = [r["text"] for r in processed_results]
            result_embeddings = self.get_embeddings(result_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, result_embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_results = []
            
            for idx in top_indices:
                result = processed_results[idx].copy()
                result["similarity"] = float(similarities[idx])
                top_results.append(result)
                
            return top_results
            
        except Exception as e:
            self.logger.error(f"Retrieval error: {str(e)}")
            raise

class MedicalNER:
    """Named Entity Recognition for medical texts using RTX-KG2 RAG"""
    
    def __init__(self, rag_system: RTXKG2RAG):
        self.rag = rag_system
        self.nlp = spacy.load("en_core_sci_md")
        self.logger = logging.getLogger(__name__)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities and enrich with KG information"""
        try:
            # Extract basic entities
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Query KG for entity information
                kg_results = self.rag.retrieve(f"What is {ent.text}?", top_k=1)
                
                entity = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "kg_info": kg_results[0] if kg_results else None
                }
                entities.append(entity)
                
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {str(e)}")
            return []

class MedicalExplainer:
    """Generate explanations for medical concepts using RTX-KG2 RAG"""
    
    def __init__(self, rag_system: RTXKG2RAG):
        self.rag = rag_system
        self.logger = logging.getLogger(__name__)
        self.explanation_cache = {}

    def explain_concept(self, concept: str, context: str = "") -> Dict[str, Any]:
        """Generate explanation for a medical concept"""
        try:
            cache_key = f"{concept.lower()}_{context.lower()}"
            if cache_key in self.explanation_cache:
                return self.explanation_cache[cache_key]

            # Generate queries for different aspects
            queries = [
                f"What is {concept}?",
                f"What are the key characteristics of {concept}?",
                f"How does {concept} relate to {context}?" if context else None
            ]
            
            explanation = {
                "concept": concept,
                "definition": [],
                "characteristics": [],
                "context_relevance": [],
                "sources": set(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get information for each aspect
            for query in queries:
                if query:
                    results = self.rag.retrieve(query, top_k=3)
                    for result in results:
                        if "definition" in query.lower():
                            explanation["definition"].append(result)
                        elif "characteristics" in query.lower():
                            explanation["characteristics"].append(result)
                        elif "relate" in query.lower():
                            explanation["context_relevance"].append(result)
                        explanation["sources"].add(result.get("source", ""))
            
            explanation["sources"] = list(explanation["sources"])
            self.explanation_cache[cache_key] = explanation
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation error: {str(e)}")
            raise

class MedicalQA:
    """Question answering system using RTX-KG2 RAG"""
    
    def __init__(self, rag_system: RTXKG2RAG, ner: MedicalNER, explainer: MedicalExplainer):
        self.rag = rag_system
        self.ner = ner
        self.explainer = explainer
        self.logger = logging.getLogger(__name__)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate comprehensive answer to medical question"""
        try:
            # Get basic answer from RAG
            rag_results = self.rag.retrieve(question, top_k=3)
            
            # Extract entities
            entities = self.ner.extract_entities(question)
            
            # Get explanations for entities
            explanations = {}
            for entity in entities:
                explanation = self.explainer.explain_concept(
                    entity["text"], 
                    context=question
                )
                explanations[entity["text"]] = explanation
            
            # Compile response
            answer = {
                "question": question,
                "direct_answer": rag_results,
                "entities": entities,
                "explanations": explanations,
                "sources": list(set(r.get("source", "") for r in rag_results)),
                "timestamp": datetime.now().isoformat()
            }
            
            return answer
            
        except Exception as e:
            self.logger.error(f"QA error: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize core RAG system
    rag_system = RTXKG2RAG()
    
    # Initialize components
    ner = MedicalNER(rag_system)
    explainer = MedicalExplainer(rag_system)
    qa = MedicalQA(rag_system, ner, explainer)
    
    # Example question
    question = "How does metformin affect blood glucose levels in type 2 diabetes?"
    answer = qa.answer_question(question)
    
    print("Question:", question)
    print("\nDirect Answer:", answer["direct_answer"][0]["text"])
    print("\nIdentified Entities:")
    for entity in answer["entities"]:
        print(f"- {entity['text']} ({entity['type']})")
    print("\nExplanations available for:", list(answer["explanations"].keys()))