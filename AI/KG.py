from py2neo import Graph
import re

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
