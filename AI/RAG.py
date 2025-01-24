from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import Ollama 
from templates import Templates
#import quickumls

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


if __name__ == "__main__":
     
    model=LLM()