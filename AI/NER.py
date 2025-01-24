from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import Ollama 
from templates import Templates
from quickumls import QuickUMLS
from file import read_json, write_json, extract_terms, process_json
import pandas as pd
from utils import getulmsdir

quickumls_path = getulmsdir()  #TODO add a config.file

class UMLS_NER:
    def __init__(self, quickumls_path):
        """
        Initialize the QuickUMLS extractor with the given UMLS index path.
        
        :param quickumls_path: Path to the QuickUMLS index folder containing the UMLS index files.
        """
        # Load QuickUMLS
        self.QUICKUMLS = QuickUMLS(quickumls_path, threshold=0.7)  # Adjust threshold as needed

    def extract_keywords(self, text):
        
        """
        Extracts UMLS-based medical keywords from the provided text using QuickUMLS.
        
        :param text: Medical report as input.
        :return: A list of extracted keywords with their associated UMLS concept IDs and types.
        """
        # Apply QuickUMLS to extract UMLS concepts
        matches = self.QUICKUMLS.match(text)

        data=[]

        for match in matches:
            for row in match:
                data.append(row)

        keywords=pd.DataFrame(data)    
        keywords = keywords.drop_duplicates('term').reset_index(drop=True)

        print(keywords['term'].to_list())


        return keywords, keywords['term'].to_list()

class BIOBERT_NER:
    def __init__(self):
        # Load BioBERT model for token classification
        self.model_name = "dmis-lab/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # Initialize a pipeline for Named Entity Recognition (NER)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, grouped_entities=True)

    def extract_keywords(self, text):
        """
        Extracts medical keywords from the provided text using BioBERT's NER pipeline.
        
        :param text: Medical report as input.
        :return: A list of extracted keywords with their associated categories.
        """
        # Apply the NER pipeline to extract entities
        ner_results = self.ner_pipeline(text)

        
        keywords = [] 
        keywords_str = []
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

                keywords_str.append(word)
            

        return keywords, keywords_str
    
class NER:
    def __init__(self, model_name='llama3.1', retriever=None, quickumls_path = '/Users/njp60/Downloads/umls/umls_index/'):
        self.llm = Ollama(model=model_name)
        self.templates = Templates()
        self.retriever = retriever

        # Set up the chains for each functionality
        self.summarize_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_summary)
        self.keyword_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_keywords)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_explanation)
        self.keyword_explanation_chain = LLMChain(llm=self.llm, prompt=self.templates.prompt_keywords_explanation)
        self.extractor1 =BIOBERT_NER()
        self.extractor2 = UMLS_NER(quickumls_path)

        if retriever is not None:
            self.rag_QA_chain = self.create_RAG_QA(self.retriever, self.templates.prompt_RAG_QA)
    
    def generate_explanation(self, input_text, keywords_str):
        """
        Generate an explanation for a given medical keyword using LLaMA.
        
        :param keyword: A dictionary containing the term and its category.
        :return: An explanation string.
        """

        response = self.explanation_chain.invoke({"input_text": input_text,"keywords": keywords_str })
        #print(response)
        return response

    def get_keywords_with_explanations(self, medical_report):
        """
        Extract keywords from a medical report and generate explanations for them.
        
        :param medical_report: Input text (medical report).
        :return: A list of keywords with explanations.
        """
        # Step 1: Extract keywords using the MedicalKeywordExtractor
        
        keywords1, keywords_str1 = self.extractor1.extract_keywords(medical_report)
        keywords1, keywords_str2 = self.extractor2.extract_keywords(medical_report)
        explanations1=self.generate_explanation(medical_report,keywords_str1)
        explanations2=self.generate_explanation(medical_report,keywords_str2)
        #TODO: write the keywords in JSON format
        #TODO: Remove common Keywords
        #TODO: Refactor the code

        return (keywords_str1, explanations1, keywords_str2, explanations2)


if __name__ == "__main__":
    
    num_files=10
    model=NER()

    for i in range(1, num_files + 1):
      
       input_filename=f'data/reports/report{i}.json'
       text=read_json(input_filename)
       print("\n NEW REPORT \n")
       data=model.get_keywords_with_explanations(text)

       explanation1=data[1]
       explanation2=data[3]
       print(explanation2)
       
       
       output_filename1=f'data/outputs/explanations{i}_BIOBERT.json'
       output_filename2=f'data/outputs/explanations{i}_UMLS.json'
       #terms_filename=f'data/outputs/terms{i}.json'
       write_json(output_filename1,explanation1)
       write_json(output_filename2,explanation2)


       #process_json(output_filename, terms_filename)
       
      
       