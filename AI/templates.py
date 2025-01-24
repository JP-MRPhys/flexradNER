
from langchain.prompts import PromptTemplate


class Templates:
    def __init__(self):
        # Template for summarization
        self.summary = """You are an intelligent chatbot to help summarize a medical report.
        Text: {text}
        Answer:"""

        # Template for keyword detection
        self.keywords = """You are a highly intelligent medical assistant specializing in analyzing medical reports to detect keyword and context. 

        Your task is to:  
        1. Identify all key medical, technical, and clinical terms in {text}  
        2. Include relevant context as a text by as caterogy of context  

        For each term you identify, provide:  
        - The term or context itself
        - The category of the term (e.g., organ, technical term, medical condition, imaging finding, etc.)

        Return your analysis formatted as a proper JSON. Each term should be an object in an array named.
        
        Keywords:"""

        self.explanation =  """
            Analyze the following medical text and provided keywords, focusing on medical terminology, 
            equipment, organs, diseases, and their clinical context. Provide a comprehensive explanation 
            for each keyword.

            Text: {input_text}
    
            Keywords: {keywords}

             For each term you identify, provide:  
                 - The term itself
                 - The category of the term (e.g., organ, technical term, medical condition, imaging finding, etc.)
                 - A brief explanation of the term
                 - Its relevance in medical imaging or diagnostics
                 - How it relates to the broader findings or impressions in the report
                 -Each term should be an object in an array named "terms"
             
            Return your analysis formatted as a proper JSON. Don't include input text or keywords list or additional comments in your output
   
            """



        self.explanation_advance = """You are a highly intelligent medical assistant specializing in analyzing medical reports and providing clear, concise explanations of keyword. 

        Your task is to:  
        For each term you identify, provide:  
        - A brief explanation of the words each of the keywords provided {keywords}
        - For each its relevance in medical imaging or diagnostics or medical term
        - How it relates to the broader findings or impressions in the report {text} e.g increasing or decreasing may be good or bad 

         For each term you identify, provide:  
        - The term itself
        - The category of the term (e.g., organ, technical term, medical condition, imaging finding, etc.)
        - A brief explanation of the term
        - Its relevance in medical imaging or diagnostics
        - How it relates to the broader findings or impressions in the report

        Return your analysis formatted as a proper JSON. Each term should be an object in an array named "terms"
        Explanation:"""


        # Template for keyword detection + explanation
        self.keywords_explanation = """You are a highly intelligent medical assistant specializing in analyzing medical reports and providing clear, concise explanations of key terms. 

        Your task is to:  
        1. Identify all key medical, technical, relevant context and clinical terms in {text}  
        2. Include relevant context from the report to enhance the explanation of each term.  

        For each term you identify, provide:  
        - The term itself
        - The category of the term (e.g., organ, technical term, medical condition, imaging finding, etc.)
        - A brief explanation of the term
        - Its relevance in medical imaging or diagnostics
        - How it relates to the broader findings or impressions in the report

        Return your analysis formatted as a proper JSON. Each term should be an object in an array named "terms".
        Answer:"""

        self.RAG_QA="Use the following context to answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"

        self.prompt_summary = PromptTemplate(template=self.summary, input_variables=["text"])
        self.prompt_keywords = PromptTemplate(template=self.keywords, input_variables=["text"])
        self.prompt_explanation = PromptTemplate(template=self.explanation, input_variables=["text", "keywords"])
        self.prompt_keywords_explanation = PromptTemplate(template=self.keywords_explanation, input_variables=["text"])
        self.prompt_RAG_QA =PromptTemplate(template=self.RAG_QA, input_variables=["context", "query"])
