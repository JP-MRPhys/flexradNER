import os
import quickumls
from quickumls import QuickUMLS
import os
from file import read_json, write_json, extract_terms, process_json
import pandas as pd



class UMLSKWExtractor:
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
        return keywords


# Example Usage
if __name__ == "__main__":


     # Path to QuickUMLS index
    quickumls_path = '/Users/njp60/Downloads/umls/umls_index/'
    # Initialize the extractor
    extractor = UMLSKWExtractor(quickumls_path)
    
    # Extract UMLS keywords
     # Print the extracted keywords
    
    num_files=1
    for i in range(1, num_files + 1):
      
       input_filename=f'data/reports/report{i}.json'
       text=read_json(input_filename)
       print("\n NEW REPORT \n")
       keywords = extractor.extract_keywords(text)
       print(keywords['term'].to_list())
       
    


       
    
    
   