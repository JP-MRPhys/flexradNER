import json
import os
import json
import re
from typing import List, Dict

def read_json(filename):
    # Open and read the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

        text=data['study'] + data['indication'] + data['content']
         
    return text

def write_json(filename, output):
    # Initialize data dictionary
    data = {}
    
    # Check if file exists and read existing content
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # Handle case where file exists but isn't valid JSON
                data = {}
    
    # Add the 'output' field to the data
    data['output'] = output

    # Write the updated JSON back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    return None

import json
import sys
from typing import Dict, List

def extract_terms(input_file: str, output_file: str) -> None:
    """
    Extracts terms from an input JSON file and writes them in a formatted JSON structure to an output file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    try:
        # Load input JSON file
        with open(input_file, "r") as file:
            data = json.load(file)

        # Extract terms
        terms = data.get("output", {}).get("terms", [])
        
        if not isinstance(terms, list):
            raise ValueError("The 'terms' field is missing or not a list in the input file.")

        # Create formatted JSON structure
        formatted_data = {"terms": terms}

        # Write formatted JSON to output file
        with open(output_file, "w") as file:
            json.dump(formatted_data, file, indent=4)
        
        print(f"Formatted terms have been written to {output_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'. Please ensure it is a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clean_text(text: str) -> str:
    """
    Cleans up poorly formatted or tokenized text by removing special characters and extra spaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: Cleaned text.
    """
    # Remove special characters like "##" and extra spaces
    cleaned_text = re.sub(r"\s*##\s*", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def extract_terms(json_data: Dict) -> List[Dict]:
    """
    Extracts terms and their explanations from the provided JSON structure.

    Args:
        json_data (Dict): The input JSON data.

    Returns:
        List[Dict]: List of terms with their metadata.
    """
    terms = []

    # Extract the text and keywords
    input_text = json_data.get("output", {}).get("input_text", "")
    keywords = json_data.get("output", {}).get("keywords", [])
    
    # Combine keywords into a coherent list of terms
    combined_keywords = clean_text(" ".join(keywords))

    # Extract sentences or key phrases from the input text
    for keyword in combined_keywords.split(","):
        keyword = keyword.strip()
        if keyword:
            # Attempt to find a category and explanation for each term
            term_data = {
                "term": keyword,
                "category": "Unknown",  # Placeholder, real category requires domain knowledge
                "explanation": f"Placeholder explanation for {keyword}",
                "relevance": f"Relevance of {keyword} in the context of the input.",
                "relation": f"Relation of {keyword} to findings in the input text."
            }
            terms.append(term_data)
    
    return terms

def process_json(input_file: str, output_file: str) -> None:
    """
    Processes a JSON file to extract terms and save the results in a new JSON file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    try:
        with open(input_file, 'r') as infile:
            data = json.load(infile)
        
        terms = extract_terms(data)
        
        # Prepare the output
        output_data = {"terms": terms}
        
        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)
        
        print(f"Terms successfully extracted and saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")


