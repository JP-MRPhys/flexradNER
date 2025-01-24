import json
import os
import logging
from typing import Dict, Optional, Any

class JSONHandler:
    def __init__(self, log_file: str = "json_operations.log"):
        """
        Initialize JSONHandler with logging configuration.
        
        Args:
            log_file (str): Path to log file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def read_json(self, filename: str) -> str:
        """
        Read and concatenate specific fields from JSON file.
        
        Args:
            filename (str): Path to JSON file
            
        Returns:
            str: Concatenated text from study, indication, and content fields
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            KeyError: If required fields are missing
        """
        try:
            self.logger.info(f"Reading JSON file: {filename}")
            
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")
                
            with open(filename, 'r') as file:
                data = json.load(file)
                
                # Validate required fields
                required_fields = ['study', 'indication', 'content']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    raise KeyError(f"Missing required fields: {missing_fields}")
                
                # Concatenate fields
                text = data['study'] + data['indication'] + data['content']
                
                self.logger.info(f"Successfully read and processed {filename}")
                return text
                
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filename}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {filename}: {str(e)}")
            raise
        except KeyError as e:
            self.logger.error(f"Missing required fields in {filename}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error reading {filename}: {str(e)}")
            raise

    def write_json(self, filename: str, output: Any) -> None:
        """
        Write or update keywords in JSON file.
        
        Args:
            filename (str): Path to JSON file
            output: Data to write to 'keywords' field
            
        Raises:
            IOError: If file cannot be written
            json.JSONDecodeError: If existing file contains invalid JSON
        """
        try:
            self.logger.info(f"Writing to JSON file: {filename}")
            
            # Initialize data dictionary
            data: Dict = {}
            
            # Read existing file if it exists
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as file:
                        data = json.load(file)
                        self.logger.debug(f"Successfully read existing content from {filename}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Existing file {filename} contains invalid JSON, creating new")
                    data = {}
            
            # Update keywords
            data['keywords'] = output
            
            # Write updated data back to file
            with open(filename, 'w') as file:
                json.dump(data, file, indent=2)
                
            self.logger.info(f"Successfully wrote keywords to {filename}")
            
        except IOError as e:
            self.logger.error(f"Error writing to file {filename}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error writing to {filename}: {str(e)}")
            raise

    def update_json(self, filename: str, field: str, value: Any) -> None:
        """
        Update specific field in JSON file.
        
        Args:
            filename (str): Path to JSON file
            field (str): Field to update
            value: New value for the field
        """
        try:
            self.logger.info(f"Updating field '{field}' in {filename}")
            
            data = {}
            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    data = json.load(file)
                    
            data[field] = value
            
            with open(filename, 'w') as file:
                json.dump(data, file, indent=2)
                
            self.logger.info(f"Successfully updated {field} in {filename}")
            
        except Exception as e:
            self.logger.error(f"Error updating {filename}: {str(e)}")
            raise

        