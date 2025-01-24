from app import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from datetime import datetime
import logging
from AI.NER import NER
app = Flask(__name__)
CORS(app)


quickumls_path = 'umls_index/'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalAnalyzer:
    def __init__(self):
        # Sample medical keywords and their explanations
        self.NER=NER()
        
    def detect_keywords(self, text):
        """Detect medical keywords in text and provide explanations. and provide output json explanations"""

        data=NER.get_keywords_with_explanations(text)

        explanation_BIOBERT=data[1]
        explanation_ULMS=data[3]

        return explanation_BIOBERT, explanation_ULMS
    
    def analyze_report(self, text):
        """Analyze medical report text."""
        keywords = self.detect_keywords(text)
        
        analysis = {
            "detected_terms": keywords,
            "term_count": len(keywords),
            "timestamp": datetime.now().isoformat()
        }
        return analysis

class ChatManager:
    def __init__(self):
        self.conversation_history = {}
    
    def process_message(self, user_id, message):
        """Process a chat message and maintain conversation history."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Simple response generation (replace with more sophisticated logic)
        return {
            "response": "Thank you for your message. How can I assist you further?",
            "timestamp": datetime.now().isoformat()
        }

class ImageAnalyzer:
    def analyze_image(self, image_data):
        """Analyze medical images."""
        # Placeholder for image analysis logic
        analysis = {
            "image_received": True,
            "image_size": len(image_data) if image_data else 0,
            "analysis_timestamp": datetime.now().isoformat(),
            "findings": "Image analysis would be implemented here"
        }
        return analysis

# Initialize components
medical_analyzer = MedicalAnalyzer()  #These will be the services we build
chat_manager = ChatManager()
image_analyzer = ImageAnalyzer()

@app.route('/api/NER', methods=['POST'])
def analyze_medical_report():
    """Endpoint for analyzing medical reports with keyword detection."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        analysis = medical_analyzer.analyze_report(data['text'])
        return jsonify(analysis)
    
    except Exception as e:
        logger.error(f"Error in analyze_medical_report: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint for chat functionality."""
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'user_id' not in data:
            return jsonify({"error": "Invalid request data"}), 400
        
        response = chat_manager.process_message(data['user_id'], data['message'])
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Endpoint for specific medical queries."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        # Process the query (implement specific query logic here)
        response = {
            "query_received": data['query'],
            "timestamp": datetime.now().isoformat(),
            "response": "Query processing would be implemented here"
        }
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/image', methods=['POST'])
def analyze_image():
    """Endpoint for medical image analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        analysis = image_analyzer.analyze_image(image_data)
        return jsonify(analysis)
    
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)