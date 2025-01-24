To test a JSON file response with your Flask API endpoint using a bash script, you can use `curl` to send a POST request with a JSON payload to the API and capture the response. Here's a sample script that assumes you have a JSON file (`input_data.json`) to send to the API and it will output the response.

### Example `test_api.sh` Script:

```bash
#!/bin/bash

# Variables
API_URL="http://localhost:5000/api/NER"  # Change this to your API's URL
JSON_FILE="input_data.json"  # Path to your input JSON file

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file $JSON_FILE not found!"
    exit 1
fi

# Send the POST request with the JSON file
response=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d @"$JSON_FILE")

# Check if the response contains an error or data
if [[ "$response" == *"error"* ]]; then
    echo "API Error Response:"
    echo "$response"
else
    echo "API Response:"
    echo "$response"
fi
```

