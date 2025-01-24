import requests
import json

# Load the API metadata
metadata_url = "https://smart-api.info/api/metadata/a6b575139cfd429b0a87f825a625d036?raw=1"
metadata_response = requests.get(metadata_url)
metadata = metadata_response.json()

# Extract the base URL and paths from metadata
base_url = metadata['servers'][0]['url']
endpoints = metadata['paths']

# Example: Using a specific endpoint
endpoint = "/query"  # Replace with an endpoint from the metadata
url = base_url + endpoint

# Example payload based on metadata parameters
payload = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"id": "MONDO:0005737", "category": "biolink:Disease"},
                "n1": {"category": "biolink:Gene"}
            },
            "edges": {
                "e0": {"subject": "n0", "object": "n1"}
            }
        }
    }
}

# Make the API request
response = requests.post(url, json=payload)

# Handle the response
if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}: {response.text}")
