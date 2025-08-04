# test_request.py

import requests

url = 'http://127.0.0.1:5000/predict'

# Example input: sepal and petal lengths/widths for Iris
payload = {
    'features': [5.1, 3.5, 1.4, 0.2]  # Replace with actual test input
}

response = requests.post(url, json=payload)
print("Prediction:", response.json())
