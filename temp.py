import requests

# Set the URL of your Django server
url = 'http://127.0.0.1:8000/query/upload_document/'  # Replace with your server's URL

# Replace 'path/to/your/document.txt' with the actual path to your document file
files = {'document': open('./tempdoc.txt', 'rb')}

# Send a POST request with the document file attached
response = requests.post(url, files=files)

# Print the response from the server
print(response.json())
