import requests
import json
end_point = "http://127.0.0.1:5000/image_recommendation"
data = {
    "image":"C:\\Users\\Pratik\\Desktop\\image_recom\\tshirt.jpeg"
}
payload = json.dumps(data)
headers = {'Content-Type': 'application/json'}
response = requests.post(end_point, data = payload, headers = headers)
print(response.json())