from __future__ import print_function
import argparse
import requests
import json
import cv2

import urllib3


addr = 'http://localhost:5000/'
#addr = 'http://20.33.92.85:5000/'

#test_url = addr + 'api/image_description' #endpoint per Anomaly Detection
#url = addr + 'run_reid' 
url = addr + 'run_ad'

# Corpo della richiesta (payload)
data = {
    "task_id": "1756817703.5984938"
}

# Token di sicurezza (sostituisci con il tuo reale token)
security_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ0cmFmZmljLW1hc3Rlci1ub2RlIiwic3ViIjoiMTMyLjE3Ny40OC4xMTMiLCJhdWQiOiJmbGFzay1hcHAtYWtzLW5vZGVwb29sMS0xNzM3OTk5Mi12bXNzMDAwMDFoLXNlcnZpY2UuNC4yMzIuMTYuMTg5Lm5pcC5pbyIsImlhdCI6MTc1NjgxNzgzNCwiZXhwIjoxNzY1NDU3ODM0LCJzY29wZSI6ImNhbWVyYV9hcGkifQ.FH8W2EP1hR0LTCL-Yn4Le8i0TV918xzwj5Wq9XH7KuLevJ-ZtJ1_jk4w0vnfwlAqawzOCwa1KcWPYjmkJiEZ5mPyja5j9FxXbXyOuZFkKoNa8sTTEGYfvPDRNIY3_FjUrkZwR08ctX0UVUbBmnL5XZegz9FHpS5lU9waPeOtWn1-nNTJBStAM_Vrf7LGN9Mf3ZfZpjxsTUQPcFJELNB7Nf--S4dhkXuCEOZIui5lbdobLvzkrrzJl4G-LP0H3_h7taW79jG8nmDU3SNh2IoUSduqQo2FucE3HxjxzVwxdxIUny3ClONkYwc-eb8mvOMKrWV1uOZ5aRx564Y8BTswaA"

# Headers con token di autorizzazione
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {security_token}"
}

# Disabilita i warning per certificati self-signed (facoltativo ma utile se il certificato HTTPS non è valido)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Invia la richiesta POST
#response = requests.post(url, data=json.dumps(data), headers=headers, verify=False)
response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

# Stampa la risposta del server
print(f"Status code: {response.status_code}")
print("Response body:", response.text)
    
