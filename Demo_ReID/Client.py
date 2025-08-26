from __future__ import print_function
import argparse
import requests
import json
import cv2

import urllib3


addr = 'http://localhost:5000/'
#addr = 'http://20.33.92.85:5000/'

#test_url = addr + 'api/image_description' #endpoint per Anomaly Detection
url = addr + 'run_reid' #endpoint per RAG

# Corpo della richiesta (payload)
data = {
    "task_id": "TASK_ID_DI_PROVA_1234567890"
}

# Token di sicurezza (sostituisci con il tuo reale token)
security_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ0cmFmZmljLW1hc3Rlci1ub2RlIiwic3ViIjoiMTMyLjE3Ny40OC4xMTMiLCJhdWQiOiJmbGFzay1hcHAtYWtzLW5vZGVwb29sMS0xNzM3OTk5Mi12bXNzMDAwMDE0LXNlcnZpY2UuNC4yMzIuMTYuMTg5Lm5pcC5pbyIsImlhdCI6MTc1NDA0NjQxNywiZXhwIjoxNzYyNjg2NDE3LCJzY29wZSI6ImNhbWVyYV9hcGkifQ.ylULJvR99_kNe9cepx81NPLGbc6y4lrT2sisaMRRtmshC1aoGJR54iheHrQKyHfG1R6HvvlK773N1S0Ozg4x15Ti3czAsteDxn9a2gqAJZw2PDQmdMfrjOoNDgzeikmrmyOaZyf9ZvoMHUM854OgugfXkKPeY6vGwdJv8xTDa6eANIUFlc2OXeeveo5VNYwrxxRTd7GTD0_xXpStQsLSIyN2PuGKrafuUqEm69MQmMWAlBIetnHG7A1GNFsT6GKvNhQyI6lUabiJuQTDrmZbVzbCggbs0hZTBLcMYNtkh5SkaEISRPartAFAqz3HOL5LM57C2oF9uB7X8q2uVzN6rA"

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
    
