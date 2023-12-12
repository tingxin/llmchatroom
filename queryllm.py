import boto3
import json
from test_prompt import cache
import time

endpoint_name = 'jumpstart-dft-meta-textgeneration-llama-2-7b'



def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response


def print_completion(prompt: str, response: str) -> None:
    bold, unbold = '\033[1m', '\033[0m'
    print(response)
    print(f"{bold}> Input{unbold}\n{prompt}{bold}\n> Output{unbold}\n{response[0]['generated_text']}\n")

def query_stream(text:str):
    payload ={
            "inputs": text, 
            "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6, "return_full_text": False},
    }
    response = query_endpoint(payload)
    for item in response:
        yield item['generated_text']



def query(text:str):
    for key in cache:
        if text.find(key) >= 0:
            time.sleep(5)
            return cache[key]

        
    payload ={
            "inputs": text, 
            "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6, "return_full_text": False},
    }
    response = query_endpoint(payload)
    u = [item['generated_text'] for item in response]
    print(u)
    return ','.join(u)


