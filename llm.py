import json
from typing import Dict

import boto3
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

roleARN = "arn:aws:iam::515491257789:role/service-role/AmazonSageMaker-ExecutionRole-20231013T093702"
roleARN ="arn:aws:iam::515491257789:role/AdminRole"
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b"
region_name="ap-northeast-1"

client = boto3.client(
    "sagemaker-runtime"
)

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


content_handler = ContentHandler()


def get():
    llm=SagemakerEndpoint(
        endpoint_name=endpoint_name,
        client=client,
        model_kwargs={"temperature": 1e-10},
        content_handler=content_handler,
    )
    return llm

l = get()
print(l("hello"))
