import json
from typing import Dict, List

from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import boto3

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/x-text"
    accepts = "application/json"

    def transform_input(self, inputs: list, model_kwargs: Dict) -> bytes:
        """
        Transforms the input into bytes that can be consumed by SageMaker endpoint.
        Args:
            inputs: List of input strings.
            model_kwargs: Additional keyword arguments to be passed to the endpoint.
        Returns:
            The transformed bytes input.
        """
        # Example: inference.py expects a JSON string with a "inputs" key:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        print(input_str)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        """
        Transforms the bytes output from the endpoint into a list of embeddings.
        Args:
            output: The bytes output from SageMaker endpoint.
        Returns:
            The transformed output - list of embeddings
        Note:
            The length of the outer list is the number of input strings.
            The length of the inner lists is the embedding dimension.
        """
        # Example: inference.py returns a JSON string with the list of
        # embeddings in a "vectors" key:
        response_json = json.loads(output.read().decode("utf-8"))
        print(response_json)
        return response_json["embedding"]


content_handler = ContentHandler()


embeddings = SagemakerEndpointEmbeddings(
    credentials_profile_name="default",
    region_name="ap-northeast-1",
    endpoint_name="jumpstart-dft-sentence-encoder-cmlm-en-large-1",
    content_handler=content_handler,
)


def get():
    return embeddings


def query_endpoint(encoded_text):
    endpoint_name = 'jumpstart-dft-sentence-encoder-cmlm-en-large-1'
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-text', Body=encoded_text, Accept='application/json;verbose')
    return response

def parse_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    embedding, model_output = model_predictions['embedding'], model_predictions['model_output']
    return embedding, model_output