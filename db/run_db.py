import os
import time

import torch
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


load_dotenv()
ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "localhost")
ES_PORT = os.environ.get("ELASTICSEARCH_PORT", "9200")
ES_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "documents")
ES_PASS = os.environ.get("ELASTICSEARCH_PASSWORD", "documents")

# es = Elasticsearch(
#     hosts=[{"host": ES_HOST, "port": int(ES_PORT), "scheme": "https"}],
#     http_auth=("elastic", "admin"),
# )


def create_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX)
    print(f"Index {ES_INDEX} created")


def search_relevant(es, query, top_k=3):
    query_embedding = torch  # TODO get real embedding
    # Используем script_score для вычисления cosine similarity
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.embedding, 'embedding') + 1.0",
                "params": {"embedding": query_embedding.tolist()},
            },
        }
    }
    response = es.search(
        index=ES_INDEX,
        body={"size": top_k, "query": script_query, "_source": {"includes": ["text"]}},
    )

    results = [hit["_source"]["text"] for hit in response["hits"]["hits"]]
    return results


def get_db_cert(os_type: str) -> str:
    return (
        os.environ.get("MAC_CA_CERTS")
        if os_type == "Darwin"
        else os.environ.get("LINUX_CA_CERTS")
    )


def initialize_es() -> Elasticsearch:
    """initialize the Elastic Search module for finding candidates document to answering the questions from users

    Return:
        Elastic Search Engine module
    """

    try:
        es = Elasticsearch(
            hosts=["http://elasticsearch:9200"], basic_auth=("elastic", "admin")
        )
        time.sleep(30)

    except ConnectionError as e:
        print("Connection Error:", e)

    return es
