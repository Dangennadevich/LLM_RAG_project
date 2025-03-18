import os
import time

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from db.index_mapping import indexMapping


load_dotenv()
ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "localhost")
ES_PORT = os.environ.get("ELASTICSEARCH_PORT", "9200")
ES_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "documents")
ES_PASS = os.environ.get("ELASTICSEARCH_PASSWORD", "admin")


def create_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, mappings=indexMapping)
    print(f"Index {ES_INDEX} created")


def initialize_es() -> Elasticsearch:
    """initialize the Elastic Search module for finding candidates document to answering the questions from users

    Return:
        Elastic Search Engine module
    """

    try:
        es = Elasticsearch(
            hosts=["http://elasticsearch:9200"], basic_auth=("elastic", "admin")
        )
        time.sleep(10)
        print(es.info())

    except ConnectionError as e:
        print("Connection Error:", e)

    return es


def search_relevant(es, embedding, top_k=3):
    es.indices.refresh(index=ES_INDEX)
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.embedding, doc['embedding']) + 1.0",
                "params": {"embedding": embedding.tolist()},
            },
        }
    }
    response = es.search(
        index=ES_INDEX,
        body={
            "size": top_k,
            "query": script_query,
            "_source": {"includes": ["text", "title"]},
        },
    )

    results = [
        (hit["_source"]["text"], hit["_source"]["title"])
        for hit in response["hits"]["hits"]
    ]
    return results
