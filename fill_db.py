import os
import pickle
import time

import dotenv
from elasticsearch.helpers import bulk
from minio import Minio

from db.run_db import initialize_es


dotenv.load_dotenv()

BUCKET_NAME = "rag-project"

es = initialize_es()

time.sleep(10)

YANDEX_CLOUD_ACCESS_KEY = os.environ.get("YANDEX_CLOUD_ACCESS_KEY")
YANDEX_CLOUD_SECRET_KEY = os.environ.get("YANDEX_CLOUD_SECRET_KEY")

client = Minio(
    "storage.yandexcloud.net",
    access_key=YANDEX_CLOUD_ACCESS_KEY,
    secret_key=YANDEX_CLOUD_SECRET_KEY,
    secure=True,
)


def download_from_yandex():

    object_name = "dict_all_embed.pkl"
    client.fget_object(
        bucket_name=BUCKET_NAME, object_name=object_name, file_path=f"{object_name}"
    )

    with open(f"{object_name}", "rb") as file:
        dict_all_embed = pickle.load(file)

    return dict_all_embed


if __name__ == "__main__":
    dict_all_embed = download_from_yandex()
    final_list = []
    for _, v in dict_all_embed.items():
        texts = v[0]
        embeddings = v[1]
        title = v[2]
        for i, embed in enumerate(embeddings):
            final_list.append([texts[i], embed, title])

    docs = []
    for i, data in enumerate(final_list):
        text = data[0]
        embedding = data[1].tolist()
        title = data[2]
        doc = {
            "_index": "documents",
            "_id": i,
            "_source": {"text": text, "title": title, "embedding": embedding},
        }
        docs.append(doc)

    bulk(es, docs)
