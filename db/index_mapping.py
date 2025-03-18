import os

from dotenv import load_dotenv


load_dotenv()
DIMS = os.getenv("EMB_DIMS")

indexMapping = {
    "properties": {
        "text": {"type": "text"},
        "title": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": DIMS,
            "index": True,
            "similarity": "cosine",
        },
    }
}
