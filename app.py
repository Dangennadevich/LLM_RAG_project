from typing import Optional

import dotenv
from flask import Flask, jsonify, request

from db.run_db import initialize_es, search_relevant
from inference.encoder import Encoder
from inference.model import Model


es = initialize_es()

dotenv.load_dotenv()
app = Flask(__name__)

MODEL = Model()
Encoder = Encoder()


def get_answer(question: str, context: Optional[str]) -> str:

    response = MODEL.model_inference(question, rag_context=context)
    return response


@app.route("/question", methods=["POST"])
def answer_question():

    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Требуется поле 'question'"}), 400

    question = data["question"]
    if not isinstance(question, str):
        return jsonify({"error": "'question' должен быть строкой"}), 400

    embedding = Encoder.encode(question)

    results = search_relevant(es, embedding)
    context = ""
    papers = []
    if results:
        for item in results:
            context += item[0] + "\n"
            papers.append(item[1])
        answer = get_answer(question, context=context)
    else:
        answer = get_answer(question, context=None)
    return (
        jsonify(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "used_papers": list(set(papers)),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
