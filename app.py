import dotenv
from flask import Flask, jsonify, request

from db.run_db import create_index, initialize_es
from inference.model import Model


es = initialize_es()
create_index(es)

dotenv.load_dotenv()
app = Flask(__name__)

MODEL = Model()


def get_answer(question: str, context: str) -> str:
    """
    Функция-заглушка для генерации ответа на вопрос с учетом контекста.
    Здесь можно подключить модель для генерации ответа.
    """
    response = MODEL.model_inference(question)
    return response


@app.route("/question", methods=["POST"])
def answer_question():
    """
    Эндпоинт для получения ответа на вопрос.
    Ожидается JSON с полем:
      - question: строка с вопросом
    Функция вычисляет эмбеддинг вопроса, ищет наиболее похожий документ в Elasticsearch,
    извлекает его текст как контекст и генерирует ответ с помощью функции get_answer.
    """
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Требуется поле 'question'"}), 400

    question = data["question"]
    if not isinstance(question, str):
        return jsonify({"error": "'question' должен быть строкой"}), 400

    # Вычисляем эмбеддинг для вопроса (используется размерность по умолчанию 128)
    # question_embedding = get_embedding(question)

    # query = {
    #     "size": 1,
    #     "query": {
    #         "script_score": {
    #             "query": {"match_all": {}},
    #             "script": {
    #                 "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
    #                 "params": {"query_vector": question_embedding}
    #             }
    #         }
    #     }
    # }
    # search_res = es.search(index=ES_INDEX, body=query)
    # hits = search_res.get('hits', {}).get('hits', [])
    # if hits:
    #     context = hits[0]['_source'].get('text', "Контекст не найден")
    # else:
    #     context = None

    answer = get_answer(question, context=None)
    return jsonify({"question": question, "context": None, "answer": answer}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
