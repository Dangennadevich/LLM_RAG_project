import gradio as gr
import requests

def get_answer(question, use_context):
    try:
        # Str >> boolean
        use_context = str(use_context).lower() == "true"

        response = requests.post(
            "http://176.99.131.161:8080/question",
            json={
                "question": question,
                "use_context": use_context
            },
            timeout=60
        )
        
        if response.ok:
            answer = response.json().get("answer", "Ответ не найден")
            context_status = "Контекст использован ✅" if use_context else "Контекст не использован ❌"
            return answer, context_status
        else:
            return "Ошибка сервера", f"Код: {response.status_code}"
            
    except Exception as e:
        return f"Ошибка: {str(e)}", "Не удалось определить статус"

# Интерфейс через Blocks
with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("## Вопрос-ответ через ваш API")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Ваш вопрос",
                placeholder="Как работает модель Deepseek-R1?",
                lines=3
            )
            context_toggle = gr.Dropdown(
                label="Использовать RAG?",
                choices=["true", "false"],
                value="true"
            )
            submit_btn = gr.Button("Отправить запрос")
            
        with gr.Column():
            answer_output = gr.Textbox(
                label="Ответ системы",
                interactive=False,
                lines=5
            )
            context_status = gr.Textbox(
                label="Статус контекста",
                interactive=False
            )

    # Обработчик
    submit_btn.click(
        fn=get_answer,
        inputs=[question_input, context_toggle],
        outputs=[answer_output, context_status]
    )

demo.launch(server_port=7860, server_name="0.0.0.0")