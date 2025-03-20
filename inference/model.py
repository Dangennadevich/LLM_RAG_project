import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def build_rag_prompt(self, user_query, rag_context):
        if rag_context:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You - AI-asistant with access to documents. "
                        "Use the provided context to respond..\n\n"
                        f"Context: \n{rag_context}"
                    ),
                },
                {"role": "user", "content": user_query},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_query},
            ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tokenize_special_tokens=True,
        )

    @torch.no_grad()
    def model_inference(self, user_query, rag_context=None):

        prompt = self.build_rag_prompt(user_query=user_query, rag_context=rag_context)

        output = self.pipe(prompt, **self.generation_args)

        return output[0]["generated_text"]
