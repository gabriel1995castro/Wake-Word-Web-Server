import ollama

class OllamaAgent:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self.system_prompt = (
            "Você é um agente de teste para interpretar comandos de voz. "
            "Responda de forma curta e objetiva."
        )

    def run(self, user_text: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        return response["message"]["content"].strip()
