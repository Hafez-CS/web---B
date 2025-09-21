from openai import OpenAI
from django.conf import settings

client = OpenAI(
    api_key=settings.DEEPSEEK_API_KEY,  # کلید Deepseek را در settings.py قرار دهید
    base_url="https://api.deepseek.com/v1"
)

def ask_ai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error {str(e)}"