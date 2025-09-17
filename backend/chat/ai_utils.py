import openai
from django.conf import settings

open.api_key = settings.OPENAI_API_KEY

def ask_ai(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user" , "content": prompt}],
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error {str(e)}"