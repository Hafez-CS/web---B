from openai import OpenAI
from django.conf import settings
import os

# تنظیمات Deepseek
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY', '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'),
    base_url="https://openrouter.ai/api/v1"
)

def ask_ai(prompt: str, history=None) -> str:
    try:
        # ساخت تاریخچه پیام‌ها
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # اضافه کردن تاریخچه اگر وجود داشته باشد
        if history:
            for msg in history:
                messages.append({"role": "user", "content": msg.user_message})
                if msg.ai_response and not msg.ai_response.startswith("Error"):
                    messages.append({"role": "assistant", "content": msg.ai_response})
        
        # اضافه کردن پیام جدید کاربر
        messages.append({"role": "user", "content": prompt})
        
        # ارسال درخواست به Deepseek
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # بررسی و استخراج پاسخ
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
            
        raise Exception("No valid response received from Deepseek")
            
    except Exception as e:
        print(f"Deepseek API Error: {str(e)}")  # برای دیباگ
        if "rate limit" in str(e).lower():
            return "System is currently busy. Please try again in a few moments."
        elif "invalid api key" in str(e).lower():
            return "Service configuration error. Please contact support."
        else:
            return f"Error communicating with Deepseek: {str(e)}"