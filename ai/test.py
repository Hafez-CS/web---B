import os
import json
import sys
from dotenv import load_dotenv
# این پکیج شامل APIStatusError و APIError است
from openai import OpenAI, APIStatusError, APIError 

# مرحله ۱: لود کردن متغیرها از فایل .env
# فرض می‌کنیم فایل .env در همان دایرکتوری این فایل قرار دارد
load_dotenv()

# خواندن متغیرهای محیطی
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") 
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat") 

print(f"--- API CONNECTION TEST START ---")
print(f"Base URL: {BASE_URL}")
print(f"Model: {MODEL_NAME}")

if not API_KEY:
    print("\nERROR: API key (DEEPSEEK_API_KEY) is not set in your .env file.")
    sys.exit(1)
    
print(f"API Key loaded (Value: {API_KEY[:4]}...{API_KEY[-4:]})")
print("---------------------------------")


# مرحله ۲: ساخت کلاینت و تلاش برای اتصال
try:
    # ساخت کلاینت OpenAI با تنظیمات OpenRouter/DeepSeek
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # مرحله ۳: اجرای یک درخواست ساده به Chat Completions
    # برای OpenRouter/DeepSeek/OpenAI این فراخوانی استاندارد است
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Just say 'Connection Test Successful'"}],
        temperature=0.0
    )

    # مرحله ۴: نمایش موفقیت
    print("\n✅ CONNECTION SUCCESSFUL! (Status 200 OK)")
    print("---------------------------------------")
    print("Full Response:")
    print(response.choices[0].message.content)
    
except APIStatusError as e:
    # مدیریت خطاهای HTTP (مثل 401, 402, 404, 429)
    print("\n❌ ERROR: API Status Error (Authentication/Balance/Rate Limit)")
    print("---------------------------------------------------------------")
    print(f"HTTP Status Code: {e.status_code}")
    
    # تلاش برای خواندن پیام ارور از JSON
    try:
        error_detail = e.response.json()
        print("API Error Details:")
        print(json.dumps(error_detail, indent=2))
    except:
        print(f"Message: {e.message}")
        
    if e.status_code in [401, 403]:
        print("\n**POSSIBLE SOLUTION:** Check if your OpenRouter/DeepSeek API Key is correct and active.")
    elif e.status_code == 402:
        print("\n**POSSIBLE SOLUTION:** Check your account balance/credit on OpenRouter/DeepSeek.")
    elif e.status_code == 404:
        print("\n**POSSIBLE SOLUTION:** Model name is probably incorrect. Check if the model name is correct for OpenRouter (e.g., 'deepseek/deepseek-chat').")

except APIError as e:
    # مدیریت خطاهای شبکه یا SSL
    print("\n❌ ERROR: Network/Connection Error")
    print("-----------------------------------")
    print(f"Message: {e}")
    print("\n**POSSIBLE")