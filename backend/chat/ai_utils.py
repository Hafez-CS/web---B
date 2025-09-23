from openai import OpenAI
from django.conf import settings
import os
from .models import UserAnalysis
# تنظیمات Deepseek - بروزرسانی API
client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY', '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'),
    base_url="https://openrouter.ai/api/v1"
)

def get_user_system_prompt(user):
    """ساخت system prompt هوشمند بر اساس تحلیل‌های AI"""
    if not user:
        return "You are a helpful assistant. Respond naturally and adapt to the user's communication style."
    
    try:
        from .analysis import get_user_analysis_summary
        
        base_prompt = "You are a highly intelligent and adaptive assistant."
        user_adaptations = []
        
        # دریافت خلاصه تحلیل‌های AI
        analysis_summary = get_user_analysis_summary(user)
        
        if not analysis_summary:
            return f"{base_prompt} Respond naturally and adapt to the user's communication style as you learn from their messages."
        
        # تحلیل احساسات
        if 'sentiment' in analysis_summary:
            sentiment_data = analysis_summary['sentiment']['data']
            if sentiment_data.get('type') == 'positive':
                user_adaptations.append("User tends to be positive and optimistic")
            elif sentiment_data.get('type') == 'negative':
                user_adaptations.append("User may need extra support and encouragement")
        
        # تحلیل لحن
        if 'tone' in analysis_summary:
            tone_data = analysis_summary['tone']['data']
            formality = tone_data.get('formality', 'mixed')
            if formality == 'formal':
                user_adaptations.append("Use formal, professional language")
            elif formality == 'casual':
                user_adaptations.append("Use casual, friendly language")
            
            urgency = tone_data.get('urgency', 5)
            if urgency > 7:
                user_adaptations.append("User values quick, direct responses")
        
        # تحلیل ترجیحات
        if 'preference' in analysis_summary:
            pref_data = analysis_summary['preference']['data']
            response_length = pref_data.get('response_length', 'medium')
            detail_level = pref_data.get('detail_level', 'intermediate')
            
            if response_length == 'short':
                user_adaptations.append("Keep responses concise and to the point")
            elif response_length == 'long':
                user_adaptations.append("Provide detailed, comprehensive explanations")
            
            if detail_level == 'basic':
                user_adaptations.append("Use simple language and basic concepts")
            elif detail_level == 'advanced':
                user_adaptations.append("Use technical terms and advanced concepts")
            
            if pref_data.get('examples_needed'):
                user_adaptations.append("Include practical examples in explanations")
            
            if pref_data.get('step_by_step'):
                user_adaptations.append("Break down complex topics into step-by-step instructions")
        
        # تحلیل رفتار
        if 'behavior' in analysis_summary:
            behavior_data = analysis_summary['behavior']['data']
            pattern = behavior_data.get('interaction_pattern')
            
            if pattern == 'question_asker':
                user_adaptations.append("Anticipate follow-up questions and provide comprehensive answers")
            elif pattern == 'help_seeker':
                user_adaptations.append("Focus on providing practical solutions and guidance")
            elif pattern == 'explorer':
                user_adaptations.append("Encourage exploration and provide additional related information")
        
        # تحلیل نیازها
        if 'user_needs' in analysis_summary:
            needs_data = analysis_summary['user_needs']['data']
            primary_need = needs_data.get('primary_need')
            
            if primary_need == 'learning':
                user_adaptations.append("Structure responses to facilitate learning and understanding")
            elif primary_need == 'problem_solving':
                user_adaptations.append("Focus on practical solutions and troubleshooting steps")
            elif primary_need == 'creativity':
                user_adaptations.append("Encourage creative thinking and provide innovative ideas")
        
        # ساخت prompt نهایی
        if user_adaptations:
            adaptations_str = ". ".join(user_adaptations)
            return f"{base_prompt} Based on user analysis: {adaptations_str}. Adapt your communication style accordingly, but respond naturally without explicitly mentioning these observations."
        else:
            return f"{base_prompt} Respond naturally and adapt to the user's communication style as you learn from their messages."
        
    except Exception as e:
        print(f"Error getting user context: {str(e)}")
        return "You are a helpful assistant."

def ask_ai(prompt: str, history=None, user=None, room_id=None) -> str:
    """ارسال درخواست به AI و دریافت پاسخ"""
    try:
        # ساخت system prompt بر اساس تحلیل کاربر
        system_prompt = get_user_system_prompt(user) if user else "You are a helpful assistant."
        messages = [{"role": "system", "content": system_prompt}]
        
        # اضافه کردن تاریخچه اگر وجود داشته باشد
        if history:
            # محدود کردن به آخر 15 پیام (برای جلوگیری از token limit)
            recent_history = list(history)[-15:]
            
            for msg in recent_history:
                # فقط پیام‌هایی که محتوای صحیح دارند
                if msg.user_message and msg.user_message.strip():
                    messages.append({"role": "user", "content": msg.user_message})
                    
                    # بررسی صحت پاسخ AI
                    if (msg.ai_response and 
                        msg.ai_response.strip() and 
                        not msg.ai_response.startswith("Error") and
                        not msg.ai_response.startswith("System is currently busy") and
                        not msg.ai_response.startswith("Service configuration error")):
                        messages.append({"role": "assistant", "content": msg.ai_response})
        
        # اضافه کردن پیام جدید کاربر
        messages.append({"role": "user", "content": prompt})
        
        # Debug: نمایش تعداد پیام‌ها
        print(f"AI Request: {len(messages)} messages, Latest: {prompt[:50]}...")
        
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
        
        # اگر مشکل API key هست، پاسخ fallback بده
        if "401" in str(e) or "User not found" in str(e) or "invalid api key" in str(e).lower():
            return "🤖 سیستم AI موقتاً در دسترس نیست. لطفاً API key را بروزرسانی کنید."
        elif "rate limit" in str(e).lower():
            return "⏰ سیستم شلوغه. لطفاً کمی صبر کنید."
        else:
            return f"❌ خطا در ارتباط با AI: {str(e)}"
        
def get_ai_response(prompt: str, user=None, room_id=None) -> str:
    """دریافت پاسخ AI و اجرای تحلیل هوشمند"""
    try:
        # دریافت تاریخچه پیام‌های کاربر برای context (فقط از همین چت روم)
        history = None
        if user and room_id:
            from .models import Message
            history = Message.objects.filter(user=user, room_id=room_id).order_by('-timestamp')[:10]
        
        # دریافت پاسخ از AI
        ai_response = ask_ai(prompt, history, user, room_id)
        
        # اجرای تحلیل AI بر روی پیام و پاسخ
        if user and ai_response and not ai_response.startswith("🤖") and not ai_response.startswith("⏰") and not ai_response.startswith("❌"):
            try:
                # تحلیل AI در پس‌زمینه
                from .analysis import analyze_user_message_sync
                analyze_user_message_sync(user.id, prompt, ai_response)
                print(f"✅ AI Analysis completed for user {user.id}")
            except Exception as analysis_error:
                print(f"⚠️ AI Analysis failed: {str(analysis_error)}")
                # عدم توقف سیستم در صورت مشکل تحلیل
        
        return ai_response
        
    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        return "❌ خطا در دریافت پاسخ از AI. لطفاً دوباره تلاش کنید."
    try:
        # ساخت system prompt بر اساس تحلیل کاربر
        system_prompt = get_user_system_prompt(user) if user else "You are a helpful assistant."
        messages = [{"role": "system", "content": system_prompt}]
        
        # اضافه کردن تاریخچه اگر وجود داشته باشد
        if history:
            # محدود کردن به آخر 15 پیام (برای جلوگیری از token limit)
            recent_history = list(history)[-15:]
            
            for msg in recent_history:
                # فقط پیام‌هایی که محتوای صحیح دارند
                if msg.user_message and msg.user_message.strip():
                    messages.append({"role": "user", "content": msg.user_message})
                    
                    # بررسی صحت پاسخ AI
                    if (msg.ai_response and 
                        msg.ai_response.strip() and 
                        not msg.ai_response.startswith("Error") and
                        not msg.ai_response.startswith("System is currently busy") and
                        not msg.ai_response.startswith("Service configuration error")):
                        messages.append({"role": "assistant", "content": msg.ai_response})
        
        # اضافه کردن پیام جدید کاربر
        messages.append({"role": "user", "content": prompt})
        
        # Debug: نمایش تعداد پیام‌ها
        print(f"AI Request: {len(messages)} messages, Latest: {prompt[:50]}...")
        
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
        
        # اگر مشکل API key هست، پاسخ fallback بده
        if "401" in str(e) or "User not found" in str(e) or "invalid api key" in str(e).lower():
            return "🤖 سیستم AI موقتاً در دسترس نیست. لطفاً API key را بروزرسانی کنید."
        elif "rate limit" in str(e).lower():
            return "⏰ سیستم شلوغه. لطفاً کمی صبر کنید."
        else:
            return f"❌ خطا در ارتباط با AI: {str(e)}"