"""
سیستم تحلیل AI-پایه پیام‌های کاربر
"""
import json
from .models import UserAnalysis
from login_signup.models import User

def analyze_user_message_sync(user_id, user_message, ai_response):
    """تحلیل پیام کاربر با AI و بروزرسانی تحلیل‌ها"""
    try:
        user = User.objects.get(id=user_id)
        
        # درخواست تحلیل کامل از AI
        ai_analysis = get_ai_comprehensive_analysis(user_message, ai_response)
        
        if ai_analysis:
            # اعمال نتایج تحلیل AI
            apply_ai_analysis_results(user, ai_analysis)
        
    except Exception as e:
        print(f"AI Analysis error for user {user_id}: {str(e)}")

def get_ai_comprehensive_analysis(user_message, ai_response):
    """دریافت تحلیل جامع از AI"""
    try:
        from openai import OpenAI
        import os
        
        client = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        
        if not os.getenv('OPENROUTER_API_KEY'):
            raise ValueError('❌ OPENROUTER_API_KEY not found in environment variables')
        
        analysis_prompt = f"""
تو یک تحلیلگر حرفه‌ای رفتار کاربر هستی. از روی این مکالمه، تحلیل کاملی ارائه بده:

پیام کاربر: "{user_message}"
پاسخ AI: "{ai_response}"

لطفاً این 7 مورد رو تحلیل کن و دقیقاً به فرمت JSON زیر برگردون:

{{
    "sentiment": {{
        "type": "positive/negative/neutral",
        "intensity": 1-10,
        "emotions": ["happy", "frustrated", "curious", "etc"],
        "confidence": 0.0-1.0
    }},
    "tone": {{
        "formality": "formal/casual/mixed",
        "urgency": 1-10,
        "politeness": 1-10,
        "directness": 1-10,
        "confidence": 0.0-1.0
    }},
    "topic": {{
        "primary_category": "technology/business/education/health/entertainment/science/arts/other",
        "subcategories": ["specific topics"],
        "complexity_level": 1-10,
        "confidence": 0.0-1.0
    }},
    "preference": {{
        "response_length": "short/medium/long",
        "detail_level": "basic/intermediate/advanced",
        "examples_needed": true/false,
        "visual_aids": true/false,
        "step_by_step": true/false,
        "confidence": 0.0-1.0
    }},
    "behavior": {{
        "interaction_pattern": "question_asker/help_seeker/explorer/validator",
        "learning_style": "visual/auditory/reading/kinesthetic",
        "decision_making": "quick/deliberate/collaborative",
        "risk_tolerance": 1-10,
        "confidence": 0.0-1.0
    }},
    "engagement": {{
        "level": "low/medium/high",
        "attention_span": "short/medium/long",
        "follow_up_likelihood": 0.0-1.0,
        "topic_switching": "frequent/moderate/rare",
        "confidence": 0.0-1.0
    }},
    "user_needs": {{
        "primary_need": "information/problem_solving/learning/guidance/validation/creativity/decision_making",
        "urgency_level": 1-10,
        "support_type": "technical/emotional/strategic/creative",
        "goal_clarity": 1-10,
        "confidence": 0.0-1.0
    }}
}}

فقط JSON خالص برگردون، هیچ توضیح اضافی نده.
"""

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        ai_analysis = response.choices[0].message.content.strip()
        
        # تمیز کردن و استخراج JSON
        if ai_analysis.startswith('```json'):
            ai_analysis = ai_analysis[7:-3]
        elif ai_analysis.startswith('```'):
            ai_analysis = ai_analysis[3:-3]
        
        return ai_analysis
        
    except Exception as e:
        print(f"AI Analysis request error: {str(e)}")
        return None

def apply_ai_analysis_results(user, ai_analysis_json):
    """اعمال نتایج تحلیل AI به دیتابیس"""
    try:
        analysis_data = json.loads(ai_analysis_json)
        
        # بروزرسانی هر نوع تحلیل
        for analysis_type, data in analysis_data.items():
            if isinstance(data, dict) and 'confidence' in data:
                update_user_analysis(user, analysis_type, data)
        
    except json.JSONDecodeError:
        print("Invalid JSON from AI analysis")
    except Exception as e:
        print(f"Error applying AI analysis: {str(e)}")

def update_user_analysis(user, analysis_type, new_data):
    """بروزرسانی یک نوع تحلیل خاص"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type=analysis_type,
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        current_data = analysis.data
        confidence = new_data.get('confidence', 0.0)
        
        # اگر اولین بار است یا confidence بالاتر است
        if created or confidence > analysis.confidence_score:
            # کپی کردن داده‌های جدید (بدون confidence)
            filtered_data = {k: v for k, v in new_data.items() if k != 'confidence'}
            
            # ترکیب با داده‌های موجود
            if not created:
                # حفظ تاریخچه
                current_data['history'] = current_data.get('history', [])
                current_data['history'].append({
                    'timestamp': str(analysis.last_updated),
                    'data': current_data.copy()
                })
                
                # حداکثر 5 تاریخچه نگه دار
                current_data['history'] = current_data['history'][-5:]
            
            # بروزرسانی با داده‌های جدید
            current_data.update(filtered_data)
            current_data['last_analysis_confidence'] = confidence
            current_data['total_analyses'] = current_data.get('total_analyses', 0) + 1
            
            analysis.data = current_data
            analysis.confidence_score = confidence
            analysis.save()
            
            print(f"Updated {analysis_type} analysis for user {user.id} (confidence: {confidence})")
        
    except Exception as e:
        print(f"Error updating {analysis_type} analysis: {str(e)}")

def get_user_analysis_summary(user):
    """خلاصه‌ای از تمام تحلیل‌های کاربر برای AI"""
    try:
        analyses = UserAnalysis.objects.filter(user=user, confidence_score__gt=0.3)
        summary = {}
        
        for analysis in analyses:
            analysis_type = analysis.analysis_type
            data = analysis.data
            confidence = analysis.confidence_score
            
            summary[analysis_type] = {
                'data': data,
                'confidence': confidence,
                'last_updated': str(analysis.last_updated)
            }
        
        return summary
        
    except Exception as e:
        print(f"Error getting analysis summary: {str(e)}")
        return {}