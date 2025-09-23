"""
سیستم تحلیل خودکار پیام‌های کاربر
"""
import re
from .models import UserAnalysis
from login_signup.models import User

def analyze_user_message_sync(user_id, user_message, ai_response):
    """تحلیل پیام کاربر و بروزرسانی تحلیل‌ها"""
    try:
        user = User.objects.get(id=user_id)
        
        # تحلیل سبک ارتباطی
        analyze_communication_style(user, user_message)
        
        # تحلیل علایق
        analyze_preferences(user, user_message)
        
        # تحلیل نوع پاسخ مطلوب
        analyze_response_preferences(user, user_message, ai_response)
        
    except Exception as e:
        print(f"Analysis error for user {user_id}: {str(e)}")

def analyze_communication_style(user, message):
    """تحلیل سبک ارتباطی کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='communication_style',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # شناسایی سبک رسمی/غیررسمی
        formal_indicators = ['لطفاً', 'متشکرم', 'ممنون', 'جناب', 'سرکار', 'احترام']
        casual_indicators = ['سلام', 'چطوری', 'خوبی', 'باحال', 'عالی', 'ممنون']
        
        formal_count = sum(1 for word in formal_indicators if word in message)
        casual_count = sum(1 for word in casual_indicators if word in message)
        
        # بروزرسانی امتیازها
        data['formal_score'] = data.get('formal_score', 0) + formal_count
        data['casual_score'] = data.get('casual_score', 0) + casual_count
        
        total_messages = data.get('total_messages', 0) + 1
        data['total_messages'] = total_messages
        
        # تعیین سبک غالب
        if data['formal_score'] > data['casual_score']:
            data['formal'] = True
            data['casual'] = False
        else:
            data['formal'] = False
            data['casual'] = True
        
        # محاسبه confidence
        analysis.confidence_score = min(total_messages / 10.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Communication style analysis error: {str(e)}")

def analyze_preferences(user, message):
    """تحلیل علایق کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='preferences',
            defaults={'data': {'interests': []}, 'confidence_score': 0.0}
        )
        
        # کلمات کلیدی برای شناسایی علایق
        interest_keywords = {
            'technology': ['فناوری', 'تکنولوژی', 'برنامه', 'کامپیوتر', 'هوش مصنوعی', 'AI', 'پروگرام'],
            'business': ['کسب و کار', 'بیزنس', 'فروش', 'مارکتینگ', 'استارتاپ', 'درآمد'],
            'education': ['آموزش', 'درس', 'دانشگاه', 'کتاب', 'یادگیری', 'مطالعه'],
            'health': ['سلامت', 'ورزش', 'تناسب اندام', 'غذا', 'رژیم', 'پزشک'],
            'entertainment': ['فیلم', 'سینما', 'موسیقی', 'بازی', 'سرگرمی', 'تفریح']
        }
        
        data = analysis.data
        interests = data.get('interests', [])
        
        # جستجو برای کلمات کلیدی
        for category, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    if category not in interests:
                        interests.append(category)
                        break
        
        data['interests'] = interests[:5]  # حداکثر 5 علاقه
        data['total_analyzed'] = data.get('total_analyzed', 0) + 1
        
        analysis.confidence_score = min(len(interests) / 3.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Preferences analysis error: {str(e)}")

def analyze_response_preferences(user, user_message, ai_response):
    """تحلیل نوع پاسخ مطلوب کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='response_preferences',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # بررسی طول پیام کاربر (اگر طولانی باشه احتمالاً پاسخ تفصیلی می‌خواد)
        message_length = len(user_message.split())
        
        if message_length > 20:
            data['detailed_score'] = data.get('detailed_score', 0) + 1
        else:
            data['concise_score'] = data.get('concise_score', 0) + 1
        
        # بررسی کلمات کلیدی برای نوع پاسخ
        detailed_indicators = ['توضیح', 'تفصیل', 'جزئیات', 'کامل', 'دقیق']
        concise_indicators = ['خلاصه', 'کوتاه', 'سریع', 'مختصر']
        
        for word in detailed_indicators:
            if word in user_message:
                data['detailed_score'] = data.get('detailed_score', 0) + 2
                break
        
        for word in concise_indicators:
            if word in user_message:
                data['concise_score'] = data.get('concise_score', 0) + 2
                break
        
        # تعیین ترجیح
        if data.get('detailed_score', 0) > data.get('concise_score', 0):
            data['detailed'] = True
            data['concise'] = False
        else:
            data['detailed'] = False
            data['concise'] = True
        
        total_interactions = data.get('total_interactions', 0) + 1
        data['total_interactions'] = total_interactions
        
        analysis.confidence_score = min(total_interactions / 8.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Response preferences analysis error: {str(e)}")





#############################################################################################################################################################
"""
سیستم تحلیل خودکار پیام‌های کاربر
"""
import re
import json
from .models import UserAnalysis
from login_signup.models import User

def analyze_user_message_sync(user_id, user_message, ai_response):
    """تحلیل پیام کاربر و بروزرسانی تحلیل‌ها"""
    try:
        user = User.objects.get(id=user_id)
        
        # تحلیل‌های جدید پیشرفته
        analyze_sentiment(user, user_message, ai_response)
        analyze_tone(user, user_message)
        analyze_topic(user, user_message)
        analyze_preference(user, user_message, ai_response)
        analyze_behavior(user, user_message)
        analyze_engagement(user, user_message, ai_response)
        analyze_user_needs(user, user_message)
        
        # تحلیل‌های قدیمی (حفظ سازگاری)
        analyze_communication_style(user, user_message)
        analyze_preferences(user, user_message)
        analyze_response_preferences(user, user_message, ai_response)
        
    except Exception as e:
        print(f"Analysis error for user {user_id}: {str(e)}")

def analyze_sentiment(user, user_message, ai_response):
    """تحلیل احساسات کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='sentiment',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # کلمات کلیدی احساسات
        positive_words = ['عالی', 'خوب', 'ممنون', 'متشکرم', 'دوست دارم', 'راضی', 'خوشحال', 'فوق‌العاده']
        negative_words = ['بد', 'ناراحت', 'مشکل', 'متاسف', 'ناامید', 'عصبانی', 'خسته', 'بدترین']
        neutral_words = ['چطور', 'کجا', 'چی', 'چرا', 'کی', 'میشه', 'لطفا']
        
        positive_count = sum(1 for word in positive_words if word in user_message)
        negative_count = sum(1 for word in negative_words if word in user_message)
        neutral_count = sum(1 for word in neutral_words if word in user_message)
        
        # محاسبه امتیازها
        data['positive_score'] = data.get('positive_score', 0) + positive_count
        data['negative_score'] = data.get('negative_score', 0) + negative_count
        data['neutral_score'] = data.get('neutral_score', 0) + neutral_count
        
        total_messages = data.get('total_messages', 0) + 1
        data['total_messages'] = total_messages
        
        # تعیین احساس غالب
        if data['positive_score'] > data['negative_score']:
            data['dominant_sentiment'] = 'positive'
        elif data['negative_score'] > data['positive_score']:
            data['dominant_sentiment'] = 'negative'
        else:
            data['dominant_sentiment'] = 'neutral'
        
        # محاسبه میانگین
        data['sentiment_average'] = (data['positive_score'] - data['negative_score']) / max(total_messages, 1)
        
        analysis.confidence_score = min(total_messages / 8.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")

def analyze_tone(user, user_message):
    """تحلیل لحن کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='tone',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # شناسایی انواع لحن
        formal_indicators = ['لطفاً', 'متشکرم', 'ممنون', 'جناب', 'سرکار', 'احترام', 'خدمت']
        casual_indicators = ['سلام', 'چطوری', 'خوبی', 'بای', 'مرسی', 'دمت گرم']
        urgent_indicators = ['فوری', 'سریع', 'زود', 'بلافاصله', 'اکنون', 'فوراً']
        question_indicators = ['چی', 'چرا', 'چطور', 'کی', 'کجا', 'چه', '؟']
        
        formal_count = sum(1 for word in formal_indicators if word in user_message)
        casual_count = sum(1 for word in casual_indicators if word in user_message)
        urgent_count = sum(1 for word in urgent_indicators if word in user_message)
        question_count = sum(1 for word in question_indicators if word in user_message)
        
        # بروزرسانی امتیازها
        data['formal_tone'] = data.get('formal_tone', 0) + formal_count
        data['casual_tone'] = data.get('casual_tone', 0) + casual_count
        data['urgent_tone'] = data.get('urgent_tone', 0) + urgent_count
        data['questioning_tone'] = data.get('questioning_tone', 0) + question_count
        
        total_interactions = data.get('total_interactions', 0) + 1
        data['total_interactions'] = total_interactions
        
        # تعیین لحن غالب
        tone_scores = {
            'formal': data['formal_tone'],
            'casual': data['casual_tone'],
            'urgent': data['urgent_tone'],
            'questioning': data['questioning_tone']
        }
        data['dominant_tone'] = max(tone_scores, key=tone_scores.get)
        
        analysis.confidence_score = min(total_interactions / 6.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Tone analysis error: {str(e)}")

def analyze_topic(user, user_message):
    """تحلیل موضوعات مورد علاقه"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='topic',
            defaults={'data': {'topics': {}}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        topics = data.get('topics', {})
        
        # دسته‌بندی موضوعات
        topic_keywords = {
            'technology': ['فناوری', 'تکنولوژی', 'برنامه', 'کامپیوتر', 'هوش مصنوعی', 'AI', 'پروگرام', 'کد', 'سایت'],
            'business': ['کسب و کار', 'بیزنس', 'فروش', 'مارکتینگ', 'استارتاپ', 'درآمد', 'پول', 'سرمایه'],
            'education': ['آموزش', 'درس', 'دانشگاه', 'کتاب', 'یادگیری', 'مطالعه', 'امتحان', 'دانش'],
            'health': ['سلامت', 'ورزش', 'تناسب اندام', 'غذا', 'رژیم', 'پزشک', 'دارو', 'بیماری'],
            'entertainment': ['فیلم', 'سینما', 'موسیقی', 'بازی', 'سرگرمی', 'تفریح', 'تلویزیون', 'کتاب'],
            'travel': ['سفر', 'مسافرت', 'گردشگری', 'هتل', 'رستوران', 'شهر', 'کشور'],
            'science': ['علم', 'تحقیق', 'آزمایش', 'فیزیک', 'شیمی', 'ریاضی', 'زیست‌شناسی'],
            'arts': ['هنر', 'نقاشی', 'موسیقی', 'شعر', 'ادبیات', 'خلاقیت', 'طراحی']
        }
        
        # جستجو برای موضوعات
        for topic, keywords in topic_keywords.items():
            count = sum(1 for keyword in keywords if keyword in user_message)
            if count > 0:
                topics[topic] = topics.get(topic, 0) + count
        
        data['topics'] = topics
        data['total_analyzed'] = data.get('total_analyzed', 0) + 1
        
        # تعیین موضوع غالب
        if topics:
            data['primary_topic'] = max(topics, key=topics.get)
            data['topic_diversity'] = len(topics)
        
        analysis.confidence_score = min(len(topics) / 4.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Topic analysis error: {str(e)}")

def analyze_preference(user, user_message, ai_response):
    """تحلیل ترجیحات کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='preference',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # ترجیحات پاسخ
        message_length = len(user_message.split())
        if message_length > 15:
            data['detailed_questions'] = data.get('detailed_questions', 0) + 1
        else:
            data['short_questions'] = data.get('short_questions', 0) + 1
        
        # ترجیح نوع محتوا
        content_preferences = {
            'examples': ['مثال', 'نمونه', 'برای مثال', 'instance'],
            'step_by_step': ['مرحله', 'قدم', 'step', 'گام به گام'],
            'technical': ['تکنیکال', 'فنی', 'technical', 'جزئیات'],
            'simple': ['ساده', 'آسان', 'راحت', 'simple']
        }
        
        for pref_type, keywords in content_preferences.items():
            if any(keyword in user_message for keyword in keywords):
                data[f'{pref_type}_preference'] = data.get(f'{pref_type}_preference', 0) + 1
        
        # زمان‌بندی تعامل
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if 6 <= current_hour < 12:
            data['morning_activity'] = data.get('morning_activity', 0) + 1
        elif 12 <= current_hour < 18:
            data['afternoon_activity'] = data.get('afternoon_activity', 0) + 1
        else:
            data['evening_activity'] = data.get('evening_activity', 0) + 1
        
        total_interactions = data.get('total_interactions', 0) + 1
        data['total_interactions'] = total_interactions
        
        analysis.confidence_score = min(total_interactions / 10.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Preference analysis error: {str(e)}")

def analyze_behavior(user, user_message):
    """تحلیل الگوی رفتاری کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='behavior',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # الگوهای رفتاری
        message_length = len(user_message)
        word_count = len(user_message.split())
        
        # آپدیت آمار پیام
        data['total_messages'] = data.get('total_messages', 0) + 1
        data['total_characters'] = data.get('total_characters', 0) + message_length
        data['total_words'] = data.get('total_words', 0) + word_count
        
        # محاسبه میانگین‌ها
        data['avg_message_length'] = data['total_characters'] / data['total_messages']
        data['avg_words_per_message'] = data['total_words'] / data['total_messages']
        
        # تشخیص نوع سوال
        if '؟' in user_message or any(q in user_message for q in ['چی', 'چرا', 'چطور', 'کی', 'کجا']):
            data['question_count'] = data.get('question_count', 0) + 1
        
        # تشخیص درخواست کمک
        help_indicators = ['کمک', 'help', 'راهنمایی', 'نمیدونم', 'سردرگم']
        if any(indicator in user_message for indicator in help_indicators):
            data['help_requests'] = data.get('help_requests', 0) + 1
        
        # تشخیص تشکر
        thanks_indicators = ['ممنون', 'متشکرم', 'مرسی', 'thanks']
        if any(indicator in user_message for indicator in thanks_indicators):
            data['gratitude_expressions'] = data.get('gratitude_expressions', 0) + 1
        
        # محاسبه نرخ‌ها
        total_msgs = data['total_messages']
        data['question_rate'] = data.get('question_count', 0) / total_msgs
        data['help_request_rate'] = data.get('help_requests', 0) / total_msgs
        data['politeness_rate'] = data.get('gratitude_expressions', 0) / total_msgs
        
        analysis.confidence_score = min(total_msgs / 12.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Behavior analysis error: {str(e)}")

def analyze_engagement(user, user_message, ai_response):
    """تحلیل میزان تعامل کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='engagement',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # محاسبه شاخص‌های تعامل
        message_complexity = len(user_message.split()) / 10  # نرمال‌سازی
        
        # تعامل با پاسخ قبلی
        if ai_response:
            # اگر کاربر سوال بعدی پرسیده باشه (نشان تعامل بالا)
            follow_up_indicators = ['همچنین', 'also', 'بعد', 'دیگه', 'more']
            if any(indicator in user_message for indicator in follow_up_indicators):
                data['follow_up_questions'] = data.get('follow_up_questions', 0) + 1
        
        # حجم تعامل
        interaction_score = min(len(user_message) / 100, 1.0)  # نرمال‌سازی به 0-1
        
        data['total_interactions'] = data.get('total_interactions', 0) + 1
        data['total_engagement_score'] = data.get('total_engagement_score', 0) + interaction_score
        data['avg_engagement'] = data['total_engagement_score'] / data['total_interactions']
        
        # تعیین سطح تعامل
        if data['avg_engagement'] > 0.7:
            data['engagement_level'] = 'high'
        elif data['avg_engagement'] > 0.4:
            data['engagement_level'] = 'medium'
        else:
            data['engagement_level'] = 'low'
        
        analysis.confidence_score = min(data['total_interactions'] / 8.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Engagement analysis error: {str(e)}")

def analyze_user_needs(user, user_message):
    """تحلیل نیازهای کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='user_needs',
            defaults={'data': {'needs': {}}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        needs = data.get('needs', {})
        
        # دسته‌بندی نیازها
        need_patterns = {
            'information_seeking': ['چیه', 'چی هست', 'توضیح', 'معنی', 'تعریف', 'اطلاعات'],
            'problem_solving': ['مشکل', 'حل', 'راه‌حل', 'چطور', 'نحوه', 'solve'],
            'learning': ['یاد بگیرم', 'آموزش', 'learn', 'فرا بگیرم', 'بیاموزم'],
            'guidance': ['راهنمایی', 'کمک', 'guide', 'هدایت', 'مشورت'],
            'validation': ['درسته', 'صحیحه', 'مطمئن', 'confirm', 'تایید'],
            'creative_help': ['ایده', 'پیشنهاد', 'خلاقیت', 'creative', 'نوآوری'],
            'decision_making': ['انتخاب', 'تصمیم', 'کدوم', 'بهتره', 'choose']
        }
        
        # شناسایی نیازها
        for need_type, patterns in need_patterns.items():
            if any(pattern in user_message for pattern in patterns):
                needs[need_type] = needs.get(need_type, 0) + 1
        
        data['needs'] = needs
        data['total_analyzed'] = data.get('total_analyzed', 0) + 1
        
        # تعیین نیاز اصلی
        if needs:
            data['primary_need'] = max(needs, key=needs.get)
            data['need_diversity'] = len(needs)
        
        analysis.confidence_score = min(len(needs) / 3.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"User needs analysis error: {str(e)}")

def analyze_communication_style(user, message):
    """تحلیل سبک ارتباطی کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='communication_style',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # شناسایی سبک رسمی/غیررسمی
        formal_indicators = ['لطفاً', 'متشکرم', 'ممنون', 'جناب', 'سرکار', 'احترام']
        casual_indicators = ['سلام', 'چطوری', 'خوبی', 'باحال', 'عالی', 'ممنون']
        
        formal_count = sum(1 for word in formal_indicators if word in message)
        casual_count = sum(1 for word in casual_indicators if word in message)
        
        # بروزرسانی امتیازها
        data['formal_score'] = data.get('formal_score', 0) + formal_count
        data['casual_score'] = data.get('casual_score', 0) + casual_count
        
        total_messages = data.get('total_messages', 0) + 1
        data['total_messages'] = total_messages
        
        # تعیین سبک غالب
        if data['formal_score'] > data['casual_score']:
            data['formal'] = True
            data['casual'] = False
        else:
            data['formal'] = False
            data['casual'] = True
        
        # محاسبه confidence
        analysis.confidence_score = min(total_messages / 10.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Communication style analysis error: {str(e)}")

def analyze_preferences(user, message):
    """تحلیل علایق کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='preferences',
            defaults={'data': {'interests': []}, 'confidence_score': 0.0}
        )
        
        # کلمات کلیدی برای شناسایی علایق
        interest_keywords = {
            'technology': ['فناوری', 'تکنولوژی', 'برنامه', 'کامپیوتر', 'هوش مصنوعی', 'AI', 'پروگرام'],
            'business': ['کسب و کار', 'بیزنس', 'فروش', 'مارکتینگ', 'استارتاپ', 'درآمد'],
            'education': ['آموزش', 'درس', 'دانشگاه', 'کتاب', 'یادگیری', 'مطالعه'],
            'health': ['سلامت', 'ورزش', 'تناسب اندام', 'غذا', 'رژیم', 'پزشک'],
            'entertainment': ['فیلم', 'سینما', 'موسیقی', 'بازی', 'سرگرمی', 'تفریح']
        }
        
        data = analysis.data
        interests = data.get('interests', [])
        
        # جستجو برای کلمات کلیدی
        for category, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    if category not in interests:
                        interests.append(category)
                        break
        
        data['interests'] = interests[:5]  # حداکثر 5 علاقه
        data['total_analyzed'] = data.get('total_analyzed', 0) + 1
        
        analysis.confidence_score = min(len(interests) / 3.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Preferences analysis error: {str(e)}")

def analyze_response_preferences(user, user_message, ai_response):
    """تحلیل نوع پاسخ مطلوب کاربر"""
    try:
        analysis, created = UserAnalysis.objects.get_or_create(
            user=user,
            analysis_type='response_preferences',
            defaults={'data': {}, 'confidence_score': 0.0}
        )
        
        data = analysis.data
        
        # بررسی طول پیام کاربر (اگر طولانی باشه احتمالاً پاسخ تفصیلی می‌خواد)
        message_length = len(user_message.split())
        
        if message_length > 20:
            data['detailed_score'] = data.get('detailed_score', 0) + 1
        else:
            data['concise_score'] = data.get('concise_score', 0) + 1
        
        # بررسی کلمات کلیدی برای نوع پاسخ
        detailed_indicators = ['توضیح', 'تفصیل', 'جزئیات', 'کامل', 'دقیق']
        concise_indicators = ['خلاصه', 'کوتاه', 'سریع', 'مختصر']
        
        for word in detailed_indicators:
            if word in user_message:
                data['detailed_score'] = data.get('detailed_score', 0) + 2
                break
        
        for word in concise_indicators:
            if word in user_message:
                data['concise_score'] = data.get('concise_score', 0) + 2
                break
        
        # تعیین ترجیح
        if data.get('detailed_score', 0) > data.get('concise_score', 0):
            data['detailed'] = True
            data['concise'] = False
        else:
            data['detailed'] = False
            data['concise'] = True
        
        total_interactions = data.get('total_interactions', 0) + 1
        data['total_interactions'] = total_interactions
        
        analysis.confidence_score = min(total_interactions / 8.0, 1.0)
        analysis.data = data
        analysis.save()
        
    except Exception as e:
        print(f"Response preferences analysis error: {str(e)}")

