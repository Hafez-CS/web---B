# سیستم نمایش قیمت ارزها و بورس

## راه‌اندازی بک‌اند

### 1. نصب پکیج‌های مورد نیاز
```bash
pip install requests
```

### 2. مهاجرت دیتابیس (در صورت نیاز)
```bash
cd backend
python manage.py makemigrations economic
python manage.py migrate
```

### 3. اجرای سرور Django
```bash
python manage.py runserver
```

## API Endpoints

سرور Django باید روی پورت 8000 در حال اجرا باشد.

### دریافت قیمت دلار
```
GET /api/economic/dollar/
Authorization: Bearer <access_token>
```

**پاسخ:**
```json
{
    "success": true,
    "currency": "USD",
    "price": 500000,
    "unit": "ریال",
    "timestamp": "..."
}
```

### دریافت قیمت طلا
```
GET /api/economic/gold/
Authorization: Bearer <access_token>
```

**پاسخ:**
```json
{
    "success": true,
    "currency": "GOLD",
    "name": "طلا 18 عیار",
    "price": 5000000,
    "unit": "ریال",
    "timestamp": null
}
```

### دریافت شاخص بورس
```
GET /api/economic/bors/
Authorization: Bearer <access_token>
```

**پاسخ:**
```json
{
    "success": true,
    "name": "شاخص کل بورس",
    "index": 2000000,
    "timestamp": null
}
```

### دریافت قیمت بیت کوین
```
GET /api/economic/bitcoin/
Authorization: Bearer <access_token>
```

**پاسخ:**
```json
{
    "success": true,
    "currency": "BTC",
    "name": "بیت کوین",
    "price_usd": 45000,
    "price_rial": 22500000000,
    "dollar_rate": 500000,
    "unit": "دلار/ریال",
    "timestamp": null
}
```

### دریافت همه قیمت‌ها (پیشنهادی)
```
GET /api/economic/all/
Authorization: Bearer <access_token>
```

**پاسخ:**
```json
{
    "success": true,
    "data": {
        "dollar": {
            "price": 500000,
            "change": 0.5,
            "unit": "ریال"
        },
        "euro": {
            "price": 550000,
            "change": -0.2,
            "unit": "ریال"
        },
        "gold": {
            "price": 5000000,
            "change": 1.2,
            "unit": "ریال"
        },
        "bitcoin": {
            "price": 45000,
            "unit": "دلار"
        }
    }
}
```

## فرانت‌اند

### استفاده از فایل HTML

1. فایل `economic_dashboard.html` را در مرورگر باز کنید
2. ابتدا باید توکن JWT دریافت کنید (لاگین)
3. صفحه به صورت خودکار هر 5 ثانیه اطلاعات را به‌روزرسانی می‌کند

### تنظیمات JavaScript

در فایل HTML، این متغیرها را تنظیم کنید:

```javascript
const API_BASE_URL = 'http://127.0.0.1:8000/api/economic';  // آدرس API
const UPDATE_INTERVAL = 5000;  // فاصله به‌روزرسانی (5000ms = 5 ثانیه)
```

### احراز هویت

قبل از استفاده، باید توکن JWT دریافت کنید:

```javascript
// در قسمت getToken() فانکشن، اطلاعات خود را وارد کنید:
body: JSON.stringify({
    username: 'your_username',
    password: 'your_password'
})
```

یا می‌توانید توکن را از `localStorage` بگیرید اگر قبلاً لاگین کرده‌اید.

## API های استفاده شده

### APIهای ایرانی (رایگان)
- **TGJU API**: برای دلار، یورو، طلا و بورس
  - `https://api.tgju.org/v1/market/indicator/summary-table-data/`
  
- **Bonbast API**: برای قیمت ارزها (جایگزین)
  - `https://api.bonbast.com/api/v1/currencies`

### APIهای بین‌المللی
- **CoinGecko API**: برای قیمت رمزارزها (رایگان)
  - `https://api.coingecko.com/api/v3/simple/price`

## ویژگی‌های سیستم

✅ به‌روزرسانی خودکار هر 5 ثانیه بدون رفرش صفحه
✅ نمایش قیمت دلار، یورو، طلا و بیت کوین
✅ نمایش درصد تغییرات قیمت
✅ رابط کاربری زیبا و فارسی
✅ امنیت با JWT Authentication
✅ نمایش وضعیت آنلاین/آفلاین
✅ مدیریت خطاها

## نکات مهم

1. **محدودیت Rate Limit**: برخی APIها محدودیت تعداد درخواست دارند. اگر خطا دریافت کردید، فاصله `UPDATE_INTERVAL` را افزایش دهید.

2. **CORS**: اطمینان حاصل کنید که در `settings.py` آدرس فرانت شما در `CORS_ALLOWED_ORIGINS` اضافه شده است.

3. **Authentication**: همه endpointها نیاز به توکن JWT دارند. قبل از استفاده باید لاگین کنید.

4. **Timeout**: همه درخواست‌های API با timeout 5 ثانیه تنظیم شده‌اند تا سرور معطل نماند.

## عیب‌یابی

### خطای 401 Unauthorized
- توکن JWT منقضی شده است
- دوباره لاگین کنید یا توکن را refresh کنید

### خطای CORS
- `CORS_ALLOWED_ORIGINS` را در settings.py بررسی کنید
- مطمئن شوید که `corsheaders` نصب و فعال است

### قیمت‌ها نمایش داده نمی‌شوند
- اتصال اینترنت را بررسی کنید
- ممکن است API موقتاً در دسترس نباشد
- Console مرورگر را برای خطاها بررسی کنید

## بهبودهای پیشنهادی

1. **Caching**: استفاده از Redis برای کش کردن قیمت‌ها و کاهش تعداد درخواست‌ها
2. **WebSocket**: استفاده از WebSocket برای Real-time updates بهتر
3. **Database Storage**: ذخیره قیمت‌ها در دیتابیس برای تحلیل روند
4. **Charts**: اضافه کردن نمودار برای نمایش تغییرات قیمت
5. **Notifications**: اعلان به کاربر در صورت تغییرات شدید قیمت

## مثال استفاده

```javascript
// دریافت قیمت دلار
fetch('http://127.0.0.1:8000/api/economic/dollar/', {
    headers: {
        'Authorization': 'Bearer YOUR_ACCESS_TOKEN'
    }
})
.then(response => response.json())
.then(data => {
    console.log('قیمت دلار:', data.price);
});
```

## لایسنس

این پروژه برای استفاده شخصی و آموزشی است.
