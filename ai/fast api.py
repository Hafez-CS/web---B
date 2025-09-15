import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
import arabic_reshaper
from bidi.algorithm import get_display

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logging.warning("کتابخانه xgboost نصب نشده است. XGBoost در دسترس نخواهد بود.")

# تنظیم لاگ‌گیری با جزئیات بیشتر
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
    level=logging.DEBUG
)

# تنظیم فونت فارسی برای Matplotlib
font_path = "path/to/Vazir.ttf"  # مسیر فونت Vazir را با مسیر واقعی جایگزین کنید
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Vazir"
else:
    logging.warning("فونت Vazir یافت نشد. از فونت پیش‌فرض استفاده می‌شود.")
    plt.rcParams["font.family"] = "sans-serif"

app = FastAPI(title="Advanced Financial Predictor API")

class DataProcessor:
    def __init__(self):
        self.df = None

    async def load_file(self, file: UploadFile):
        logging.debug(f"بارگذاری فایل: {file.filename}")
        try:
            # بررسی پسوند فایل و استفاده از تابع مناسب برای خواندن
            if file.filename.endswith('.csv'):
                self.df = pd.read_csv(file.file)
                logging.debug("فایل CSV با موفقیت خوانده شد.")
            elif file.filename.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file.file)
                logging.debug("فایل Excel با موفقیت خوانده شد.")
            else:
                logging.error("فرمت فایل پشتیبانی نمی‌شود.")
                raise ValueError("فرمت فایل پشتیبانی نمی‌شود. فقط فایل‌های CSV و Excel مجاز هستند.")
            
            # بررسی اولیه داده‌ها
            if self.df.empty:
                logging.error("فایل بارگذاری‌شده خالی است.")
                raise ValueError("فایل بارگذاری‌شده خالی است.")
            
            # اجرای خودکار پاک‌سازی و داده‌کاوی
            logging.debug("شروع پاک‌سازی داده‌ها")
            clean_report = self.clean_data()
            logging.debug("شروع داده‌کاوی")
            mine_report = self.mine_data()
            
            logging.info(f"فایل {file.filename} با موفقیت بارگذاری و پردازش شد.")
            return {
                "message": "فایل با موفقیت بارگذاری شد!",
                "columns": self.df.columns.tolist(),
                "cleaning_report": clean_report,
                "mining_report": mine_report
            }
        except Exception as e:
            logging.error(f"خطا در بارگذاری فایل {file.filename}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در بارگذاری فایل: {str(e)}")

    def clean_data(self):
        if self.df is None:
            logging.error("هیچ داده‌ای برای پاک‌سازی وجود ندارد.")
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            df_cleaned = self.df.copy()
            logging.debug(f"شروع پاک‌سازی داده‌ها با {len(df_cleaned)} ردیف و {len(df_cleaned.columns)} ستون")
            
            # حذف ستون‌های کاملاً NaN
            df_cleaned = df_cleaned.dropna(axis=1, how='all')
            logging.debug(f"پس از حذف ستون‌های کاملاً NaN: {len(df_cleaned.columns)} ستون باقی ماند")
            
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
            logging.debug(f"ستون‌های عددی: {numeric_cols.tolist()}")
            logging.debug(f"ستون‌های غیرعددی: {non_numeric_cols.tolist()}")

            # پر کردن مقادیر گمشده برای ستون‌های عددی
            if not numeric_cols.empty:
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
                logging.debug("مقادیر گمشده ستون‌های عددی با میانگین پر شدند.")
            
            # پر کردن مقادیر گمشده برای ستون‌های غیرعددی
            if not non_numeric_cols.empty:
                for col in non_numeric_cols:
                    mode_value = df_cleaned[col].mode()
                    if not mode_value.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('')
                logging.debug("مقادیر گمشده ستون‌های غیرعددی با مد یا رشته خالی پر شدند.")

            # حذف داده‌های پرت برای ستون‌های عددی
            initial_rows = len(df_cleaned)
            for col in numeric_cols:
                if df_cleaned[col].var() > 0:  # فقط ستون‌هایی با واریانس غیرصفر
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    logging.debug(f"داده‌های پرت برای ستون {col} حذف شدند.")
                else:
                    logging.debug(f"ستون {col} واریانس صفر دارد و از حذف پرت‌ها صرف‌نظر شد.")

            # بررسی اینکه داده‌های کافی باقی مانده‌اند
            if len(df_cleaned) < 2 or df_cleaned[numeric_cols].dropna().empty:
                logging.error("پس از پاک‌سازی، داده‌های کافی یا ستون‌های عددی معتبر باقی نمانده است.")
                raise HTTPException(status_code=400, detail="پس از پاک‌سازی، داده‌های کافی یا ستون‌های عددی معتبر باقی نمانده است.")

            self.df = df_cleaned
            logging.info(f"پاک‌سازی داده‌ها با موفقیت انجام شد. ردیف‌های اولیه: {initial_rows}, ردیف‌های نهایی: {len(df_cleaned)}")
            return {
                "initial_rows": initial_rows,
                "cleaned_rows": len(df_cleaned),
                "message": "مقادیر گمشده پرشده با میانگین (ستون‌های عددی) و مد یا رشته خالی (ستون‌های غیرعددی). داده‌های پرت حذف شدند (روش IQR)."
            }
        except Exception as e:
            logging.error(f"خطا در پاک‌سازی داده‌ها: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.df is None:
            logging.error("هیچ داده‌ای برای داده‌کاوی وجود ندارد.")
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            desc_stats = self.df.describe(include='all').to_dict()
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            corr_matrix = self.df[numeric_cols].corr().to_dict() if not numeric_cols.empty else {}
            outlier_report = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                outlier_report[col] = len(outliers)
            logging.info("داده‌کاوی با موفقیت انجام شد.")
            return {
                "descriptive_stats": desc_stats,
                "correlation_matrix": corr_matrix,
                "outlier_report": outlier_report
            }
        except Exception as e:
            logging.error(f"خطا در داده‌کاوی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در داده‌کاوی: {str(e)}")

class Predictor:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        # تنظیم API Key برای Gemini
        os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')  # جایگزین با کلید واقعی
        try:
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model = "gemini-2.5-pro"
            logging.debug("اتصال به Gemini API با موفقیت برقرار شد.")
        except Exception as e:
            logging.error(f"خطا در اتصال به Gemini API: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در اتصال به Gemini API: {str(e)}")

    def analyze_dataset_with_gemini(self, df):
        logging.debug("شروع تحلیل دیتاست با Gemini API")
        try:
            # نمونه‌برداری از داده‌ها (حداکثر 100 ردیف)
            sample_size = min(100, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            logging.debug(f"نمونه‌برداری انجام شد: {sample_size} ردیف")

            # آماده‌سازی مشخصات دیتاست
            numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            all_cols = df.columns.tolist()
            desc_stats = df_sample.describe(include='all').to_string()
            corr_matrix = df_sample[numeric_cols].corr().to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
            num_rows, num_cols = df.shape
            missing_values = df.isnull().sum().sum()

            # ساخت پرامپت برای Gemini
            prompt = f"""
            شما یک متخصص یادگیری ماشین هستید. من یک نمونه از دیتاست با مشخصات زیر دارم (نمونه شامل {sample_size} ردیف است):
            - تعداد ردیف‌های کل دیتاست: {num_rows}
            - تعداد ستون‌ها: {num_cols}
            - تمام ستون‌ها: {all_cols}
            - آمار توصیفی (بر اساس نمونه):
            {desc_stats}
            - ماتریس همبستگی (برای ستون‌های عددی نمونه):
            {corr_matrix}
            - تعداد مقادیر گمشده در کل دیتاست: {missing_values}

            با توجه به این اطلاعات:
            1. بهترین ستون برای استفاده به عنوان ستون هدف (target) در رگرسیون را پیشنهاد دهید. ستون هدف باید عددی باشد و بر اساس همبستگی، واریانس، یا اهمیت پیش‌بینی انتخاب شود.
            2. بهترین الگوریتم یادگیری ماشین برای رگرسیون را از بین گزینه‌های زیر پیشنهاد دهید:
            Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, XGBoost (اگر موجود باشد).
            لطفاً فقط نام ستون هدف و نام الگوریتم را به صورت دقیق (مثلاً 'target_column: Sales' و 'model: Random Forest') و توضیح مختصری برای هر پیشنهاد ارائه دهید.
            """

            # تنظیم محتوا برای Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                ),
            )

            # دریافت پاسخ از Gemini
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text
            logging.debug("پاسخ Gemini دریافت شد.")

            # استخراج نام ستون هدف و الگوریتم از پاسخ
            recommended_target = None
            recommended_model = None
            available_models = ["Linear Regression", "Random Forest", "Decision Tree", 
                               "Gradient Boosting", "SVR"]
            if xgb:
                available_models.append("XGBoost")
            
            lower_response = response_text.lower()
            for col in numeric_cols:
                if col.lower() in lower_response and "target_column" in lower_response:
                    recommended_target = col
                    break
            
            for model_name in available_models:
                if model_name.lower() in lower_response:
                    recommended_model = model_name
                    break

            if recommended_target and recommended_model:
                logging.info(f"Gemini ستون هدف {recommended_target} و مدل {recommended_model} را پیشنهاد داد.")
                return recommended_target, recommended_model, response_text
            else:
                logging.error("Gemini نتوانست ستون هدف یا مدل مناسبی پیشنهاد دهد.")
                return None, None, response_text

        except Exception as e:
            logging.error(f"خطا در تحلیل دیتاست با Gemini API: {str(e)}", exc_info=True)
            return None, None, f"خطا در تحلیل دیتاست با Gemini API: {str(e)}"

    async def train_and_predict(self):
        if self.data_processor.df is None:
            logging.error("هیچ داده‌ای برای پیش‌بینی وجود ندارد.")
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            logging.debug("شروع فرآیند پیش‌بینی")
            # تحلیل دیتاست با Gemini API
            recommended_target, recommended_model, recommendation_text = self.analyze_dataset_with_gemini(self.data_processor.df)
            if not recommended_target or not recommended_model:
                logging.error(f"Gemini نتوانست ستون هدف یا الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")
                raise HTTPException(status_code=500, detail=f"Gemini نتوانست ستون هدف یا الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")

            target_column = recommended_target
            logging.debug(f"ستون هدف: {target_column}, مدل: {recommended_model}")

            # پردازش تمام ستون‌ها (عددی و غیرعددی)
            df_processed = pd.get_dummies(self.data_processor.df, drop_first=True)
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]

            # بررسی نوع داده‌های عددی برای هدف
            if y.dtype not in [np.float64, np.float32, np.int64, np.int32]:
                logging.error(f"ستون هدف {target_column} غیرعددی است.")
                raise HTTPException(status_code=400, detail="ستون هدف پیشنهادی باید عددی باشد.")

            if X.empty:
                logging.error("هیچ ستون برای ویژگی‌ها یافت نشد.")
                raise HTTPException(status_code=400, detail="هیچ ستون برای ویژگی‌ها یافت نشد.")

            if len(X) < 2 or len(y) < 2:
                logging.error("داده‌های کافی برای آموزش مدل وجود ندارد.")
                raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")

            # حذف ستون‌های با واریانس صفر یا کاملاً NaN
            X = X.loc[:, X.var(numeric_only=True) > 0]
            X = X.loc[:, X.notna().any()]
            X.fillna(X.mean(numeric_only=True), inplace=True)
            y.fillna(y.mean(), inplace=True)

            if X.empty or len(X.columns) == 0:
                logging.error("پس از حذف ستون‌های نامعتبر، هیچ ویژگی برای آموزش باقی نماند.")
                raise HTTPException(status_code=400, detail="هیچ ویژگی معتبری برای آموزش مدل باقی نماند.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # بررسی مقادیر نامعتبر در X_scaled
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logging.error("داده‌های استانداردشده شامل مقادیر NaN یا بی‌نهایت هستند.")
                raise HTTPException(status_code=400, detail="داده‌های استانداردشده شامل مقادیر نامعتبر (NaN یا بی‌نهایت) هستند.")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
                logging.error("داده‌های آزمایشی کافی نیست.")
                raise HTTPException(status_code=400, detail="داده‌های آزمایشی کافی نیست.")

            # تعریف مدل‌ها
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }
            if xgb:
                models["XGBoost"] = xgb.XGBRegressor(random_state=42)

            # بررسی معتبر بودن مدل پیشنهادی
            if recommended_model not in models:
                logging.error(f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")
                raise HTTPException(status_code=400, detail=f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")

            # آموزش و پیش‌بینی
            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                logging.debug(f"مدل {recommended_model} با موفقیت آموزش دید.")

                # تولید داده‌های آینده‌نگرانه (فرضی)
                future_indices = np.arange(len(y_test), len(y_test) + 5)
                future_X = X_test[-5:]
                future_pred = model.predict(future_X)
                logging.debug("پیش‌بینی‌های آینده تولید شدند.")

                # تولید نمودار خطی
                fig = plt.Figure(figsize=(14, 8))
                ax = fig.add_subplot(111)
                indices = np.arange(len(y_test))
                
                ax.plot(indices, y_test.values, color='blue', label=get_display(arabic_reshaper.reshape('مقادیر واقعی')), linewidth=2)
                ax.plot(indices, y_pred, color='orange', label=get_display(arabic_reshaper.reshape('مقادیر پیش‌بینی‌شده')), linewidth=2)
                ax.plot(future_indices, future_pred, color='green', linestyle='--', 
                        label=get_display(arabic_reshaper.reshape('پیش‌بینی آینده')), linewidth=2)
                
                ax.set_xlabel(get_display(arabic_reshaper.reshape("اندیس داده‌ها")))
                ax.set_ylabel(get_display(arabic_reshaper.reshape("مقادیر")))
                ax.set_title(get_display(arabic_reshaper.reshape(f"پیش‌بینی {target_column} با مدل {recommended_model}")))
                ax.legend()
                ax.grid(True)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
                
                # ذخیره نمودار
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                
                filename = f"prediction_{target_column}.png"
                with open(filename, "wb") as f:
                    f.write(buf.getvalue())
                logging.debug(f"نمودار در {filename} ذخیره شد.")

                logging.info(f"پیش‌بینی با مدل {recommended_model} برای ستون {target_column} با موفقیت انجام شد.")
                return {
                    "message": f"پیش‌بینی با مدل {recommended_model} انجام شد.",
                    "target_column": target_column,
                    "gemini_recommendation": recommendation_text,
                    "plot": f"/plot/{filename}"
                }

            except Exception as e:
                logging.error(f"خطا در آموزش مدل {recommended_model}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"خطا در آموزش مدل {recommended_model}: {str(e)}")

        except Exception as e:
            logging.error(f"خطا در فرآیند پیش‌بینی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در فرآیند پیش‌بینی: {str(e)}")

data_processor = DataProcessor()
predictor = Predictor(data_processor)

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        logging.error(f"فرمت فایل {file.filename} پشتیبانی نمی‌شود.")
        raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV یا Excel بارگذاری کنید.")
    return await data_processor.load_file(file)

@app.post("/predict")
async def predict():
    return await predictor.train_and_predict()

@app.get("/plot/{filename}")
async def get_plot(filename: str):
    if not os.path.exists(filename):
        logging.error(f"نمودار {filename} یافت نشد.")
        raise HTTPException(status_code=404, detail="نمودار یافت نشد.")
    with open(filename, "rb") as f:
        logging.debug(f"نمودار {filename} با موفقیت ارسال شد.")
        return StreamingResponse(io.BytesIO(f.read()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)