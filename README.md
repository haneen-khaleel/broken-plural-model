# Broken Plural Extraction App

هذا المشروع هو واجهة تفاعلية لاكتشاف **جمع التكسير** في الجمل العربية باستخدام الذكاء الاصطناعي.

## 🧠 ما يقدّمه المشروع
- تصنيف الجمل العربية لاكتشاف وجود جمع تكسير
- استخراج الكلمات التي تُعد جمع تكسير باستخدام أدوات تحليل لغوي
- واجهة مستخدم مبسطة ومناسبة للهاتف والكمبيوتر

## 🗃️ الملفات الرئيسية
- `haneen.py` → الكود الرئيسي لتشغيل الواجهة على Streamlit
- `finalmodel.pkl` → نموذج تصنيف الجمل
- `finalmodelvectorizer.pkl` → المحول اللغوي للجمل
- `rf-final.pkl` → نموذج تصنيف الكلمات
- `rf-finalvectorizer.pkl` → المحول اللغوي للكلمات

## ⚙️ المتطلبات
```bash
streamlit==1.34.0
stanza==1.7.0
joblib==1.4.2
scikit-learn==1.6.1
pillow==10.4.0
torch>=2.3.0 ; platform_system != "Windows"
torch==1.13.1+cpu ; platform_system == "Windows"
