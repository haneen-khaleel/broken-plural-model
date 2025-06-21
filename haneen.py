# --- تحميل المكتبات ---
import streamlit as st
import stanza
import re
import string
import joblib
from PIL import Image

# --- إعداد الصفحة ---
st.set_page_config(page_title="Broken Plural Extraction Model", layout="centered")

# --- تنسيقات CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: white !important;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .logo-container img {
            width: 230px;
        }

        .title-banner {
            background-color: #FFF9C4;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 25px;
        }

        .title-banner h1 {
            color: #003366 !important;
            font-size: 34px !important;
            margin: 0;
        }

        .title-banner p {
            color: #003366;
            font-size: 20px;
            font-weight: 500;
            margin-top: 8px;
        }

        .stTextArea textarea {
            background-color: #262730;
            color: white;
            font-size: 18px;
        }

        .stButton > button {
            background-color: #111;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }

        .result-text {
            color: #003366 !important;
            font-weight: bold;
            font-size: 18px;
        }

        .footer-note {
            text-align: center;
            font-size: 12px;
            color: #555555;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# --- عرض الشعار ---
image = Image.open("logo.png")  # تأكدي أن اسم الصورة صحيح
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image(image)
st.markdown('</div>', unsafe_allow_html=True)

# --- عنوان الواجهة ---
st.markdown("""
    <div class="title-banner">
        <h1>Broken Plural Extraction Model</h1>
        <p>نموذج استخراج جمع التكسير</p>
    </div>
""", unsafe_allow_html=True)

# --- تحميل النماذج ---
sentence_model = joblib.load("svc_test_model.pkl")
sentence_vectorizer = joblib.load("svc_test_vectorizer.pkl")

# ✅ تعديل أسماء ملفات نموذج الكلمات والفيكتورايزر إلى الامتداد الصحيح
word_model = joblib.load("rf_final_model.pkl")
word_vectorizer = joblib.load("rf_final_vectorizer.pkl")

# --- تحميل مكتبة Stanza ---
nlp = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos')

# --- دوال التنظيف ---
def remove_diacritics(text):
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation + '«»…“”،؛؟ـ'))

def clean_text(text):
    return remove_punctuation(remove_diacritics(text)).strip()

# --- واجهة الإدخال ---
input_text = st.text_area("أدخل الجملة:")

if st.button("تحليل"):
    if not input_text.strip():
        st.warning("⚠️ يرجى إدخال جملة أولاً.")
    else:
        cleaned = clean_text(input_text)

        # نموذج الجملة
        tfidf_sent = sentence_vectorizer.transform([cleaned])
        sent_pred = sentence_model.predict(tfidf_sent)[0]

        # تحليل الكلمات
        doc = nlp(cleaned)
        candidates = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ["NOUN", "ADJ"] and word.feats and "Number=Plur" in word.feats:
                    candidates.append(word.text)

        if not candidates:
            st.markdown("""
                <div style='background-color:#ffecec; padding:10px; border-radius:10px; color:#003366; font-weight:bold; font-size:18px;'>
                    ❌ الجملة لا تحتوي على جمع تكسير
                </div>
            """, unsafe_allow_html=True)
        else:
            X = word_vectorizer.transform(candidates)
            preds = word_model.predict(X)

            found = False
            for word, label in zip(candidates, preds):
                if label == 1 or (isinstance(label, str) and label.strip() == "broken"):
                    st.markdown(f"<div class='result-text'>✅ الكلمة: {word}</div>", unsafe_allow_html=True)
                    found = True
                    break

            if not found:
                st.markdown("""
                    <div style='background-color:#ffecec; padding:10px; border-radius:10px; color:#003366; font-weight:bold; font-size:18px;'>
                        ❌ الجملة لا تحتوي على جمع تكسير
                    </div>
                """, unsafe_allow_html=True)

# --- التذييل ---
st.markdown("""
    <div class="footer-note">
        Palestine Technical University – Kadoorie (PTUK) – Computer Systems Engineering Department
    </div>
""", unsafe_allow_html=True)
