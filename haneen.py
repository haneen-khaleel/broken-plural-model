# --- تحميل المكتبات ---
import streamlit as st
import stanza
import re
import string
import joblib
from PIL import Image
import random

# --- إعداد الصفحة ---
st.set_page_config(page_title="Broken Plural Extraction Model", layout="centered")

# --- تنسيقات CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #F7F9FC !important;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .logo-container img {
            width: 200px;
        }
        .title-banner {
            background: linear-gradient(to right, #003366, #005f99);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        .title-banner h1 {
            color: white !important;
            font-size: 36px !important;
            margin: 0;
        }
        .title-banner p {
            color: #e0f7fa;
            font-size: 18px;
            margin-top: 10px;
        }
        .stTextArea textarea {
            background-color: #fff;
            color: #003366;
            font-size: 18px;
            border: 1px solid #cccccc;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #003366;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 0.6em 1.4em;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #005f99;
            transform: scale(1.03);
        }
        .result-text {
            background-color: #e0f7fa;
            color: #003366 !important;
            font-weight: bold;
            font-size: 20px;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .footer-note {
            text-align: center;
            font-size: 12px;
            color: #555555;
            margin-top: 40px;
        }
        .example-text {
            color: #003366;
            font-weight: 500;
            font-size: 16px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- الشريط الجانبي ---
st.sidebar.title("📘 تعليمات")
st.sidebar.markdown("""
- أدخل جملة باللغة العربية.
- النموذج يحلل الجملة ويبحث عن جمع التكسير.
- يدعم الكلمات من نوع اسم أو صفة جمع فقط.
- يمكنك تجربة جملة جاهزة أو إعادة التعيين.
""")

# --- عرض الشعار ---
image = Image.open("logo.png")
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
word_model = joblib.load("rf_final_model.pkl")
word_vectorizer = joblib.load("rf_final_vectorizer.pkl")

# --- تحميل مكتبة Stanza ---
nlp = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos')

# --- دوال المساعدة ---
def remove_diacritics(text):
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation + '«»…“”،؛؟ـ'))

def clean_text(text):
    return remove_punctuation(remove_diacritics(text)).strip()

examples = [
    "ذهب المعلمون إلى المدارس.",
    "يلعب الأطفال في الساحات الواسعة.",
    "قرأتُ الكتب المفيدة."
]

# --- التعامل مع الجملة ---
input_key = "input_text"
if "input_text_buffer" not in st.session_state:
    st.session_state.input_text_buffer = ""

st.markdown(f"<div class='example-text'>✍️ مثال: {random.choice(examples)}</div>", unsafe_allow_html=True)

input_text = st.text_area("أدخل الجملة:", value=st.session_state.input_text_buffer, key=input_key)

col1, col2, col3 = st.columns(3)
with col1:
    analyze = st.button("🔍 تحليل")
with col2:
    example_btn = st.button("🧪 تجربة تلقائية")
with col3:
    reset_btn = st.button("🔁 إعادة التعيين")

if example_btn:
    st.session_state.input_text_buffer = random.choice(examples)
    st.rerun()

if reset_btn:
    st.session_state.input_text_buffer = ""
    st.rerun()

# --- التحليل ---
if analyze:
    if not input_text.strip():
        st.warning("⚠️ يرجى إدخال جملة أولاً.")
    else:
        cleaned = clean_text(input_text)
        tfidf_sent = sentence_vectorizer.transform([cleaned])
        sent_pred = sentence_model.predict(tfidf_sent)[0]

        doc = nlp(cleaned)
        candidates = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ["NOUN", "ADJ"] and word.feats and "Number=Plur" in word.feats:
                    candidates.append(word.text)

        if not candidates:
            st.markdown("""
                <div class='result-text'>
                    😅 لم يتم العثور على جمع تكسير في هذه الجملة.
                </div>
            """, unsafe_allow_html=True)
        else:
            X = word_vectorizer.transform(candidates)
            preds = word_model.predict(X)
            results = [(w, l) for w, l in zip(candidates, preds) if l == 1 or (isinstance(l, str) and l.strip() == "broken")]

            if results:
                st.markdown("**🔍 الكلمات التي تحتوي على جمع تكسير:**")
                for word, _ in results:
                    st.markdown(f"<div class='result-text'>✅ الكلمة: {word}</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='result-text'>
                        ❌ لم يتم العثور على جمع تكسير في الكلمات المرشحة.
                    </div>
                """, unsafe_allow_html=True)

# --- التذييل ---
st.markdown("""
    <div class="footer-note">
        Palestine Technical University – Kadoorie (PTUK) – Computer Systems Engineering Department
    </div>
""", unsafe_allow_html=True)
