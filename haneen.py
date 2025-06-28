"""
Streamlit interface — Broken Plural Extraction Model (integrated)
================================================================
• Design: white background, dark sidebar, orange buttons, centered circular logo.
• Responsive: fits laptop & mobile (≤ 600 px) with no horizontal scroll.
• Back‑end: two‑stage analysis (sentence classification + word‑level Stanza filtering).
"""

# ──────────────────────────────────────────────
# 1) Imports & helpers
# ──────────────────────────────────────────────
import re
import string
import random
from pathlib import Path

import streamlit as st
import stanza
import joblib
from PIL import Image

# ──────────────────────────────────────────────
# 2) Page & global style
# ──────────────────────────────────────────────
st.set_page_config(page_title="Broken Plural Extraction", layout="wide")

# -----------------------------  Global CSS  -----------------------------
st.markdown(
    """
    <style>
    body, .stApp { background:#ffffff; color:#000000; }

    /* Title banner */
    .title-banner{
        background:#ffffff; max-width:55%; margin:0 auto 20px; padding:8px 6px;
        border-radius:10px; text-align:center; border:1px solid #e0e0e0;
    }
    .title-banner h1{color:#005f99; font-size:24px; margin:0}
    .title-banner p {color:#005f99; font-size:13px; margin:3px 0 0 0}

    /* Sidebar */
    section[data-testid="stSidebar"][aria-expanded="true"]{
        width:230px!important; background:#1e1e1e; color:#ffffff;
    }
    section[data-testid="stSidebar"] *{color:#ffffff!important;}

    /* Textarea */
    .stTextArea textarea{
        min-height:90px!important; max-height:90px!important;
        font-size:16px; background:#ffffff!important; color:#000000!important;
        resize:none!important;
    }

    /* Buttons */
    .stButton>button{
        background:#ff8c00!important; color:#ffffff!important;
        padding:6px 20px; font-size:.9rem; border:none; border-radius:6px;
    }
    .stButton>button:hover{background:#ffa733!important; color:#000000!important;}

    /* Centered circular logo */
    .stImage{display:flex; justify-content:center; margin-top:20px; margin-bottom:-10px;}
    .stImage img{
        width:130px!important; height:130px!important; border-radius:50%!important;
        border:3px solid #005f99; object-fit:cover;
    }

    /* Mobile tweaks */
    @media(max-width:600px){
        .title-banner{max-width:90%;}
        .title-banner h1{font-size:18px!important;}
        .title-banner p {font-size:11px!important;}
        .stButton>button{width:100%; margin-bottom:8px;}
        section[data-testid="stSidebar"][aria-expanded="true"]{width:180px!important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 3) Sidebar instructions
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📝 تعليمات:")
    st.markdown(
        """
1. ✍️ **أدخل جملةً باللغة العربية** في الصندوق الأبيض.
2. 🔍 **اضغط على زر "تحليل"** ليتم تحليل الجملة واستخراج جمع التكسير إن وُجد.
3. 🔁 **اضغط "إعادة التعيين"** للبدء من جديد.
4. ✏️ **اضغط "تجربة تلقائية"** لوضع جملة تجريبية.
"""
    )

# ──────────────────────────────────────────────
# 4) Load ML artifacts
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentence_artifacts():
    """Sentence‑level SVM"""
    vec = joblib.load(Path("finalmodelvectorizer.pkl"))
    clf = joblib.load(Path("finalmodel.pkl"))
    return vec, clf

@st.cache_resource(show_spinner=False)
def load_word_artifacts():
    """Word‑level Random‑Forest"""
    vec = joblib.load(Path("rf_final_vectorizer.pkl"))
    clf = joblib.load(Path("rf_final_model.pkl"))
    return vec, clf

sentence_vectorizer, sentence_model = load_sentence_artifacts()
word_vectorizer, word_model         = load_word_artifacts()

# ──────────────────────────────────────────────
# 5) NLP helpers
# ──────────────────────────────────────────────
_DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_PUNCT = string.punctuation + "«»…“”،؛؟ـ"
_PUNCT_TABLE = str.maketrans('', '', _PUNCT)

def remove_diacritics(text: str) -> str:
    return _DIACRITICS_PATTERN.sub('', text)

def remove_punctuation(text: str) -> str:
    return text.translate(_PUNCT_TABLE)

def clean_text(text: str) -> str:
    return remove_punctuation(remove_diacritics(text)).strip()

@st.cache_resource(show_spinner=False)
def load_stanza():
    return stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos')

nlp = load_stanza()

EXAMPLES = [
    "ذهب المعلمون إلى المدارس.",
    "يلعب الأطفال في الساحات الواسعة.",
    "قرأتُ الكتب المفيدة."
]

# ──────────────────────────────────────────────
# 6) Main layout
# ──────────────────────────────────────────────
col1, col2, logo_col, col4, col5 = st.columns(5)
with logo_col:
    if Path("logo.png").exists():
        st.image("logo.png", width=130)

st.markdown(
    """
    <div class='title-banner'>
        <h1>Broken Plural Extraction Model</h1>
        <p>نموذج ذكاء صناعي يحلل الجمل العربية لاستخراج جمع التكسير إن وُجد.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"✍️ مثال: {random.choice(EXAMPLES)}")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

input_txt = st.text_area("أدخل الجملة:", value=st.session_state.input_text)

col_analyze, col_demo, col_reset = st.columns(3)
with col_analyze:
    analyze_clicked = st.button("🔍 تحليل")
with col_demo:
    demo_clicked = st.button("✏️ تجربة تلقائية")
with col_reset:
    reset_clicked = st.button("📝 إعادة التعيين")

if demo_clicked:
    st.session_state.input_text = random.choice(EXAMPLES)
    st.rerun()

if reset_clicked:
    st.session_state.input_text = ""
    st.rerun()

# ──────────────────────────────────────────────
# 7) Analysis
# ──────────────────────────────────────────────
if analyze_clicked:
    if not input_txt.strip():
        st.warning("⚠️ يرجى إدخال جملة أولاً.")
    else:
        cleaned = clean_text(input_txt)
        _ = sentence_model.predict(sentence_vectorizer.transform([cleaned]))  # (مستقبلاً للاستفادة)

        doc = nlp(cleaned)
        candidates = [
            w.text for s in doc.sentences for w in s.words
            if w.upos in ["NOUN", "ADJ"] and w.feats and "Number=Plur" in w.feats
        ]

        if not candidates:
            st.markdown(
                "<div style='background-color:#e0f7fa;color:#003366;font-weight:bold;font-size:18px;padding:10px;border-radius:8px;'>"
                "❌ لم يتم العثور على جمع تكسير في هذه الجملة."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            preds = word_model.predict(word_vectorizer.transform(candidates))

            # DEBUG: أزل السطر التالي بعد الانتهاء
            # st.write(list(zip(candidates, preds)))

            results = [
                w for w, p in zip(candidates, preds)
                if p == 1 or (isinstance(p, str) and p.strip().lower() == "broken")
            ]

            if results:
                st.markdown("**🔍 الكلمات التي تحتوي على جمع تكسير:**")
                for w in results:
                    st.markdown(
                        f"<div style='background-color:#e0f7fa;color:#003366;font-weight:bold;font-size:18px;padding:10px;border-radius:8px;'>✅ الكلمة: {w}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div style='background-color:#e0f7fa;color:#003366;font-weight:bold;font-size:18px;padding:10px;border-radius:8px;'>"
                    "❌ لم يتم العثور على جمع تكسير في الكلمات المرشحة."
                    "</div>",
                    unsafe_allow_html=True,
                )

# ──────────────────────────────────────────────
# 8) Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <center>
        Palestine Technical University – Kadoorie (PTUK) – Computer Systems Engineering Department<br>
        
    </center>
    """,
    unsafe_allow_html=True,
)
