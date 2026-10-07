import os
import re
import pickle
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify

# Preprocessing NLP Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)

app = Flask(__name__)

# ============================================================
# BASE DIRECTORY FOR MODEL LOADING (Vercel & Local Friendly)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD MODELS & SCALER
# ============================================================
skipgram_model = None
naive_bayes_model = None
scaler = None
model_load_error = None

try:
    with open(os.path.join(BASE_DIR, "skipgram_model.pkl"), "rb") as f:
        skipgram_model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "naive_bayes_skipgram.pkl"), "rb") as f:
        naive_bayes_model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "scaler_skipgram.pkl"), "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    model_load_error = str(e)

# ============================================================
# INITIALIZE SASTRAWI PREPROCESSING
# ============================================================
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def preprocessing(text):
    """
    Melakukan preprocessing teks berita:
    1. Case folding (lowercase)
    2. Menghapus karakter non-alphanumeric & tanda baca
    3. Menghapus angka
    4. Normalisasi spasi
    5. Stopword removal (Sastrawi)
    6. Stemming (Sastrawi)
    7. Tokenisasi
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Sastrawi stopword removal & stemming
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    
    tokens = text.split()
    return tokens

# ============================================================
# WEBSCRAPING FUNCTION FOR ANY NEWS URL
# ============================================================
def get_news_content(url):
    """
    Mengambil judul dan isi artikel dari link URL berita apapun.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract article title
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.find("title"):
        title = soup.find("title").get_text(strip=True)

    # Clean non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Extract paragraph texts
    paragraphs = soup.find_all("p")
    contents = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 20:  # exclude tiny metadata lines
            contents.append(txt)

    article_text = " ".join(contents)
    article_text = re.sub(r"\s+", " ", article_text).strip()

    return title, article_text

# ============================================================
# DOCUMENT VECTOR GENERATION (SKIP-GRAM AVERAGE)
# ============================================================
def document_vector(model, tokens):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
            
    if len(vectors) == 0:
        return [0.0] * model.vector_size

    return (sum(vectors) / len(vectors)).tolist()

# ============================================================
# URL CATEGORY CHECK (HEURISTICS)
# ============================================================
def cek_kategori_url(url):
    url_lower = url.lower()

    finance_keywords = ["finance", "finansial", "bisnis", "business", "market", "ekonomi", "investasi", "bursa"]
    sport_keywords = ["sport", "sports", "olahraga", "sepakbola", "bola", "football", "soccer", "motogp", "badminton"]
    other_keywords = [
        "health", "kesehatan", "fotohealth", "lifestyle", "gaya-hidup",
        "travel", "wisata", "food", "kuliner", "entertainment", "hiburan",
        "technology", "teknologi", "tekno", "otomotif", "detikhot", "wolipop", "inet"
    ]

    for kw in finance_keywords:
        if kw in url_lower:
            return True, "finance"

    for kw in sport_keywords:
        if kw in url_lower:
            return True, "sport"

    for kw in other_keywords:
        if kw in url_lower:
            return False, "other"

    return True, "unknown"

# ============================================================
# LABEL NORMALIZATION (Sport, Finance, Other)
# ============================================================
def normalisasi_label(label):
    lbl = str(label).lower().strip()
    if lbl in ["finance", "finansial", "bisnis", "business", "ekonomi"]:
        return "finance"
    elif lbl in ["sport", "sports", "olahraga", "suport"]:
        return "sport"
    return "other"

# ============================================================
# ROUTE: WEB UI
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ============================================================
# ROUTE: API CLASSIFICATION ENDPOINT
# ============================================================
@app.route("/api/classify", methods=["POST"])
def classify():
    if model_load_error:
        return jsonify({
            "status": "error",
            "message": f"Model gagal dimuat: {model_load_error}"
        }), 500

    # Get URL from JSON or Form Data
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "").strip()

    if not url:
        return jsonify({
            "status": "error",
            "message": "Silakan masukkan URL berita terlebih dahulu."
        }), 400

    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({
            "status": "error",
            "message": "URL harus diawali dengan http:// atau https://"
        }), 400

    try:
        # 1. Cek URL heuristic category
        _, url_category = cek_kategori_url(url)

        # 2. Extract news text from URL
        title, article_text = get_news_content(url)

        if not article_text:
            return jsonify({
                "status": "error",
                "message": "Gagal mengambil isi berita dari URL. Pastikan link dapat diakses publik."
            }), 400

        # 3. Preprocessing text
        tokens = preprocessing(article_text)

        if not tokens:
            return jsonify({
                "status": "error",
                "message": "Tidak ada kata yang valid untuk diproses setelah preprocessing."
            }), 400

        # 4. Skip-Gram Vector Representation
        vec = document_vector(skipgram_model, tokens)

        # 5. Scaling with MinMaxScaler
        vec_scaled = scaler.transform([vec])

        # 6. Predict using Naive Bayes Model
        probabilities = naive_bayes_model.predict_proba(vec_scaled)[0]
        classes = naive_bayes_model.classes_

        max_idx = probabilities.argmax()
        max_prob = float(probabilities[max_idx])
        raw_pred_label = classes[max_idx]
        predicted_label = normalisasi_label(raw_pred_label)

        # Build probabilities dictionary with normalized labels
        prob_dict = {}
        for cls_name, prob in zip(classes, probabilities):
            norm_cls = normalisasi_label(cls_name)
            prob_dict[norm_cls] = round(float(prob) * 100, 2)

        # 7. Apply Thresholding Logic & URL Overrides
        threshold = 0.60
        final_prediction = predicted_label
        explanation = ""

        if url_category == "other":
            final_prediction = "other"
            explanation = "URL terindikasi dari kategori di luar Finance & Sport."
        elif max_prob < threshold:
            final_prediction = "other"
            explanation = f"Confidence model ({max_prob * 100:.1f}%) di bawah threshold {threshold * 100:.0f}%."
        else:
            explanation = f"Diklasifikasikan sebagai {final_prediction.upper()} dengan tingkat keyakinan {max_prob * 100:.1f}%."

        return jsonify({
            "status": "success",
            "data": {
                "url": url,
                "title": title or "Berita Tanpa Judul",
                "label": final_prediction,  # 'sport', 'finance', or 'other'
                "confidence": round(max_prob * 100, 2),
                "explanation": explanation,
                "probabilities": prob_dict,
                "token_count": len(tokens),
                "tokens": tokens[:50],  # sample first 50 tokens
                "article_preview": article_text[:500] + ("..." if len(article_text) > 500 else ""),
                "full_article": article_text
            }
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "status": "error",
            "message": "Waktu pengambilan berita habis (Timeout). Situs web tidak merespons."
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal mengambil berita dari URL: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan internal: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Local development server (without needing .env)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)