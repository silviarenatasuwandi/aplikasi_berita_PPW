"""
================================================================================
APLIKASI KLASIFIKASI BERITA INDONESIA
Mata Kuliah : Pencarian & Penambangan Web (PPW)
Arsitektur  : Web Scraping -> NLP Preprocessing (Tanpa Stemming)
              -> Word2Vec Skip-Gram (Dokumen Embedding)
              -> MinMaxScaler -> Multinomial Naive Bayes
Framework   : Flask (Backend API & Web UI)
================================================================================
"""

import os
import sys
import re
import pickle
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify

# Preprocessing Stopword Bahasa Indonesia menggunakan Sastrawi
# CATATAN: Stemming sengaja DIHAPUS untuk menjaga makna kata dan kecepatan sistem.
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Scikit-Learn & Gensim untuk Pipeline Training & Klasifikasi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec

# Inisialisasi Flask App
app = Flask(__name__)

# ==============================================================================
# KONFIGURASI PATH & FILE MODEL
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_SKIPGRAM_MODEL = os.path.join(BASE_DIR, "skipgram_model.pkl")
PATH_NAIVE_BAYES_MODEL = os.path.join(BASE_DIR, "naive_bayes_skipgram.pkl")
PATH_SCALER = os.path.join(BASE_DIR, "scaler_skipgram.pkl")
PATH_DATASET = os.path.join(BASE_DIR, "dataset_detik_200_berita.csv")
PATH_EVALUASI = os.path.join(BASE_DIR, "hasil_grid_search.csv")

# ==============================================================================
# INISIALISASI NLP PREPROCESSING (SASTRAWI)
# ==============================================================================
# Membuat objek stopword remover dari pustaka Sastrawi untuk menyaring kata-kata
# umum bahasa Indonesia (seperti: 'dan', 'di', 'dari', 'yang', 'adalah', dll).
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()


# ==============================================================================
# PIPELINE FUNGSI 1: VALIDASI INPUT URL
# ==============================================================================
def validasi_url(url: str):
    """
    [TAHAP 1: VALIDASI INPUT]
    Memeriksa apakah URL yang diinputkan pengguna tidak kosong dan
    menggunakan skema protokol web yang valid (http:// atau https://).
    """
    if not url:
        return False, "URL berita tidak boleh kosong. Silakan masukkan link berita."
    
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False, "Format URL tidak valid. URL harus diawali dengan http:// atau https://"
    
    return True, url


# ==============================================================================
# PIPELINE FUNGSI 2: WEB SCRAPING BERITA (SCRAPER)
# ==============================================================================
def scrape_konten_berita(url: str):
    """
    [TAHAP 2: WEB SCRAPING]
    Mengambil konten teks berita dari halaman web secara otomatis menggunakan
    Requests (HTTP GET) dan BeautifulSoup (HTML Parsing).

    Alur Kerja Scraper:
    1. Mengirim HTTP Request dengan header User-Agent menyerupai browser asli.
    2. Parsing dokumen HTML menggunakan BeautifulSoup.
    3. Ekstraksi Judul Berita dari tag <h1> (atau <title> sebagai fallback).
    4. Membersihkan elemen pengganggu / non-konten (<script>, <style>, <nav>, <footer>, dll).
    5. Mengambil seluruh teks paragraf dari tag <p>, memfilter teks pendek / iklan.
    6. Menggabungkan seluruh paragraf menjadi satu kesatuan teks artikel.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # Request ke web berita dengan batas waktu timeout 15 detik
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 1. Ekstraksi Judul Berita
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.find("title"):
        title = soup.find("title").get_text(strip=True)

    # 2. Hapus tag yang bukan bagian dari artikel utama
    for unwanted_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        unwanted_tag.decompose()

    # 3. Ekstraksi Paragraf Berita (tag <p>)
    paragraphs = soup.find_all("p")
    content_list = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        # Filter: abaikan paragraf yang terlalu pendek (< 25 karakter) karena umumnya teks metadata/iklan
        if txt and len(txt) > 25:
            content_list.append(txt)

    article_text = " ".join(content_list)
    article_text = re.sub(r"\s+", " ", article_text).strip()

    return title, article_text


# ==============================================================================
# PIPELINE FUNGSI 3: TEXT PREPROCESSING (PEMBERSIHAN TEKS - TANPA STEMMING)
# ==============================================================================
def preprocess_teks(text: str):
    """
    [TAHAP 3: NLP PREPROCESSING]
    Membersihkan teks berita mentah menjadi deretan token kata yang bersih.

    Langkah-langkah Pemrosesan:
    1. Case Folding      : Mengubah teks menjadi huruf kecil (lowercase).
    2. Cleaning Simbol   : Menghapus tanda baca & simbol selain huruf dan spasi.
    3. Cleaning Angka    : Menghapus angka/digit numerik (karena tidak signifikan untuk kategori topik).
    4. Whitespace Normal : Merapikan spasi berlebih dan spasi di ujung teks.
    5. Stopword Removal  : Membuang kata hubung / kata umum menggunakan Sastrawi.
       (CATATAN: Tahap Stemming DIHAPUS agar kata tetap mempertahankan bentuk aslinya,
       menghindari distorsi semantik dari over-stemming, dan mempercepat runtime proses).
    6. Tokenisasi        : Memecah string teks menjadi kumpulan kata (tokens).
    """
    # 1. Case Folding
    text = str(text).lower()

    # 2. Menghapus tanda baca dan karakter khusus
    text = re.sub(r"[^\w\s]", " ", text)

    # 3. Menghapus angka
    text = re.sub(r"\d+", " ", text)

    # 4. Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Stopword Removal menggunakan Sastrawi
    text = stopword_remover.remove(text)

    # 6. Tokenisasi (pemisahan kata berdasarkan spasi)
    tokens = text.split()

    return tokens


# ==============================================================================
# PIPELINE FUNGSI 4: EKSTRAKSI FITUR (WORD EMBEDDING SKIP-GRAM DOKUMEN)
# ==============================================================================
def ekstraksi_vektor_skipgram(tokens: list, model_w2v):
    """
    [TAHAP 4: EKSTRAKSI FITUR / WORD EMBEDDING]
    Mengubah daftar token kata menjadi satu vektor numerik representasi dokumen
    menggunakan metode Average Pooling dari Word2Vec Skip-Gram.

    Konsep Matematika:
    - Setiap kata 'w' dalam vocabulary memiliki vektor berukuran 100 dimensi: V(w)
    - Representasi seluruh dokumen dihitung dari rata-rata (mean) seluruh vektor kata:
      Vektor_Dokumen = (1 / N) * SUM_{i=1}^{N} V(w_i)
    - Jika tidak ada kata yang terdaftar dalam vocabulary, dikembalikan vektor nol.
    """
    vectors = []
    for word in tokens:
        if word in model_w2v.wv:
            vectors.append(model_w2v.wv[word])

    if len(vectors) == 0:
        return np.zeros(model_w2v.vector_size)

    # Rata-rata vektor kata untuk membentuk 1 vektor dokumen
    doc_vec = np.mean(vectors, axis=0)
    return doc_vec


# ==============================================================================
# PIPELINE FUNGSI 5: FEATURE SCALING (MINMAX SCALER)
# ==============================================================================
def skala_vektor_dokumen(doc_vec, scaler_model):
    """
    [TAHAP 5: FEATURE SCALING]
    Mentransformasikan vektor dokumen ke rentang [0, 1] menggunakan MinMaxScaler.

    Alasan Teoretis:
    - Vektor Word2Vec bernilai kontinu dan dapat memiliki nilai negatif (rentang [-1, 1]).
    - Algoritma Multinomial Naive Bayes mengasumsikan frekuensi fitur dan TIDAK
      menerima input fitur bernilai negatif.
    - MinMaxScaler memastikan seluruh komponen vektor berada pada rentang [0, 1]
      sehingga kompatibel dengan Multinomial Naive Bayes.
    """
    # Mengubah bentuk array menjadi matriks 2D (1 baris, 100 kolom)
    doc_vec_2d = np.array(doc_vec).reshape(1, -1)
    doc_vec_scaled = scaler_model.transform(doc_vec_2d)
    return doc_vec_scaled


# ==============================================================================
# PIPELINE FUNGSI 6: KLASIFIKASI NAIVE BAYES & HITUNG PROBABILITAS
# ==============================================================================
def prediksi_naive_bayes(doc_vec_scaled, nb_model):
    """
    [TAHAP 6: KLASIFIKASI NAIVE BAYES]
    Menggunakan model Multinomial Naive Bayes untuk menghitung probabilitas
    posterior dari dokumen untuk tiap kelas kategori (Sport dan Finance).

    Output:
    - raw_label: Kelas dengan probabilitas terbesar
    - max_prob: Nilai probabilitas tertinggi (0.0 s.d 1.0)
    - prob_dict: Dictionary persentase probabilitas untuk setiap kelas
    """
    probabilities = nb_model.predict_proba(doc_vec_scaled)[0]
    classes = nb_model.classes_

    max_idx = int(np.argmax(probabilities))
    max_prob = float(probabilities[max_idx])
    raw_label = str(classes[max_idx])

    prob_dict = {}
    for cls_name, prob in zip(classes, probabilities):
        norm_name = normalisasi_label(cls_name)
        prob_dict[norm_name] = round(float(prob) * 100, 2)

    return raw_label, max_prob, prob_dict


# ==============================================================================
# PIPELINE FUNGSI 7: POST-PROCESSING, HEURISTIK URL & THRESHOLDING
# ==============================================================================
def normalisasi_label(label: str) -> str:
    """
    Menyeragamkan penamaan label ke dalam kategori baku: 'sport', 'finance', 'other'.
    """
    lbl = str(label).lower().strip()
    if lbl in ["finance", "finansial", "bisnis", "business", "ekonomi"]:
        return "finance"
    elif lbl in ["sport", "sports", "olahraga", "sepakbola"]:
        return "sport"
    return "other"


def cek_heuristik_url(url: str):
    """
    Mendeteksi kanal berita berdasarkan kata kunci yang tercantum pada URL.
    Berguna sebagai filter awal untuk mendeteksi berita yang jelas di luar cakupan
    model (misalnya: berita kesehatan, otomotif, hiburan, gaya hidup).
    """
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
            return "finance"

    for kw in sport_keywords:
        if kw in url_lower:
            return "sport"

    for kw in other_keywords:
        if kw in url_lower:
            return "other"

    return "unknown"


def evaluasi_kategori_final(url: str, predicted_label: str, max_prob: float, threshold: float = 0.60):
    """
    [TAHAP 7: DECISION LOGIC & THRESHOLDING]
    Menentukan kategori final berdasarkan hasil prediksi model, ambang batas keyakinan
    (confidence threshold), dan heuristik URL.

    Aturan Keputusan:
    1. Jika URL berasal dari kanal lain (misal: health, wolipop), maka final label = 'other'.
    2. Jika tingkat keyakinan (max_prob) di bawah threshold (< 60%), maka final label = 'other'
       karena model tidak cukup yakin membedakan apakah itu Sport atau Finance.
    3. Jika lolos kedua syarat di atas, gunakan kategori hasil prediksi model.
    """
    url_category = cek_heuristik_url(url)
    norm_pred = normalisasi_label(predicted_label)

    if url_category == "other":
        final_label = "other"
        penjelasan = "URL terindikasi berasal dari topik di luar Finance & Sport (terdeteksi via filter kanal URL)."
    elif max_prob < threshold:
        final_label = "other"
        penjelasan = (
            f"Tingkat keyakinan model ({max_prob * 100:.1f}%) berada di bawah batas ambang "
            f"threshold ({threshold * 100:.0f}%), sehingga dikategorikan sebagai OTHER."
        )
    else:
        final_label = norm_pred
        penjelasan = (
            f"Berita berhasil diklasifikasikan sebagai {final_label.upper()} "
            f"dengan tingkat keyakinan {max_prob * 100:.1f}%."
        )

    return final_label, penjelasan


# ==============================================================================
# PIPELINE UTAMA: END-TO-END PREDIKSI BERITA
# ==============================================================================
def pipeline_klasifikasi_berita(url: str):
    """
    ============================================================================
    PIPELINE LENGKAP END-TO-END (DARI INPUT LINK SAMPAI MUNCUL PREDIKSI)
    ============================================================================
    Fungsi ini menyatukan seluruh tahapan proses secara berurutan dan terstruktur:
      Tahap 1 : Validasi format URL input
      Tahap 2 : Scraping judul dan isi teks artikel dari URL (Requests & BeautifulSoup)
      Tahap 3 : Preprocessing teks (Case folding, cleaning simbol & angka,
                stopword removal Sastrawi, tokenisasi — TANPA STEMMING)
      Tahap 4 : Ekstraksi fitur teks ke vektor 100 dimensi (Word2Vec Skip-Gram)
      Tahap 5 : Normalisasi vektor ke rentang [0, 1] (MinMaxScaler)
      Tahap 6 : Perhitungan probabilitas kelas (Multinomial Naive Bayes)
      Tahap 7 : Evaluasi ambang batas keyakinan & heuristik URL (Final Class)
      Tahap 8 : Penyusunan hasil luaran terstruktur untuk Web UI & API
    ============================================================================
    """
    # --------------------------------------------------------------------------
    # TAHAP 1: Validasi Input URL
    # --------------------------------------------------------------------------
    is_valid, url_or_msg = validasi_url(url)
    if not is_valid:
        raise ValueError(url_or_msg)
    valid_url = url_or_msg

    # --------------------------------------------------------------------------
    # TAHAP 2: Web Scraping Konten Berita
    # --------------------------------------------------------------------------
    title, article_text = scrape_konten_berita(valid_url)
    if not article_text:
        raise ValueError("Gagal mengekstrak teks berita dari URL. Pastikan artikel dapat diakses publik.")

    # --------------------------------------------------------------------------
    # TAHAP 3: NLP Preprocessing (Tanpa Stemming)
    # --------------------------------------------------------------------------
    tokens = preprocess_teks(article_text)
    if not tokens:
        raise ValueError("Tidak ada token kata yang valid setelah proses preprocessing.")

    # --------------------------------------------------------------------------
    # TAHAP 4: Ekstraksi Fitur Dokumen (Word2Vec Skip-Gram Averaging)
    # --------------------------------------------------------------------------
    doc_vec = ekstraksi_vektor_skipgram(tokens, skipgram_model)

    # --------------------------------------------------------------------------
    # TAHAP 5: Feature Scaling (MinMaxScaler)
    # --------------------------------------------------------------------------
    doc_vec_scaled = skala_vektor_dokumen(doc_vec, scaler_model)

    # --------------------------------------------------------------------------
    # TAHAP 6: Klasifikasi & Perhitungan Probabilitas (Naive Bayes)
    # --------------------------------------------------------------------------
    raw_pred, max_prob, prob_dict = prediksi_naive_bayes(doc_vec_scaled, naive_bayes_model)

    # --------------------------------------------------------------------------
    # TAHAP 7: Post-Processing & Confidence Thresholding
    # --------------------------------------------------------------------------
    final_label, penjelasan = evaluasi_kategori_final(
        url=valid_url,
        predicted_label=raw_pred,
        max_prob=max_prob,
        threshold=0.60
    )

    # Pastikan label 'other' tercantum dalam dictionary probabilitas tampilan
    if "other" not in prob_dict:
        if final_label == "other":
            prob_dict["other"] = round((1.0 - max_prob) * 100, 2)
        else:
            prob_dict["other"] = 0.0

    # --------------------------------------------------------------------------
    # TAHAP 8: Response Dictionary
    # --------------------------------------------------------------------------
    return {
        "url": valid_url,
        "title": title or "Berita Tanpa Judul",
        "label": final_label,
        "confidence": round(max_prob * 100, 2),
        "explanation": penjelasan,
        "probabilities": prob_dict,
        "token_count": len(tokens),
        "tokens": tokens[:50],  # Menampilkan sampel 50 token pertama untuk UI
        "article_preview": article_text[:500] + ("..." if len(article_text) > 500 else ""),
        "full_article": article_text
    }


# ==============================================================================
# PIPELINE TRAINING MODEL: PELATIHAN MODEL SKIP-GRAM & NAIVE BAYES
# ==============================================================================
def latih_dan_simpan_model():
    """
    Fungsi untuk melatih ulang seluruh model machine learning dari dataset CSV:
    1. Membaca dataset 200 berita (dataset_detik_200_berita.csv)
    2. Melakukan Text Preprocessing (Stopword Sastrawi, TANPA STEMMING)
    3. Split Data: 80% Training, 20% Testing (Stratified)
    4. Melatih Word2Vec Skip-Gram (sg=1, vector_size=100, window=5)
    5. Normalisasi vektor fitur menggunakan MinMaxScaler
    6. Melatih Multinomial Naive Bayes (alpha=0.1)
    7. Evaluasi performa (Akurasi & F1-Score)
    8. Menyimpan model ke format file pickle (.pkl)
    """
    print("\n" + "="*70)
    print("MEMULAI PELATIHAN MODEL (SKIP-GRAM + NAIVE BAYES - TANPA STEMMING)")
    print("="*70)

    if not os.path.exists(PATH_DATASET):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {PATH_DATASET}")

    # 1. Load Data
    df = pd.read_csv(PATH_DATASET).dropna(subset=["isi_berita", "label"]).copy()
    print(f"Dataset berhasil dimuat: {len(df)} baris data.")
    print("Distribusi Kategori:")
    print(df["label"].value_counts())

    # 2. Preprocessing
    print("\nMelakukan preprocessing teks (pembersihan, stopword, tanpa stemming)...")
    df["tokens"] = df["isi_berita"].apply(preprocess_teks)

    # 3. Split Dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        df["tokens"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"Data Training: {len(X_train)} data | Data Testing: {len(X_test)} data")

    # 4. Training Word2Vec Skip-Gram
    print("\nMelatih Word2Vec Skip-Gram (vector_size=100, window=5, epochs=20)...")
    model_w2v = Word2Vec(
        sentences=X_train.tolist(),
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,  # sg=1 berarti Skip-Gram (sg=0 adalah CBOW)
        epochs=20,
        seed=42
    )

    # Ekstraksi vektor training dan testing
    X_train_vec = np.array([ekstraksi_vektor_skipgram(t, model_w2v) for t in X_train])
    X_test_vec = np.array([ekstraksi_vektor_skipgram(t, model_w2v) for t in X_test])

    # 5. Normalisasi Fitur (MinMaxScaler)
    print("Normalisasi vektor fitur menggunakan MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_vec)
    X_test_scaled = scaler.transform(X_test_vec)

    # 6. Training Multinomial Naive Bayes
    print("Melatih model Multinomial Naive Bayes (alpha=0.1)...")
    model_nb = MultinomialNB(alpha=0.1)
    model_nb.fit(X_train_scaled, y_train)

    # 7. Evaluasi Model pada Data Testing
    y_pred = model_nb.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nHasil Evaluasi Testing:")
    print(f"  - Akurasi   : {acc * 100:.2f}%")
    print(f"  - Presisi   : {prec * 100:.2f}%")
    print(f"  - Recall    : {rec * 100:.2f}%")
    print(f"  - F1-Score  : {f1 * 100:.2f}%")

    # 8. Simpan Model ke File Pickle
    with open(PATH_SKIPGRAM_MODEL, "wb") as f:
        pickle.dump(model_w2v, f, protocol=4)
    print(f"Model Skip-Gram tersimpan ke: {PATH_SKIPGRAM_MODEL}")

    with open(PATH_NAIVE_BAYES_MODEL, "wb") as f:
        pickle.dump(model_nb, f, protocol=4)
    print(f"Model Naive Bayes tersimpan ke: {PATH_NAIVE_BAYES_MODEL}")

    with open(PATH_SCALER, "wb") as f:
        pickle.dump(scaler, f, protocol=4)
    print(f"Scaler tersimpan ke: {PATH_SCALER}")

    # Simpan rekap hasil evaluasi
    df_eval = pd.DataFrame([{
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "epochs": 20,
        "alpha": 0.1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }])
    df_eval.to_csv(PATH_EVALUASI, index=False)
    print(f"Rekap evaluasi tersimpan ke: {PATH_EVALUASI}")
    print("="*70 + "\n")

    return model_w2v, model_nb, scaler


# ==============================================================================
# PEMUATAN MODEL MACHINE LEARNING (LOAD MODELS AT STARTUP)
# ==============================================================================
skipgram_model = None
naive_bayes_model = None
scaler_model = None
model_error = None

def muat_model():
    """
    Memuat model yang sudah tersimpan (.pkl). Jika file model belum ada,
    maka sistem akan secara otomatis memicu proses pelatihan model.
    """
    global skipgram_model, naive_bayes_model, scaler_model, model_error

    try:
        # Jika salah satu model belum ada, latih ulang otomatis
        if not (os.path.exists(PATH_SKIPGRAM_MODEL) and 
                os.path.exists(PATH_NAIVE_BAYES_MODEL) and 
                os.path.exists(PATH_SCALER)):
            print("File model tidak lengkap. Memulai pelatihan model otomatis...")
            skipgram_model, naive_bayes_model, scaler_model = latih_dan_simpan_model()
            return

        with open(PATH_SKIPGRAM_MODEL, "rb") as f:
            skipgram_model = pickle.load(f)

        with open(PATH_NAIVE_BAYES_MODEL, "rb") as f:
            naive_bayes_model = pickle.load(f)

        with open(PATH_SCALER, "rb") as f:
            scaler_model = pickle.load(f)

        print("Berhasil memuat model Skip-Gram, Naive Bayes, dan Scaler.")
    except Exception as e:
        model_error = str(e)
        print(f"Peringatan: Gagal memuat model: {model_error}")

# Muat model saat server aplikasi mulai
muat_model()


# ==============================================================================
# ROUTE WEB: HALAMAN UTAMA (INDEX UI)
# ==============================================================================
@app.route("/", methods=["GET"])
def index():
    """
    Menampilkan antarmuka web interaktif aplikasi klasifikasi berita.
    """
    return render_template("index.html")


# ==============================================================================
# ROUTE API: ENDPOINT PREDIKSI BERITA DARI LINK URL
# ==============================================================================
@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Endpoint API untuk menerima URL berita, menjalankan pipeline pemrosesan,
    dan mengembalikan hasil klasifikasi dalam format JSON.
    """
    if model_error:
        return jsonify({
            "status": "error",
            "message": f"Model machine learning gagal dimuat: {model_error}"
        }), 500

    # Mengambil input data dari JSON body atau Form Data
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "").strip()

    try:
        # Menjalankan Master Pipeline Klasifikasi Berita
        hasil_prediksi = pipeline_klasifikasi_berita(url)

        return jsonify({
            "status": "success",
            "data": hasil_prediksi
        })

    except ValueError as ve:
        # Kesalahan input URL atau ekstraksi teks
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400

    except requests.exceptions.Timeout:
        return jsonify({
            "status": "error",
            "message": "Waktu permintaan habis (Timeout). Server web berita tidak merespons dalam 15 detik."
        }), 504

    except requests.exceptions.RequestException as re_err:
        return jsonify({
            "status": "error",
            "message": f"Gagal mengunduh halaman web berita: {str(re_err)}"
        }), 502

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan pada sistem: {str(e)}"
        }), 500


# ==============================================================================
# ENTRY POINT RUNNER (LOCAL & CLI)
# ==============================================================================
if __name__ == "__main__":
    # Dukungan opsi CLI --train untuk melatih ulang model secara manual
    if "--train" in sys.argv:
        latih_dan_simpan_model()
    else:
        # Jalankan server lokal Flask
        port = int(os.environ.get("PORT", 5000))
        print(f"\nServer Flask berjalan di: http://127.0.0.1:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)