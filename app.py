import streamlit as st
import pickle
import re
import requests

from bs4 import BeautifulSoup
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)


# ============================================================
# KONFIGURASI STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    # Memuat model Skip-Gram
    with open("skipgram_model.pkl", "rb") as file:
        skipgram_model = pickle.load(file)

    # Memuat model Naive Bayes
    with open("naive_bayes_skipgram.pkl", "rb") as file:
        naive_bayes_model = pickle.load(file)

    # Memuat scaler
    with open("scaler_skipgram.pkl", "rb") as file:
        scaler = pickle.load(file)

    return (
        skipgram_model,
        naive_bayes_model,
        scaler
    )


# Mencoba memuat model
try:

    (
        skipgram_model,
        naive_bayes_model,
        scaler
    ) = load_models()

except Exception as e:

    st.error("❌ Model gagal dimuat.")

    st.code(
        str(e),
        language="text"
    )

    st.warning(
        "Pastikan file skipgram_model.pkl, "
        "naive_bayes_skipgram.pkl, dan scaler_skipgram.pkl "
        "sudah dibuat menggunakan environment Python yang sama."
    )

    st.stop()


# ============================================================
# LOAD PREPROCESSING
# ============================================================

# Stopword bahasa Indonesia
stopword_factory = StopWordRemoverFactory()
stopword_remover = (
    stopword_factory.create_stop_word_remover()
)

# Stemmer bahasa Indonesia
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# ============================================================
# FUNGSI PREPROCESSING
# ============================================================

def preprocessing(text):

    # Memastikan input berupa string
    text = str(text)

    # Case folding
    text = text.lower()

    # Menghapus tanda baca
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Menghapus angka
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Menghapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Stopword removal
    text = stopword_remover.remove(text)

    # Stemming
    text = stemmer.stem(text)

    # Tokenisasi
    tokens = text.split()

    return tokens


# ============================================================
# FUNGSI MENGAMBIL ISI BERITA DARI URL
# ============================================================

def get_news_content(url):

    # User-Agent agar request menyerupai browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        )
    }

    # Mengambil halaman website
    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    # Memastikan request berhasil
    response.raise_for_status()

    # Membaca HTML
    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Menghapus elemen HTML yang bukan isi utama
    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript"
    ]):

        tag.decompose()

    # Mengambil seluruh paragraf
    paragraphs = soup.find_all("p")

    contents = []

    for paragraph in paragraphs:

        text = paragraph.get_text(
            " ",
            strip=True
        )

        if text:

            contents.append(text)

    # Menggabungkan seluruh paragraf
    article_text = " ".join(contents)

    # Membersihkan spasi
    article_text = re.sub(
        r"\s+",
        " ",
        article_text
    ).strip()

    return article_text


# ============================================================
# FUNGSI DOCUMENT VECTOR
# ============================================================

def document_vector(model, tokens):

    vectors = []

    # Mengambil vector dari setiap kata
    # yang terdapat pada vocabulary Skip-Gram
    for word in tokens:

        if word in model.wv:

            vectors.append(
                model.wv[word]
            )

    # Jika tidak ada kata yang dikenali
    if len(vectors) == 0:

        return [0.0] * model.vector_size

    # Menghasilkan document vector
    # dengan menghitung rata-rata word vector
    return sum(vectors) / len(vectors)


# ============================================================
# CEK KATEGORI BERDASARKAN URL
# ============================================================

def cek_kategori_url(url):

    # Mengubah URL menjadi lowercase
    url_lower = url.lower()

    # Keyword Finance
    finance_keywords = [
        "finance",
        "finansial",
        "bisnis",
        "business",
        "market",
        "ekonomi"
    ]

    # Keyword Sport
    sport_keywords = [
        "sport",
        "sports",
        "olahraga",
        "sepakbola",
        "bola",
        "football",
        "soccer"
    ]

    # Keyword kategori lain
    other_keywords = [
        "health",
        "kesehatan",
        "fotohealth",
        "lifestyle",
        "gaya-hidup",
        "travel",
        "wisata",
        "food",
        "kuliner",
        "entertainment",
        "hiburan",
        "technology",
        "teknologi",
        "tekno",
        "otomotif",
        "detikhot",
        "wolipop",
        "inet"
    ]

    # Cek Finance
    for keyword in finance_keywords:

        if keyword in url_lower:

            return True, "finance"

    # Cek Sport
    for keyword in sport_keywords:

        if keyword in url_lower:

            return True, "sport"

    # Cek kategori lain
    for keyword in other_keywords:

        if keyword in url_lower:

            return False, "other"

    # Jika tidak ditemukan
    return True, "unknown"


# ============================================================
# NORMALISASI LABEL
# ============================================================

def normalisasi_label(label):

    label = str(label).lower().strip()

    # Finance
    if label in [
        "finance",
        "finansial",
        "bisnis",
        "business",
        "ekonomi"
    ]:

        return "finance"

    # Sport
    if label in [
        "sport",
        "sports",
        "olahraga"
    ]:

        return "sport"

    # Selain itu
    return "other"


# ============================================================
# TAMPILAN UTAMA
# ============================================================

st.title(
    "📰 Klasifikasi Berita Menggunakan Skip-Gram + Naive Bayes"
)

st.write(
    """
    Aplikasi ini menggunakan **Skip-Gram** untuk menghasilkan
    representasi numerik dari teks berita dan **Naive Bayes**
    untuk melakukan klasifikasi berita.
    """
)

st.info(
    """
    **Kategori model:** Finance dan Sport.

    Berita yang berada di luar kategori tersebut dapat ditampilkan
    sebagai **Other** berdasarkan aturan aplikasi.
    """
)


# ============================================================
# INPUT URL
# ============================================================

st.subheader(
    "🔗 Masukkan URL Berita"
)

url = st.text_input(
    "URL berita:",
    placeholder="https://news.detik.com/..."
)


# ============================================================
# PROSES KLASIFIKASI
# ============================================================

if st.button(
    "🔍 Klasifikasikan Berita",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDASI URL
    # --------------------------------------------------------

    if not url.strip():

        st.warning(
            "⚠️ Silakan masukkan URL berita terlebih dahulu."
        )

        st.stop()

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):

        st.error(
            "❌ URL harus diawali dengan http:// atau https://"
        )

        st.stop()


    # --------------------------------------------------------
    # CEK KATEGORI URL
    # --------------------------------------------------------

    url_valid, url_category = cek_kategori_url(url)


    # --------------------------------------------------------
    # MENGAMBIL ISI BERITA
    # --------------------------------------------------------

    with st.spinner(
        "⏳ Mengambil isi berita..."
    ):

        try:

            article_text = get_news_content(url)

        except requests.exceptions.Timeout:

            st.error(
                "❌ Waktu pengambilan berita habis (timeout)."
            )

            st.stop()

        except requests.exceptions.RequestException as e:

            st.error(
                "❌ Gagal mengambil berita dari URL."
            )

            st.code(
                str(e),
                language="text"
            )

            st.stop()

        except Exception as e:

            st.error(
                "❌ Terjadi kesalahan saat mengambil berita."
            )

            st.code(
                str(e),
                language="text"
            )

            st.stop()


    # --------------------------------------------------------
    # VALIDASI ISI BERITA
    # --------------------------------------------------------

    if not article_text:

        st.error(
            "❌ Isi berita tidak berhasil ditemukan."
        )

        st.info(
            """
            Website mungkin menggunakan struktur HTML yang berbeda,
            membutuhkan JavaScript, atau membatasi akses scraping.
            """
        )

        st.stop()


    # --------------------------------------------------------
    # TAMPILKAN ISI BERITA
    # --------------------------------------------------------

    st.success(
        "✅ Isi berita berhasil diambil."
    )

    with st.expander(
        "📄 Lihat Isi Berita"
    ):

        st.write(article_text)


    # ========================================================
    # PREPROCESSING
    # ========================================================

    with st.spinner(
        "⏳ Melakukan preprocessing..."
    ):

        tokens = preprocessing(
            article_text
        )


    # Memastikan token tidak kosong
    if len(tokens) == 0:

        st.error(
            "❌ Tidak ada token yang dapat digunakan "
            "setelah preprocessing."
        )

        st.stop()


    # ========================================================
    # HASIL PREPROCESSING
    # ========================================================

    with st.expander(
        "🔤 Lihat Hasil Preprocessing"
    ):

        st.write(
            "Jumlah token:",
            len(tokens)
        )

        st.write(
            "Token:"
        )

        st.write(
            tokens
        )


    # ========================================================
    # SKIP-GRAM DOCUMENT VECTOR
    # ========================================================

    with st.spinner(
        "⏳ Membuat representasi Skip-Gram..."
    ):

        vector = document_vector(
            skipgram_model,
            tokens
        )


    # Mengubah vector menjadi bentuk 2 dimensi
    vector = [vector]


    # ========================================================
    # SCALING
    # ========================================================

    try:

        vector_scaled = scaler.transform(
            vector
        )

    except Exception as e:

        st.error(
            "❌ Terjadi masalah saat melakukan scaling."
        )

        st.code(
            str(e),
            language="text"
        )

        st.stop()


    # ========================================================
    # NAIVE BAYES PREDICTION
    # ========================================================

    try:

        # Prediksi dari Naive Bayes
        prediction = naive_bayes_model.predict(
            vector_scaled
        )[0]

        # Probabilitas setiap kelas
        probabilities = naive_bayes_model.predict_proba(
            vector_scaled
        )[0]

        # Nama kelas
        classes = naive_bayes_model.classes_

    except Exception as e:

        st.error(
            "❌ Terjadi masalah saat melakukan klasifikasi."
        )

        st.code(
            str(e),
            language="text"
        )

        st.stop()


    # ========================================================
    # MENGAMBIL PROBABILITAS TERTINGGI
    # ========================================================

    # Posisi probabilitas terbesar
    max_probability_index = probabilities.argmax()

    # Nilai probabilitas terbesar
    max_probability = probabilities[
        max_probability_index
    ]

    # Label dengan probabilitas terbesar
    predicted_label = classes[
        max_probability_index
    ]

    # Normalisasi label
    predicted_label = normalisasi_label(
        predicted_label
    )


    # ========================================================
    # THRESHOLD CONFIDENCE
    # ========================================================

    # Threshold confidence
    threshold = 0.60

    # Secara default menggunakan hasil model
    final_prediction = predicted_label

    # Jika URL masuk kategori lain
    if url_category == "other":

        final_prediction = "other"

    # Jika confidence di bawah 60%
    elif max_probability < threshold:

        final_prediction = "other"


    # ========================================================
    # HASIL KLASIFIKASI
    # ========================================================

    st.subheader(
        "📊 Hasil Klasifikasi"
    )


    # --------------------------------------------------------
    # HASIL OTHER
    # --------------------------------------------------------

    if final_prediction == "other":

        st.warning(
            "📌 Kategori Berita: OTHER"
        )

        if url_category == "other":

            st.info(
                """
                URL berita terindikasi berasal dari kategori
                di luar Finance dan Sport.
                """
            )

        elif max_probability < threshold:

            st.info(
                f"""
                Confidence model sebesar
                **{max_probability * 100:.2f}%**,
                sehingga hasil dikategorikan sebagai Other
                karena berada di bawah threshold
                **{threshold * 100:.0f}%**.
                """
            )

        else:

            st.info(
                """
                Model menghasilkan kategori yang tidak termasuk
                kelas Finance atau Sport.
                """
            )


    # --------------------------------------------------------
    # HASIL FINANCE / SPORT
    # --------------------------------------------------------

    else:

        if final_prediction == "finance":

            st.success(
                "💰 Kategori Berita: FINANCE"
            )

        elif final_prediction == "sport":

            st.success(
                "⚽ Kategori Berita: SPORT"
            )

        else:

            st.info(
                f"Kategori Berita: {final_prediction.upper()}"
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.metric(
            "Confidence Model",
            f"{max_probability * 100:.2f}%"
        )


        # ----------------------------------------------------
        # PROBABILITAS SETIAP KELAS
        # ----------------------------------------------------

        st.subheader(
            "📈 Probabilitas Setiap Kelas"
        )

        probability_data = {}

        # Menggabungkan kelas dengan probabilitas
        for class_name, probability in zip(
            classes,
            probabilities
        ):

            normalized_class = normalisasi_label(
                class_name
            )

            probability_data[
                normalized_class
            ] = probability


        # Menampilkan probabilitas
        for class_name, probability in (
            probability_data.items()
        ):

            st.write(
                f"**{class_name.upper()}**: "
                f"{probability * 100:.2f}%"
            )

            # Progress bar
            st.progress(
                float(probability)
            )


# ============================================================
# SELESAI
# ============================================================