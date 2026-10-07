# ============================================================
# MODEL KLASIFIKASI BERITA
# SKIP-GRAM + NAIVE BAYES
# DENGAN GRID SEARCH HYPERPARAMETER
# ============================================================

import pandas as pd
import numpy as np
import re
import pickle

# Sastrawi untuk preprocessing bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)

# Scikit-learn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Gensim untuk Skip-Gram
from gensim.models import Word2Vec


# ============================================================
# 1. MEMBACA DATASET
# ============================================================

# Dataset menggunakan delimiter koma (,)
df = pd.read_csv("dataset_detik_200_berita.csv")

print("Ukuran dataset:", df.shape)
print("Kolom dataset:", df.columns.tolist())

# Pastikan kolom yang dibutuhkan tersedia
required_columns = ["isi_berita", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Kolom '{column}' tidak ditemukan dalam dataset."
        )

# Menghapus data kosong
df = df.dropna(subset=["isi_berita", "label"]).copy()

print("Jumlah data setelah menghapus data kosong:", len(df))


# ============================================================
# 2. INISIALISASI STOPWORD DAN STEMMER
# ============================================================

# Stopword bahasa Indonesia
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Stemmer bahasa Indonesia
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# ============================================================
# 3. FUNGSI PREPROCESSING
# ============================================================

def preprocessing(text):
    """
    Melakukan preprocessing teks:
    1. Case folding
    2. Cleaning
    3. Menghapus angka
    4. Menghapus stopword
    5. Stemming
    6. Tokenisasi
    """

    # Pastikan data berupa string
    text = str(text)

    # --------------------------------------------------------
    # Case Folding
    # Mengubah seluruh teks menjadi huruf kecil
    # --------------------------------------------------------
    text = text.lower()

    # --------------------------------------------------------
    # Cleaning
    # Menghapus tanda baca dan karakter khusus
    # --------------------------------------------------------
    text = re.sub(r"[^\w\s]", " ", text)

    # --------------------------------------------------------
    # Menghapus angka
    # --------------------------------------------------------
    text = re.sub(r"\d+", " ", text)

    # --------------------------------------------------------
    # Normalisasi spasi
    # --------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    # --------------------------------------------------------
    # Stopword Removal
    # Menghapus kata umum seperti:
    # yang, dan, di, ke, dari, adalah, dll.
    # --------------------------------------------------------
    text = stopword_remover.remove(text)

    # --------------------------------------------------------
    # Stemming
    # Mengubah kata menjadi bentuk dasarnya
    # contoh:
    # bermain -> main
    # berlari -> lari
    # makanan -> makan
    # --------------------------------------------------------
    text = stemmer.stem(text)

    # --------------------------------------------------------
    # Tokenisasi
    # Memecah teks menjadi kumpulan kata
    # --------------------------------------------------------
    tokens = text.split()

    return tokens


# ============================================================
# 4. PREPROCESSING SELURUH DATASET
# ============================================================

print("\nMelakukan preprocessing dataset...")

df["tokens_stemmed"] = df["isi_berita"].apply(preprocessing)

print("Preprocessing selesai.")

# Menampilkan contoh hasil preprocessing
print("\nContoh preprocessing:")
print(df[["isi_berita", "tokens_stemmed", "label"]].head())


# ============================================================
# 5. MENENTUKAN FITUR DAN LABEL
# ============================================================

# X = teks yang sudah diproses
X = df["tokens_stemmed"]

# y = kategori berita
y = df["label"]


# ============================================================
# 6. MEMBAGI DATA TRAINING DAN TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPembagian dataset:")
print("Data training:", len(X_train))
print("Data testing :", len(X_test))


# ============================================================
# 7. FUNGSI MENGUBAH DOKUMEN MENJADI VEKTOR
# ============================================================

def document_vector(model, tokens):
    """
    Mengubah sekumpulan token menjadi satu vektor dokumen.

    Caranya:
    - Ambil vector setiap kata yang terdapat di vocabulary Skip-Gram
    - Kemudian hitung rata-rata vector tersebut
    """

    vectors = []

    for word in tokens:

        # Hanya mengambil kata yang terdapat
        # dalam vocabulary Skip-Gram
        if word in model.wv:

            vectors.append(model.wv[word])

    # Jika tidak ada kata yang dikenal
    if len(vectors) == 0:

        return np.zeros(model.vector_size)

    # Menghitung rata-rata vector semua kata
    return np.mean(vectors, axis=0)


# ============================================================
# 8. FUNGSI TRAINING SKIP-GRAM
# ============================================================

def train_skipgram(
    X_train,
    X_test,
    vector_size,
    window,
    min_count,
    epochs
):
    """
    Melatih model Word2Vec menggunakan metode Skip-Gram.
    """

    # Membuat model Word2Vec
    model = Word2Vec(
        sentences=X_train.tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,

        # sg=1 berarti menggunakan Skip-Gram
        # sg=0 berarti CBOW
        sg=1,

        epochs=epochs,

        # Agar hasil dapat direproduksi
        seed=42
    )

    # --------------------------------------------------------
    # Mengubah setiap dokumen training menjadi vector
    # --------------------------------------------------------

    X_train_vectors = np.array([
        document_vector(model, tokens)
        for tokens in X_train
    ])

    # --------------------------------------------------------
    # Mengubah setiap dokumen testing menjadi vector
    # --------------------------------------------------------

    X_test_vectors = np.array([
        document_vector(model, tokens)
        for tokens in X_test
    ])

    return model, X_train_vectors, X_test_vectors


# ============================================================
# 9. GRID SEARCH HYPERPARAMETER
# ============================================================

# Parameter yang akan dicoba
param_grid = {

    # Ukuran vector Skip-Gram
    "vector_size": [100],

    # Jumlah kata di sekitar kata target
    "window": [5],

    # Minimum jumlah kemunculan kata
    "min_count": [1],

    # Jumlah epoch training Word2Vec
    "epochs": [20],

    # Parameter alpha untuk Naive Bayes
    "alpha": [0.1]
}


# Membuat semua kombinasi parameter
grid = list(ParameterGrid(param_grid))

print("\n============================================================")
print("GRID SEARCH")
print("============================================================")
print("Jumlah kombinasi parameter:", len(grid))


# ============================================================
# 10. MENYIAPKAN VARIABEL HASIL
# ============================================================

results = []

best_f1 = -1

best_params = None
best_skipgram_model = None
best_naive_bayes_model = None
best_scaler = None


# ============================================================
# 11. PROSES GRID SEARCH
# ============================================================

for i, params in enumerate(grid, start=1):

    print("\n------------------------------------------------------------")
    print(f"Percobaan {i}/{len(grid)}")
    print("Parameter:", params)
    print("------------------------------------------------------------")

    # --------------------------------------------------------
    # Training Skip-Gram
    # --------------------------------------------------------

    skipgram_model, X_train_vectors, X_test_vectors = train_skipgram(
        X_train,
        X_test,
        vector_size=params["vector_size"],
        window=params["window"],
        min_count=params["min_count"],
        epochs=params["epochs"]
    )

    # --------------------------------------------------------
    # Scaling
    #
    # MultinomialNB membutuhkan fitur non-negatif.
    # Oleh karena itu vector Word2Vec dinormalisasi
    # menggunakan MinMaxScaler.
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_vectors
    )

    X_test_scaled = scaler.transform(
        X_test_vectors
    )

    # --------------------------------------------------------
    # Membuat model Naive Bayes
    # --------------------------------------------------------

    naive_bayes_model = MultinomialNB(
        alpha=params["alpha"]
    )

    # Training Naive Bayes
    naive_bayes_model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Prediksi data testing
    # --------------------------------------------------------

    y_pred = naive_bayes_model.predict(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Menghitung evaluasi
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Menampilkan hasil
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # --------------------------------------------------------
    # Menyimpan hasil percobaan
    # --------------------------------------------------------

    result = {
        "vector_size": params["vector_size"],
        "window": params["window"],
        "min_count": params["min_count"],
        "epochs": params["epochs"],
        "alpha": params["alpha"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    results.append(result)

    # --------------------------------------------------------
    # Mengecek apakah model ini merupakan model terbaik
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_params = params.copy()

        best_skipgram_model = skipgram_model

        best_naive_bayes_model = naive_bayes_model

        best_scaler = scaler


# ============================================================
# 12. MENAMPILKAN MODEL TERBAIK
# ============================================================

print("\n============================================================")
print("MODEL TERBAIK")
print("============================================================")

print("Parameter terbaik:")
print(best_params)

print(f"\nBest F1-Score: {best_f1:.4f}")


# ============================================================
# 13. MENYIMPAN HASIL GRID SEARCH
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)

results_df.to_csv(
    "hasil_grid_search.csv",
    index=False
)

print("\nHasil Grid Search disimpan sebagai:")
print("hasil_grid_search.csv")


# ============================================================
# 14. MENYIMPAN MODEL SKIP-GRAM
# ============================================================

with open(
    "skipgram_model.pkl",
    "wb"
) as file:

    pickle.dump(
        best_skipgram_model,
        file,
        protocol=4
    )

print("Model Skip-Gram disimpan sebagai:")
print("skipgram_model.pkl")


# ============================================================
# 15. MENYIMPAN MODEL NAIVE BAYES
# ============================================================

with open(
    "naive_bayes_skipgram.pkl",
    "wb"
) as file:

    pickle.dump(
        best_naive_bayes_model,
        file,
        protocol=4
    )

print("Model Naive Bayes disimpan sebagai:")
print("naive_bayes_skipgram.pkl")


# ============================================================
# 16. MENYIMPAN SCALER
# ============================================================

with open(
    "scaler_skipgram.pkl",
    "wb"
) as file:

    pickle.dump(
        best_scaler,
        file,
        protocol=4
    )

print("Scaler disimpan sebagai:")
print("scaler_skipgram.pkl")


# ============================================================
# 17. SELESAI
# ============================================================

print("\n============================================================")
print("PROSES SELESAI")
print("============================================================")

print("File yang berhasil dibuat:")

print("1. skipgram_model.pkl")
print("2. naive_bayes_skipgram.pkl")
print("3. scaler_skipgram.pkl")
print("4. hasil_grid_search.csv")

print("\nModel siap digunakan oleh app.py.")