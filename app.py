import streamlit as st
import Orange
import pandas as pd
import pickle

# ==============================
# Judul dan Deskripsi
# ==============================
st.title("🎬 Prediksi Rating Anime Menggunakan Model Orange")
st.write("Masukkan data anime untuk memprediksi rating atau kategori popularitas berdasarkan model Random Forest dari Orange.")

# ==============================
# Load Model dari Orange
# ==============================
try:
    with open("model.pkcls", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat dari file 'model.pkcls'")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==============================
# Input dari Pengguna
# ==============================
st.subheader("📊 Masukkan Data Anime")

anime_name = st.text_input("Nama Anime", "Attack on Titan")
anime_type = st.selectbox("Tipe Anime", ["TV", "Movie", "OVA", "ONA", "Special"])
episodes = st.number_input("Jumlah Episode", 1, 2000, 12)
score = st.number_input("Skor (0.0 - 10.0)", 0.0, 10.0, 8.5)
members = st.number_input("Jumlah Member (popularitas)", 0, 10000000, 500000)
favorites = st.number_input("Jumlah Favorite", 0, 1000000, 20000)
rank = st.number_input("Peringkat (Rank)", 1, 20000, 500)
popularity = st.number_input("Posisi Popularitas", 1, 20000, 1000)
studio = st.text_input("Studio Produksi", "MAPPA")
genre = st.selectbox("Genre Utama", ["Action", "Romance", "Comedy", "Adventure", "Fantasy", "Drama", "Slice of Life"])

# ==============================
# Bentuk DataFrame Input
# ==============================
data_input = pd.DataFrame({
    'Name': [anime_name],
    'Type': [anime_type],
    'Episodes': [episodes],
    'Score': [score],
    'Members': [members],
    'Favorites': [favorites],
    'Rank': [rank],
    'Popularity': [popularity],
    'Studio': [studio],
    'Genre': [genre]
})

st.write("📄 Data yang akan diprediksi:")
st.dataframe(data_input)

# ==============================
# Konversi ke Format Orange
# ==============================
try:
    domain = model.domain
    features = [var.name for var in domain.attributes]
    row = [data_input.iloc[0].get(col, 0) for col in features]
    orange_data = Orange.data.Table(domain, [row])
except Exception as e:
    st.error(f"Gagal mengonversi data ke format Orange: {e}")
    st.stop()

# ==============================
# Prediksi
# ==============================
if st.button("🔮 Prediksi"):
    try:
        pred = model(orange_data)
        pred_value = str(pred[0])

        st.write("### 🎯 Hasil Prediksi:")
        st.success(f"Model memprediksi: **{pred_value}**")

        st.write("---")
        st.caption("Model ini menggunakan algoritma yang dilatih di Orange, misalnya Random Forest atau Decision Tree.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")