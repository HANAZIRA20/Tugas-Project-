import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="🎬 Anime Rating Prediction", page_icon="🎬")

st.title("🎬 Anime Rating Prediction App")
st.write("Prediksi rating anime berdasarkan data seperti skor, popularitas, dan sumber.")

# =======================
# Upload dataset
# =======================
uploaded_file = st.file_uploader("Upload dataset CSV (contoh: Top_Anime_data.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil diunggah!")
    st.dataframe(df.head())

    # Pilih kolom target
    target_col = st.selectbox("🎯 Pilih kolom target (rating)", df.columns, index=len(df.columns)-1)

    # Hapus NA pada target
    df = df.dropna(subset=[target_col])

    # Encode target label
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Tentukan kolom numerik dan kategorikal yang masuk akal
    num_cols = ['Score', 'Popularity', 'Rank', 'Members']
    cat_cols = [c for c in X.columns if c not in num_cols and c not in ['Description', 'Producers', 'Licensors', 'Aired', 'Broadcast']]

    # Encode kategorikal
    X_encoded = pd.get_dummies(X[num_cols + cat_cols])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    st.subheader("📊 Evaluasi Model")
    st.write(f"**Akurasi Model:** {acc:.2f}")
    st.text("Classification Report:\n" + report)

    # Simpan sesi
    st.session_state["model"] = model
    st.session_state["encoder"] = le
    st.session_state["features"] = X_encoded.columns
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols
    st.session_state["raw_df"] = df

# =======================
# Prediksi Baru
# =======================
if "model" in st.session_state:
    st.markdown("---")
    st.header("🔮 Coba Prediksi Rating Baru")

    anime_name = st.text_input("🎬 Nama Anime", placeholder="Contoh: Attack on Titan")

    # Input numerik
    st.subheader("📈 Fitur Numerik")
    input_data = {}
    for col in st.session_state["num_cols"]:
        input_data[col] = st.number_input(f"{col}", value=0.0)

    # Input kategorikal
    st.subheader("🏷️ Fitur Kategorikal")
    for col in st.session_state["cat_cols"]:
        options = st.session_state["raw_df"][col].dropna().unique().tolist()
        if len(options) > 50:
            continue  # lewati kolom dengan terlalu banyak opsi
        input_data[col] = st.selectbox(f"{col}", options)

    if st.button("🎯 Prediksi Rating"):
        model = st.session_state["model"]
        le = st.session_state["encoder"]
        all_cols = st.session_state["features"]

        # Buat DataFrame input
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)

        # Tambah kolom yang hilang
        for col in all_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[all_cols]

        # Prediksi
        pred = model.predict(input_encoded)[0]
        pred_label = le.inverse_transform([pred])[0]

        st.success(f"🎬 Anime: **{anime_name if anime_name else 'Tanpa Nama'}**")
        st.markdown(f"📢 **Prediksi Rating:** `{pred_label}`")
        st.balloons()
