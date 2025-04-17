import streamlit as st
import json
import numpy as np
import joblib
import os

# Fungsi untuk memuat file CSS
def load_css(file_name):
    if os.path.exists(file_name):  # Mengecek apakah file ada
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"⚠️ File CSS '{file_name}' tidak ditemukan.")

# Fungsi untuk memuat pertanyaan dari file JSON
def load_questions(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"⚠️ File '{file_name}' tidak ditemukan.")
        return []

# Fungsi untuk memuat model machine learning
def load_model(file_name):
    try:
        with open(file_name, 'rb') as f:
            return joblib.load(f)
    except FileNotFoundError:
        st.error(f"⚠️ Model '{file_name}' tidak ditemukan.")
        return None

# Fungsi untuk melakukan prediksi
def predict_major(responses, model):
    input_data = np.array(responses).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Memuat CSS
load_css('tampilan/style.css')

# Inisialisasi session state untuk halaman
if "page" not in st.session_state:
    st.session_state["page"] = "Beranda"

# Header Navigasi
st.markdown(
    """
    <style>
        .nav {
            display: flex;
            justify-content: center;
            gap: 20px;
            background: #3498db;
            padding: 12px;
            border-radius: 10px;
            flex-wrap: wrap;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .nav a {
            color: white;
            font-size: 18px;
            font-weight: bold;
            text-decoration: none;
            padding: 10px 15px;
            border-radius: 5px;
        }
        .nav a:hover {
            background: #2980b9;
        }
    </style>
    <div class="nav">
        <a href="?page=Beranda">Beranda</a>
        <a href="?page=Tes Minat">Tes Minat</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cek halaman yang aktif berdasarkan URL parameter
query_params = st.query_params
if "page" in query_params:
    st.session_state["page"] = query_params["page"]

# Tampilan Halaman Beranda
if st.session_state["page"] == "Beranda":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #2c3e50;">🔍 Tes Minat dan Bakat</h1>
            <p style="color: #7f8c8d; font-size: 18px;">
                Selamat datang di tes minat dan bakat. Temukan jurusan yang sesuai dengan kepribadian dan minat Anda.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Menampilkan gambar hero
    st.image("tampilan/p.png", use_column_width=True, output_format="PNG")

    # Tombol untuk mulai tes
    if st.button("Mulai Tes Sekarang"):
        st.session_state["page"] = "Tes Minat"
        st.rerun()

# Tampilan Halaman Tes Minat
elif st.session_state["page"] == "Tes Minat":
    st.title("Tes Minat")
    questions = load_questions('questions.json')

    # Inisialisasi sesi state untuk menyimpan jawaban
    if 'responses' not in st.session_state:
        st.session_state.responses = [None] * len(questions)

    # Menampilkan pertanyaan
    for i, question in enumerate(questions):
        st.session_state.responses[i] = st.radio(f"**{question}**", ["Ya", "Tidak"], key=f"q{i}")

    # Tombol untuk melihat hasil
    if st.button("Lihat Hasil"):
        if None in st.session_state.responses:
            st.warning("Harap jawab semua pertanyaan.")
        else:
            responses = [1 if ans == "Ya" else 0 for ans in st.session_state.responses]

            # Memuat dua model
            model_sma = load_model('rf_sma_model.pkl')
            model_s1 = load_model('rf_s1_model.pkl')

            # Mengecek apakah model tersedia
            if model_sma and model_s1:
                result_sma = predict_major(responses, model_sma)
                result_s1 = predict_major(responses, model_s1)

                # Menampilkan hasil
                st.success(f"🏫 **Rekomendasi Jurusan SMA/SMK:** **{result_sma}**")
                st.success(f"🎓 **Rekomendasi Program Studi (S1):** **{result_s1}**")
