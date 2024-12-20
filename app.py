import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL {url}")
        return None

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen Clash of Champions Ruangguru 👩‍💻✍")

    # Bagian untuk upload file
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.subheader("Data yang diunggah")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/afridalailiyah/final-project-datmin/main/rf_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/afridalailiyah/final-project-datmin/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Pastikan model dan vectorizer berhasil di-load
        if model and vectorizer:
            # Validasi kolom 'stemming'
            if 'stemming' in data.columns:
                # Cek apakah ada nilai kosong atau teks kosong di kolom 'stemming'
                if data['stemming'].isnull().any() or data['stemming'].str.strip().eq("").any():
                    st.warning("Terdapat baris dengan data kosong di kolom 'stemming'. Baris ini akan diberi keterangan 'Prediksi sentimen tidak ada'.")

                # Transformasi data menggunakan vectorizer
                data['stemming'] = data['stemming'].fillna("").replace(r"^\s*$", "NA", regex=True)

                if st.button("Prediksi Sentimen"):
                    # Prediksi Sentimen
                    predictions = []
                    for text in data['stemming']:
                        if text == "NA":
                            predictions.append("Prediksi sentimen tidak ada")
                        else:
                            transformed_text = vectorizer.transform([text])
                            predictions.append(model.predict(transformed_text)[0])

                    # Tambahkan hasil prediksi ke data
                    data['predicted sentiment'] = predictions
                    st.session_state['results'] = data

                    # Evaluasi Akurasi jika ada label 'sentiment'
                    if 'sentiment' in data.columns:
                        valid_rows = data[data['predicted sentiment'] != "Prediksi sentimen tidak ada"]
                        if not valid_rows.empty:
                            accuracy = accuracy_score(valid_rows['sentiment'], valid_rows['predicted sentiment'])
                            report_dict = classification_report(valid_rows['sentiment'], valid_rows['predicted sentiment'], output_dict=True)
                            report_df = pd.DataFrame(report_dict).transpose()
                            st.session_state['accuracy'] = accuracy
                            st.session_state['report_df'] = report_df
                        else:
                            st.warning("Tidak ada data valid untuk evaluasi akurasi.")
                            st.session_state['accuracy'] = None
                            st.session_state['report_df'] = None
                    else:
                        st.session_state['accuracy'] = None
                        st.session_state['report_df'] = None

                    # Tampilkan hasil prediksi
                    st.subheader("Hasil Prediksi Sentimen")
                    st.write(data[['stemming', 'predicted sentiment']])

                    # Visualisasi distribusi sentimen (hanya untuk prediksi yang valid)
                    valid_predictions = data[data['predicted sentiment'] != "Prediksi sentimen tidak ada"]
                    if not valid_predictions.empty:
                        sentiment_counts = valid_predictions['predicted sentiment'].value_counts()
                        fig_bar = px.bar(
                            sentiment_counts,
                            x=sentiment_counts.index,
                            y=sentiment_counts.values,
                            labels={'x': 'Sentimen', 'y': 'Jumlah'},
                            title="Distribusi Sentimen"
                        )
                        st.plotly_chart(fig_bar)

                    # Evaluasi akurasi jika tersedia
                    if st.session_state['accuracy'] is not None:
                        st.success(f"Akurasi Model {st.session_state['accuracy']:.2%}")
                        st.subheader("Hasil Klasifikasi")
                        st.table(st.session_state['report_df'])  # Tampilkan laporan dalam bentuk tabel
                    else:
                        st.warning("Tidak ada data valid untuk menghitung akurasi.")

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

            else:
                st.error("Kolom 'stemming' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
