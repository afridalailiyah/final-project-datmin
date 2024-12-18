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
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Analisis Sentimen SpotifyWrapped 2024")

    # Bagian untuk upload file
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/rf_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/dhavinaocxa/fp-datmin/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Pastikan model dan vectorizer berhasil di-load
        if model and vectorizer:
            # Validasi kolom 'stemming_data'
            if 'stemming_data' in data.columns:
                # Transformasi data menggunakan vectorizer
                X_test = vectorizer.transform(data['stemming_data'])

                # Prediksi Sentimen
                if st.button("Prediksi Sentimen"):
                    # Prediksi dengan model yang sudah dilatih
                    predictions = model.predict(X_test)

                    # Simpan hasil prediksi di session_state
                    st.session_state['predictions'] = predictions
                    st.session_state['data'] = data
                    st.session_state['X_test'] = X_test

                    # Tambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions

                    # Simpan hasil prediksi di session_state
                    st.session_state['results'] = data

                    # Evaluasi Akurasi jika ada label 'sentiment'
                    if 'sentiment' in data.columns:
                        accuracy = accuracy_score(data['sentiment'], predictions)
                        report_dict = classification_report(data['sentiment'], predictions, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()  # Konversi ke DataFrame

                        st.session_state['accuracy'] = accuracy
                        st.session_state['report_df'] = report_df
                    else:
                        st.session_state['accuracy'] = None
                        st.session_state['report_df'] = None

                # Tampilkan hasil prediksi dan evaluasi jika tersedia di session_state
                if 'predictions' in st.session_state:
                    data = st.session_state['data']
                    predictions = st.session_state['predictions']
                    data['Predicted Sentiment'] = predictions

                    st.write("Hasil Prediksi Sentimen:")
                    st.write(data[['stemming_data', 'Predicted Sentiment']])

                    # Visualisasi distribusi sentimen
                    sentiment_counts = data['Predicted Sentiment'].value_counts()
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
                        st.success(f"Akurasi Model: {st.session_state['accuracy']:.2%}")
                        st.write("Laporan Klasifikasi:")
                        st.table(st.session_state['report_df'])  # Tampilkan laporan dalam bentuk tabel
                    else:
                        st.warning("Kolom 'sentiment' tidak ditemukan. Tidak dapat menghitung akurasi.")

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )

            else:
                st.error("Kolom 'stemming_data' tidak ditemukan dalam file yang diunggah.")

if __name__ == '__main__':
    main()
