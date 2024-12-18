import streamlit as st
import pandas as pd
from nltk.tag import CRFTagger
from textblob import TextBlob
import plotly.express as px
import urllib.request  # Untuk mengunduh file

# Unduh model CRF jika belum ada
MODEL_URL = "https://raw.githubusercontent.com/dhavinaocxa/latihan-datmin/main/all_indo_man_tag_corpus_model.crf.tagger"
MODEL_PATH = "all_indo_man_tag_corpus_model.crf.tagger"

try:
    with open(MODEL_PATH, "r") as f:
        pass
except FileNotFoundError:
    st.write("Mengunduh model CRF...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Judul aplikasi
st.title("Analisis Sentimen")

# Upload file
uploaded_file = st.file_uploader("Upload dataset CSV Anda", type=["csv"])
if uploaded_file is not None:
    try:
        # Baca file CSV
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("Dataset asli:")
        st.dataframe(data.head())

        # Asumsi kolom tweet bernama 'tweet'
        if 'tweet' in data.columns:
            tweets = data['tweet']

            # POS Tagging
            st.write("Proses POS Tagging...")
            ct = CRFTagger()
            ct.set_model_file(MODEL_PATH)

            # Tagging setiap tweet
            tagged_tweets = [ct.tag_sents([tweet.split()])[0] for tweet in tweets]

            # Tambahkan hasil POS tagging ke dataset
            data['pos_tagging'] = tagged_tweets

            # Filter hanya nouns
            filtered_tweets = []
            for tagged in tagged_tweets:
                nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
                filtered_tweets.append(" ".join(nouns))

            data['filtered_nouns'] = filtered_tweets

            # Analisis Sentimen menggunakan TextBlob
            st.write("Proses Analisis Sentimen...")
            sentiments = []
            for tweet in data['filtered_nouns']:
                analysis = TextBlob(tweet)
                if analysis.sentiment.polarity > 0:
                    sentiments.append("Positive")
                elif analysis.sentiment.polarity < 0:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")

            # Tambahkan hasil sentimen ke dataset
            data['sentiment'] = sentiments


            # Tampilkan dataset hasil tagging dan sentimen
            st.write("Dataset dengan hasil POS tagging dan analisis sentimen:")
            st.dataframe(data[['tweet', 'pos_tagging', 'filtered_nouns', 'sentiment']])


            # Distribusi hasil sentimen
            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            
            # Membuat chart interaktif
            fig = px.bar(
                sentiment_counts,
                x='Sentimen',
                y='Jumlah',
                color='Sentimen',
                title='Distribusi Sentimen dari Tweet',
                labels={'Jumlah': 'Jumlah Tweet'},
                text='Jumlah',
                color_discrete_map={'positif': 'green', 'negatif': 'red', 'netral': 'blue'}  # Warna khusus
            )
            
            # Menampilkan chart di Streamlit
            st.plotly_chart(fig)

            # Download dataset hasil
            csv = data.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="Download Dataset Hasil",
                data=csv,
                file_name="sentiment_analysis_tweets.csv",
                mime="text/csv"
            )
        else:
            st.warning("Dataset Anda harus memiliki kolom 'tweet'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
