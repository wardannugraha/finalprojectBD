import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Sentiment Analysis - Bohemian Rhapsody Comments", layout="wide")

st.title("🎵 Sentiment Analysis of Bohemian Rhapsody (Queen) YouTube Comments")
st.markdown("This project analyzes YouTube comments using NLP and data visualization techniques.")

# Load data with caching
@st.cache_data
def load_data():
    dtype_fix = {
        "author": str,
        "text": str,
        "clean_text": str,
        "sentiment": str
    }

    df = pd.read_csv("cleaned_bohemian_comments.csv", dtype=dtype_fix)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['date'] = df['publishedAt'].dt.date
    df['clean_text'] = df['clean_text'].fillna("")
    return df

df = load_data()

# Show sample data (drop 'author' column to avoid display issues)
st.subheader("📄 Sample Data")
st.write(df.drop(columns=["author"]).head())

# Sentiment distribution visualization
st.subheader("📊 Sentiment Distribution")
sentiment_count = df['sentiment'].value_counts()

fig1, ax1 = plt.subplots()
sns.barplot(x=sentiment_count.index, y=sentiment_count.values, hue=sentiment_count.index, palette='pastel', ax=ax1, legend=False)
ax1.set_title("Number of Comments per Sentiment")
ax1.set_ylabel("Comment Count")
st.pyplot(fig1)

# WordCloud by sentiment
st.subheader("☁️ WordCloud by Sentiment")
sentiment_option = st.selectbox("Choose sentiment for WordCloud:", options=sentiment_count.index.tolist())

# Remove NaN and concatenate text
text_wc = " ".join(df[df['sentiment'] == sentiment_option]['clean_text'].dropna().astype(str))

if text_wc.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_wc)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wc, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)
else:
    st.info("No text data available for WordCloud.")

# Pie chart for sentiment composition
st.subheader("📈 Sentiment Composition (Pie Chart)")
fig3, ax3 = plt.subplots()
ax3.pie(sentiment_count.values, labels=sentiment_count.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
ax3.axis('equal')
st.pyplot(fig3)

# Top liked comments
st.subheader("🌟 Top Liked Comments")
top_comments = df[['text', 'likeCount', 'sentiment']].sort_values(by='likeCount', ascending=False).head(10)
st.write(top_comments)

# Time-based sentiment trends
st.subheader("📅 Sentiment Trends Over Time")
trend_data = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
st.line_chart(trend_data, use_container_width=True)

# Download cleaned data
st.subheader("⬇️ Download Cleaned Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="cleaned_bohemian_comments.csv", mime='text/csv')
