import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# -------------------------------
# Load Sales Data
# -------------------------------
st.title("📊 Sales Data Sentiment Analysis")

uploaded_file = st.file_uploader("Upload your Sales CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Expecting columns: 'Product', 'Sales', 'Feedback'
    if "Feedback" not in df.columns:
        st.error("Dataset must contain a 'Feedback' column with customer comments.")
    else:
        # -------------------------------
        # Sentiment Analysis
        # -------------------------------
        st.subheader("Sentiment Analysis")

        def get_sentiment(text):
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0:
                return "Positive"
            elif polarity < 0:
                return "Negative"
            else:
                return "Neutral"

        df["Sentiment"] = df["Feedback"].apply(get_sentiment)

        st.write(df[["Product", "Sales", "Feedback", "Sentiment"]].head())

        # -------------------------------
        # Visualization
        # -------------------------------
        st.subheader("Visualizations")

        # Sentiment distribution
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Sales vs Sentiment
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Sentiment", y="Sales", data=df, ax=ax2)
        ax2.set_title("Sales by Sentiment Category")
        st.pyplot(fig2)

        # Product-wise sentiment
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.countplot(x="Product", hue="Sentiment", data=df, ax=ax3)
        ax3.set_title("Sentiment per Product")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        st.success("✅ Sentiment analysis complete! Explore the charts above.")
