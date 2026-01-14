import streamlit as st
import joblib

# 1. Model aur Vectorizer load karein
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 2. Website ka Title aur UI
st.title("📰 Fake News Detection System")
st.write("Niche diye gaye box mein news paste karein aur check karein ki wo Real hai ya Fake.")

# 3. Input Box
user_input = st.text_area("News Text Paste Karein:", height=200)

# 4. Predict Button
if st.button("Check News"):
    if user_input:
        # Transform and Predict
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        
        # Result Display
        if prediction[0] == 1:
            st.success("✅ Ye News REAL hai!")
        else:
            st.error("⚠️ Ye News FAKE hai!")
    else:
        st.warning("Please enter some text first.")