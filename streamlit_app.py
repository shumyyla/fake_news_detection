import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

model = joblib.load('model_fnd.pkl')
st.title("Pakistani Fake News Detection App")

user_input = st.text_area("Enter a news:")

predict_button = st.button("Predict")

def tokenize_and_clean_text(text):
    # Remove digits and words containing digits
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove the stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

if user_input and predict_button:
    cleaned_text = tokenize_and_clean_text(user_input) 
    user_input_vectorized = model.named_steps['tfidf'].transform([user_input])
    prediction = model.named_steps['classifier'].predict(user_input_vectorized)
    # Make a prediction
    prediction = model.predict([cleaned_text])[0]
    label = 'Fake' if prediction == 1 else 'Real'
    st.write(f"The news is:")
    st.subheader(f"{label}")
