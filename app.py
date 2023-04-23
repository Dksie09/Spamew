import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  text = [word for word in text if word not in stopwords.words('english')]
  #remove punctuations and special characters
  text = [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in text]
  text = ' '.join(text)
  return text
  
st.title('SMS spam classifier')

input = st.text_area('Enter your message')

if st.button('Predict'):
    # Preprocess the input
    input = transform_text(input)

    # Vectorize the input
    vector_input = tfidf_vectorizer.transform([input])

    # Make predictions
    result = model.predict(vector_input)[0]

    #display
    st.empty()
    if result == 1:
        st.error('Spam', icon="🚨")
    else:
        st.success('Not Spam', icon="✅")
