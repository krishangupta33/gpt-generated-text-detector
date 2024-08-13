# Import necessary libraries
import streamlit as st
from joblib import load
import nltk



import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import nltk
from sklearn.naive_bayes import MultinomialNB
#import pipeline
from sklearn.pipeline import Pipeline
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



# Define the TextCleanerLemmatizer class used in your pipeline
# This class is responsible for cleaning and lemmatizing the input text
class TextCleanerLemmatizer:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def fit(self, X, y=None):
        # This method is required by the sklearn pipeline,
        # but it doesn't need to do anything for this transformer
        return self

    def transform(self, X, y=None):
        # Process each text in X to clean and lemmatize it
        X_transformed = []
        for text in X:
            text = text.lower()
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            word_tokens = nltk.tokenize.word_tokenize(text)
            filtered_text = [word for word in word_tokens if word not in self.stop_words]
            lemmatized_text = ' '.join([self.lemmatizer.lemmatize(w) for w in filtered_text])
            X_transformed.append(lemmatized_text)
        return X_transformed

# Load the trained pipeline
pipeline = load('text_classification_pipeline.joblib')

# Streamlit app setup
st.title('Text Classification App')
st.write('Insert text with more than 250 words. The text should be argumentative essays')

# User text input
user_input = st.text_area("Enter text here:")


# Display word count
word_count = len(user_input.split())
st.caption(f"Word Count: {word_count}")


if user_input:  # Check if the user has started typing
    if word_count > 250:
        st.success("Your text meets the word count requirement.")
    else:
        st.warning("Your text should contain more than 250 words for reliable prediction.")

# Predict button action
if st.button('Predict'):
    if user_input:
        if word_count > 250:
            # Make prediction on the input text
            prediction = pipeline.predict([user_input])[0]
            
            # Display the prediction
            if prediction == 1:
                st.write('The text is predicted to be GPT-generated.')
            else:
                st.write('The text is predicted to be written by a human.')
                
        else:
            st.write('Please enter text with more than 250 words for a reliable prediction.')
    else:
        st.write('Please enter some text to make a prediction.')
    
