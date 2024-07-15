import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
import tempfile

# Check if nltk punkt and stopwords are already downloaded, otherwise download them
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the trained model
model_path = 'resume_classifier_model.pkl'
classifier = joblib.load(model_path)

# Load the dataset
resume_data = pd.read_csv("UpdatedResumeDataSet.csv")
resume_text = resume_data['Resume'].values

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), '', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Apply the cleaning function to the resumes
resume_data['clean_data'] = resume_data['Resume'].apply(cleanResume)
clean_resume_text = resume_data['clean_data'].values
categories = resume_data['Category'].values

# Load other necessary components (e.g., TfidfVectorizer)
vectorizer = TfidfVectorizer()
vectorizer.fit(clean_resume_text)

def extract_text_from_pdf(pdf_path):
    try:
        pages = convert_from_path(pdf_path, 500)
        text_data = ''
        for page in pages:
            text = pytesseract.image_to_string(page)
            text_data += text + '\n'
        return text_data
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def sent_split(paragraph, max_length):
    words = paragraph.split()
    sentences = []
    current_sentence = ""

    for word in words:
        if len(current_sentence) + len(word) <= max_length:
            current_sentence += word + " "
        else:
            sentences.append(current_sentence.strip())
            current_sentence = word + " "

    sentences.append(current_sentence.strip())
    return sentences

def extractive_summary(text):
    sentences = sent_split(text, 250)
    sentences_clean = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]
    stop_words = stopwords.words('english')
    sentence_tokens = [[word for word in sentence.split(' ') if word not in stop_words] for sentence in sentences_clean]

    vectorizer = TfidfVectorizer()
    sentence_embeddings = vectorizer.fit_transform([' '.join(tokens) for tokens in sentence_tokens]).toarray()

    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i in range(len(sentence_tokens)):
        for j in range(len(sentence_tokens)):
            similarity_matrix[i][j] = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[j]])[0, 0]

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    top_sentence = {sentence: scores[index] for index, sentence in enumerate(sentences)}
    top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:5])

    summary = "\n".join(top.keys())
    return summary

# Streamlit app
st.title('Resume Classifier and Summarizer')

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(tmp_file_path)
    os.remove(tmp_file_path)  # Remove the temporary file

    if text:
        st.write("Extracted Text:")
        st.write(text)

        # Clean the extracted text
        clean_text = cleanResume(text)

        # Temporarily append the cleaned text to the dataset for vectorization
        temp_resume_text = np.append(clean_resume_text, clean_text)
        vectorized_resumes = vectorizer.fit_transform(temp_resume_text)

        # Get the vectorized input resume (last one)
        vectorized_input_resume = vectorized_resumes[-1]
        vectorized_resumes = vectorized_resumes[:-1]  # Remove the input resume from the dataset

        # Predict the category
        prediction = classifier.predict(vectorized_input_resume)
        st.write("Predicted Category:")
        st.write(prediction[0])

        # Summarize the text using extractive summarization
        summary = extractive_summary(clean_text)
        st.write("Extractive Summary:")
        st.write(summary)
    else:
        st.write("Could not extract text from the uploaded PDF.")
