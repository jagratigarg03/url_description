from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import requests

from bs4 import BeautifulSoup
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Flask app
app = Flask(__name__)

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the dataset 
file_path = "website_classification.csv"
df = pd.read_csv(file_path)

# Preprocess the dataset
stop_words = set(stopwords.words('english'))
df['original_url'] = df['website_url']
df = df.dropna(subset=['website_url', 'cleaned_website_text'])

def preprocess_url(url):
    url = url.strip()
    url = re.sub(r'%[0-9A-Fa-f]{2}', '', url)
    return url

def extract_keywords(url):
    parsed_url = urlparse(url)
    keywords = parsed_url.path.split('/') + parsed_url.query.split('&')
    keywords = [word for word in keywords if word]
    return ' '.join(keywords)

df['website_url'] = df['original_url'].apply(preprocess_url)
df['keywords'] = df['website_url'].apply(extract_keywords)
df['keywords'] = df['keywords'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['cleaned_website_text'].tolist() + df['keywords'].tolist())
description_vector = tfidf_matrix[:len(df)]
keyword_vector = tfidf_matrix[len(df):]

def get_similar_description(url):
    keywords = extract_keywords(url)
    keywords_tfidf = vectorizer.transform([keywords])
    similarity_scores = cosine_similarity(keywords_tfidf, description_vector).flatten()
    most_similar_index = similarity_scores.argmax()
    return df.iloc[most_similar_index]['cleaned_website_text']

def extract_website_description(url, timeout=30, user_agent="MyWebScraper/1.0"):
    try:
        response = requests.get(url, timeout=timeout, headers={'User-Agent': user_agent})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            return meta_description['content']
        title = soup.title.string if soup.title else "No title available"
        body = soup.find('body')
        text = title + " " + (body.get_text(separator=" ") if body else "No body content available")
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        description = " ".join(filtered_words)
        return description
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def predict_or_scrape(url):
    preprocessed_url = preprocess_url(url)
    if preprocessed_url in df['original_url'].values:
        return get_similar_description(preprocessed_url)
    else:
        try:
            description = extract_website_description(preprocessed_url)
            return description if description else "No description available."
        except Exception as e:
            print(f"Error fetching URL: {e}")
            return "Unable to retrieve description for the provided URL."

# API endpoint
@app.route('/get-description', methods=['GET', 'POST'])
def get_description():
    if request.method == 'POST':
        data = request.get_json()  # Get the request JSON data
        url = data.get('url', '')  # Extract the URL from the request
    elif request.method == 'GET':
        url = request.args.get('url', '')  # Extract the URL from the query parameters

    if url:
        description = predict_or_scrape(url)  # Get the description
        return jsonify({'url': url, 'description': description})
    else:
        return jsonify({'error': 'No URL provided'}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
