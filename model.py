#importing library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
import re
import webbrowser
import pickle

# Download necessary NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
file_path = "website_classification.csv"
df = pd.read_csv(file_path)

# Remove rows with NaN values in 'website_url' or 'cleaned_website_text'
df = df.dropna(subset=['website_url', 'cleaned_website_text'])

# Preprocess URLs
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

df['original_url'] = df['website_url']  # Keep the original before preprocessing

# Modify preprocess_url to handle the comparison issue
def preprocess_url(url):
    if not isinstance(url, str):
        return ''
    url = re.sub('[^a-zA-Z0-9/:?&=._-]', ' ', url).lower().split()
    return ' '.join(url)

df['website_url'] = df['original_url'].apply(preprocess_url)
def extract_keywords(url):
    # Check if the URL is valid
    if not url or pd.isna(url):
        return ''
    
    parsed_url = urlparse(url)
    # Extract path and query parameters, then split into keywords
    keywords = parsed_url.path.split('/') + parsed_url.query.split('&')
    # Clean up keywords
    keywords = [word for word in keywords if word]  # Remove empty strings
    return ' '.join(keywords)

# Extract keywords from the URLs
df['keywords'] = df['website_url'].apply(extract_keywords)

# Check for NaN values in the 'keywords' column and replace them with an empty string
df['keywords'] = df['keywords'].fillna('')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['cleaned_website_text'].tolist() + df['keywords'].tolist())

# Split into description and keyword vectors
description_vector = tfidf_matrix[:len(df)]  # Descriptions
keyword_vector = tfidf_matrix[len(df):]      # Keywords

# Function to get most similar description for a given URL
def get_similar_description(url):
    keywords = extract_keywords(url)
    keywords_tfidf = vectorizer.transform([keywords])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(keywords_tfidf, description_vector).flatten()
    
   

    most_similar_index = similarity_scores.argmax()
    return df.iloc[most_similar_index]['cleaned_website_text']

def scrape_website_description(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    search_url = f"https://duckduckgo.com/html/?q={query}"
    try:
        search_response = requests.get(search_url, headers=headers, timeout=20)
        search_response.raise_for_status()

        search_soup = BeautifulSoup(search_response.text, 'html.parser')
        results = search_soup.find_all('a', class_='result__a')

        if results:
            first_result_url = results[0]['href']
        else:
            return "No results found for the query."

        response = requests.get(first_result_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text.strip() if soup.find('title') else "No title available"
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        min_length = 200
        words = content.split()[:min_length]
        description = ' '.join(words) if words else "descriptio"

    except requests.RequestException as e:
        return f"Failed to retrieve information: {e}"

    general_description = (f"{title} is a website that provides the following content: {description}."
                           " It offers a variety of resources and information on its main focus topics.")
    return general_description

def predict_or_scrape(url):
    # Preprocess the input URL similarly to how you processed the dataset
    preprocessed_url = preprocess_url(url)
    
    # Check if the original URL (before preprocessing) is in the dataset
    if url in df['original_url'].values:
        return get_similar_description(preprocessed_url)  # Use the preprocessed version for prediction
    
    # Otherwise, scrape the web to get the description
    
    return scrape_website_description(url)


# Example usage
url_example = input("Enter the URL: ")
output_description = predict_or_scrape(url_example)
print(f"Output Description: {output_description}")
