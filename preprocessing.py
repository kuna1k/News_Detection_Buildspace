import re
import pandas as pd
import nltk
import os

# Ensure NLTK is available
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

# Initialize NLTK data
ensure_nltk_data()

def advanced_text_preprocessing(text: str) -> str:
    """
    Fast text preprocessing:
    - Lowercase
    - Remove URLs and emails
    - Remove non-alpha (keep spaces + basic punctuation)
    - Normalize whitespace
    """
    if pd.isna(text) or text == '':
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s.!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with empty strings"""
    return df.fillna('')

def combine_content_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title and text into full_content"""
    df['content'] = df['title'].astype(str)
    df['full_content'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    return df

def preprocess_datasets(true_file: str, fake_file: str) -> str:
    """
    Load and preprocess true and fake news datasets
    Returns path to preprocessed CSV file, delete after one parse
    """
    # Load datasets
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)
    
    # Add labels
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Basic preprocessing
    df = load_and_fillna(df)
    df = combine_content_columns(df)
    
    # Apply text preprocessing
    df['processed_content'] = df['full_content'].apply(advanced_text_preprocessing)
    
    # Save preprocessed data
    output_file = 'preprocessed_data.csv'
    df.to_csv(output_file, index=False)
    
    return output_file
