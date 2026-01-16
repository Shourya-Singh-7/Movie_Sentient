import os
import re
import string
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Text Cleaning
def clean_text(text: str) -> str:
    """
    Robust text cleaning: handle non-strings, remove HTML tags, 
    punctuation, numbers, and extra spaces.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Load & Clean Dataset
def load_and_clean_data(csv_path: str, text_col: str, label_col: str) -> tuple:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file at: {csv_path}")
        
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Progress indicator
    print(f"Cleaning {len(df)} reviews...")
    df[text_col] = df[text_col].apply(clean_text)

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    return X, y


# 3. Train/Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# 4. TF-IDF Vectorization
def vectorize_text(X_train, X_test, max_features=5000, ngrams=(1, 2)):
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngrams
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


# 5. Save Vectorizer
def save_vectorizer(vectorizer, path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {path}")


# 6. Main Pipeline
def preprocess_pipeline(
        csv_path: str,
        text_col: str,
        label_col: str,
        save_vec_path: str,  # Removed default to force explicit path
        test_size=0.2,
        random_state=42,
        max_features=5000,
        ngrams=(1, 2)
    ):
    """
    Full pipeline: load → clean → split → vectorize → save vectorizer.
    Returns vectorized splits & labels.
    """

    # Load & clean
    X, y = load_and_clean_data(csv_path, text_col, label_col)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # Vectorize
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train, X_test,
        max_features=max_features,
        ngrams=ngrams
    )

    # Save vectorizer
    save_vectorizer(vectorizer, save_vec_path)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# Run standalone for quick testing
if __name__ == "__main__":
    # PATH FIX: Handle the 'train' folder structure
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Inside 'train/'
    project_root = os.path.dirname(current_dir)             # Inside 'Movie_review_analysis/'
    
    csv_file = os.path.join(project_root, "Data", "Raw_data.csv")
    vec_path = os.path.join(project_root, "models", "vectorizer.pkl")

    # Correct column names based on your dataset
    text_column = "review"
    label_column = "sentiment"

    try:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer = preprocess_pipeline(
            csv_path=csv_file, 
            text_col=text_column, 
            label_col=label_column,
            save_vec_path=vec_path
        )

        print("\nPreprocessing complete!")
        print("Train Matrix Shape:", X_train_vec.shape)
        print("Test Matrix Shape: ", X_test_vec.shape)
        
    except Exception as e:
        print(f"\nError: {e}")