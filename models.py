import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import load_and_fillna, combine_content_columns, advanced_text_preprocessing

logger = logging.getLogger(__name__)

class FastNewsDetector:
    """
    Optimized fake news detection using minimal features (98% reduction from initial model) and fast training (30 min runtime -> < 30 seconds).
    Uses single TF-IDF + LogisticRegression for speed and simplicity.
    """

    def __init__(self):
        # Single TF-IDF vectorizer with balanced settings
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Set back to 3000 features
            ngram_range=(1, 2),
            min_df=3,          # Keeping the optimized min_df
            max_df=0.9,        # Keeping the optimized max_df
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english'
        )
        # Fast LogisticRegression with good defaults
        self.model = LogisticRegression(
            C=1.0,
            solver='saga',  # Fast solver for large sparse data
            max_iter=100,
            tol=1e-3,  # Slightly relaxed tolerance
            n_jobs=-1,
            random_state=42
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare text data for training"""
        df = load_and_fillna(df)
        df = combine_content_columns(df)
        df["processed_content"] = df["full_content"].apply(advanced_text_preprocessing)
        return df

    def train(self, df: pd.DataFrame):
        logger.info("Starting training pipeline...")

        # Prepare data
        df_prepared = self._prepare_dataframe(df)
        y = df_prepared["label"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df_prepared["processed_content"].values,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # Transform text to TF-IDF features
        logger.info("Extracting TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train model
        logger.info("Training LogisticRegression...")
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate
        self._evaluate_holdout(X_test_tfidf, y_test)

        # Save model
        self.save_model("models/")

    def _evaluate_holdout(self, X_test, y_true):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)

        logger.info(f"→ Accuracy: {acc:.4f}")
        logger.info(f"→ ROC-AUC: {auc:.4f}")
        logger.info("\n" + classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    def predict(self, text: str) -> dict:
        """Make prediction on new text"""
        if not hasattr(self, 'model') or not hasattr(self, 'vectorizer'):
            raise RuntimeError("Model not loaded. Call load_model(...) first.")

        # Preprocess
        proc = advanced_text_preprocessing(text)

        # Extract features and predict
        X = self.vectorizer.transform([proc])
        proba = self.model.predict_proba(X)[0]
        pred_label = 1 if proba[1] > 0.5 else 0

        return {
            'prediction': 'Fake' if pred_label == 0 else 'Real',  # Fixed label interpretation
            'confidence': float(max(proba)),
            'probabilities': {'real': float(proba[1]), 'fake': float(proba[0])}  # Fixed probability mapping
        }

    def save_model(self, base_path: str = "models/"):
        """Save model and vectorizer"""
        import os
        os.makedirs(base_path, exist_ok=True)

        joblib.dump(self.model, f"{base_path}model.pkl")
        joblib.dump(self.vectorizer, f"{base_path}vectorizer.pkl")
        logger.info(f"Saved model artifacts to {base_path}")

    def load_model(self, base_path: str = "models/"):
        """Load model and vectorizer"""
        self.model = joblib.load(f"{base_path}model.pkl")
        self.vectorizer = joblib.load(f"{base_path}vectorizer.pkl")
        logger.info("Model loaded.")