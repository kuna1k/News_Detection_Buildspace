import logging
import os
import pandas as pd
from models import FastNewsDetector
from preprocessing import preprocess_datasets

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Preprocess data first
    logger.info("Starting preprocessing phase...")
    preprocessed_file = preprocess_datasets('True.csv', 'Fake.csv')
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    df = pd.read_csv(preprocessed_file)
    
    # Train model
    logger.info("Training model...")
    detector = FastNewsDetector()
    detector.train(df)
    
    logger.info("Training complete! Model saved in models/ directory.")

if __name__ == '__main__':
    main()
    