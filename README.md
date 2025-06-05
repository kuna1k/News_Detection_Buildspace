# Fast Fake News Detector

A streamlined and efficient fake news detection system using ML. This project provides both an interface for training and a web interface for making predictions (make the user interaction more straight forward rather than using tool within VS Code/Code editor). In today's age of short-form content and a lack of reputable sources, use this tool to combat misinformation.

## Features

- Fast and efficient text processing
- Simple but effective machine learning model
- Clean web interface for predictions
- Detailed confidence scores and text statistics

## Project Structure

```
├── models.py           # Core model implementation
├── preprocessing.py    # Text preprocessing utilities
├── train.py           # Training script
├── web_application.py # Web interface
├── utils.py           # Logging utilities
├── requirements.txt   # Python dependencies
└── templates/         # Web templates
    └── index.html    # Main web interface
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the training data:
   - Download the following files from [Kaggle's Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset):
     - `True.csv`
     - `Fake.csv`
   - Place both files in the project root directory

## Usage

### Important: Two-Step Training Process

1. First, preprocess the data:
   ```python
   from preprocessing import ensure_nltk_data, advanced_text_preprocessing, load_and_fillna, combine_content_columns
   
   # Ensure NLTK data is downloaded
   ensure_nltk_data()
   
   # Use preprocessing functions on your data before training
   # Example:
   df = load_and_fillna(your_dataframe)
   df = combine_content_columns(df)
   df['processed_content'] = df['full_content'].apply(advanced_text_preprocessing)
   ```
   This step is required as otherwise data utilized by model is not taken in appropriately (format).

2. Then train the model:
   ```bash
   python train.py
   ```
   This will create a `models/` directory with the trained model.

3. Run the web application:
   ```bash
   python web_application.py
   ```
   Your default web browser will automatically open to http://localhost:8000

   Note: On macOS, if port 8000 is in use, modify the port in web_application.py (ex: port 5000).
   Also let's say you run the model once and then want to rerun for whatever reason, you should clear cache (both the python cache as well previously stored models and their information)

## Model Details

The system uses:
- TF-IDF vectorization for text features
- Logistic Regression for classification
- Optimized preprocessing pipeline

Performance metrics on the test set:
- Accuracy: ~99%
- ROC-AUC: ~0.99

## Input Format

The model accepts news article text and returns:
- Prediction (Real/Fake)
- Confidence score
- Text statistics (character count, word count, sentence count)

## Development

- The project uses a simplified architecture focusing on speed and efficiency (initial approach saw 30 minute runtime and computer overload)
- Model training is done using a single, well-tuned classifier (rather than the intensive ensemble methods)
- Web interface provides real-time predictions with detailed statistics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Dataset from Kaggle's "Fake and Real News Dataset"
- Built with Flask, scikit-learn, and pandas 