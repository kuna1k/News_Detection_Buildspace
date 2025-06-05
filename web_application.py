"""
Web Application for Fake News Detection
=======================================
Built with flask, light python based framework for similar apps.
For fake news detection using the fast and efficient model.

Features:
- Clean and responsive web interface
- Real-time predictions with confidence scores (both probabilities displayed upfront)
- Text analysis statistics
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
from datetime import datetime
import webbrowser
from threading import Timer

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the detector
from models import FastNewsDetector
from utils import setup_logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize the detector
detector = FastNewsDetector()

# Global variable to track if model is loaded
model_loaded = False

def load_model():
    """Load the trained model"""
    global model_loaded
    try:
        if os.path.exists('models/model.pkl'):
            detector.load_model('models/')
            model_loaded = True
            logging.info("Model loaded successfully")
        else:
            logging.warning("No trained model found. Please train the model first.")
            model_loaded = False
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        model_loaded = False

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://127.0.0.1:8000')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for news classification"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if not model_loaded:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get prediction
        result = detector.predict(text)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        # Add text stats
        result['text_stats'] = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()])
        }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model endpoint"""
    try:
        # Prepare combined dataset from True.csv and Fake.csv 
        # This should also be removed after one pass
        from train import prepare_training_data
        
        if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
            return jsonify({'error': 'Training files (True.csv and/or Fake.csv) not found'}), 400
        
        # Train the model directly with the DataFrame
        combined_df = prepare_training_data('True.csv', 'Fake.csv')
        detector.train(combined_df)
        
        # Load the trained model
        load_model()
        
        return jsonify({'message': 'Model trained successfully', 'model_loaded': model_loaded})
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get application status"""
    return jsonify({
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Set up logging
    setup_logging()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Load model if available
    load_model()

    # Only open browser if this is the main process (not the reloader process) (to counter bug of double tab upon running web app)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1.5, open_browser).start()
    
    # Open browser after a short delay
    # Timer(1.5, open_browser).start()
    
    # Run the application
    app.run(debug=False, host='0.0.0.0', port=8000)

    # App runs at http://127.0.0.1:8000, it is a background process and can be opened in this manner if the tab is closed out
    # Further edits to program will including adding a manual browser opening endpoint or shortcut functionalities