```markdown
# Amharic Sentiment Analysis using Neural Networks

This project is designed to perform sentiment analysis on Amharic text using a neural network model. The model is built using **PyTorch** for the machine learning component and **Flask** for the web interface. The project includes preprocessing of Amharic text, training a sentiment analysis model, and deploying it as a web application.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

The project is organized as follows:

```
.
├── data/
│   ├── embeddings/                  # Contains pre-trained embeddings
│   │   └── embedding_dictionary.pkl
│   ├── cleaned_dataset.csv          # Cleaned dataset for training
│   ├── dataset.csv                  # Raw dataset
│   ├── notebooks/                   # Jupyter notebooks for preprocessing and analysis
│   │   ├── Amharic_text_Preprocessing.ipynb
│   │   └── SentimentAnalyzer.ipynb
│   └── amharic_preprocessing_data.py  # Script for preprocessing Amharic text
├── templates/                       # HTML templates for the web interface
│   └── index.html
├── .gitignore                       # Specifies files to ignore in Git
├── app.py                           # Flask application for the web interface
├── best_sentiment_model.pth         # Pre-trained sentiment analysis model
├── requirements.txt                 # List of dependencies
└── sentiment_model.py               # Sentiment analysis model implementation
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Danitilahun/Amharic_NLP_Sentiment_Analysis_Neural_Networks_LSTM.git
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

1. **Run the Flask application:**
   ```bash
    flask run --port=5001
   ```

2. **Access the web interface:**
   Open your browser and go to `http://127.0.0.1:5001/`.

3. **Analyze sentiment:**
   Enter Amharic text in the provided input field and submit to get the sentiment analysis result.

---

## Model Performance

The model's performance on the **train** and **test** datasets is as follows:

### Train Metrics:
- **Accuracy**: 0.9461
- **Precision**: 0.9757
- **Recall**: 0.9185
- **F1 Score**: 0.9464
- **AUC**: 0.9781

### Test Metrics:
- **Accuracy**: 0.9219
- **Precision**: 0.9522
- **Recall**: 0.8937
- **F1 Score**: 0.9220
- **AUC**: 0.9575

### Classification Report (Train):
- **Class 0.0**:
  - Precision: 0.9179
  - Recall: 0.9754
  - F1-Score: 0.9458
  - Support: 26757.0
- **Class 1.0**:
  - Precision: 0.9756
  - Recall: 0.9188

### Classification Report (Test):
- **Class 0.0**:
  - Precision: 0.8939
  - Recall: 0.9529
  - F1-Score: 0.9218
  - Support: 2983.0
- **Class 1.0**:
  - Precision: 0.9523
  - Recall: 0.8937

---

## API Endpoints

The Flask application provides the following API endpoints:

- **GET `/`**: Renders the home page with the sentiment analysis form.
- **POST `/analyze`**: Accepts JSON data with a `text` field and returns the sentiment analysis result.

### Example Request:
```json
{
  "text": "ይህ ፕሮጀክት በጣም ጥሩ ነው!"
}
```

### Example Response:
```json
{
  "sentiment": "positive",
  "probability": 95.7
}
```

---

## Dependencies

The project relies on the following Python libraries:

- **Flask**: For building the web interface.
- **gunicorn**: For deploying the Flask app.
- **torch**: For building and training the neural network.
- **nltk**: For text tokenization.
- **numpy**: For numerical operations.
- **pickle**: For loading pre-trained embeddings.

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

**Note**: This project is for educational purposes and may require further tuning for production use.
```

---

### How to Use:
1. Copy the entire content above.
2. Paste it into your `README.md` file in your project repository.
3. Save the file, and it will be ready to display on GitHub with all the metrics and details.