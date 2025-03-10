import os
from flask import Flask, request, jsonify, render_template
from sentiment_model import SentimentModel

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(current_dir, 'embeddings')
model_path = os.path.join(current_dir, 'best_sentiment_model.pth')

sentiment_model = SentimentModel(
    embedding_dir=embeddings_dir,
    embedding_file='embedding_dictionary.pkl',
    model_path=model_path,
    max_len=100
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    sentiment = sentiment_model.analyze_sentiment(text)
    probability, label = sentiment_model.get_sentiment_probability(text)

    return jsonify({
        'sentiment': label,
        'probability': probability * 100  # Percentage
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8000, debug=True)