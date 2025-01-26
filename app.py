from flask import Flask, request, jsonify, render_template
from sentiment_model import analyze_sentiment 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    print(data)
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    text = data['text']
    sentiment = analyze_sentiment(text) 
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
