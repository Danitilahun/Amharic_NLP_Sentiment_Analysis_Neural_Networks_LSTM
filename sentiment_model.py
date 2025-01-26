import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim=1, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim, batch_first=True)
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_norm_fc = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]

        hidden = self.batch_norm_lstm(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)

        out = self.fc(hidden)
        out = self.batch_norm_fc(out)

        return out
class SentimentModel:
    def __init__(self, embedding_dir, embedding_file, model_path, max_len=50):
        self.device = DEVICE
        self.max_len = max_len
        self.embedding_matrix = self.load_embedding_dictionary(embedding_dir, embedding_file)
        self.model = self.load_trained_model(model_path)
    
    def load_trained_model(self, model_path):
        model = SentimentLSTM(embedding_dim=EMBEDDING_DIM,
                              hidden_dim=HIDDEN_DIM,
                              output_dim=OUTPUT_DIM).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print('Trained sentiment model loaded successfully.')
        return model
    
    def load_embedding_dictionary(self, embeddings_dir, embedding_file='embedding_dictionary.pkl'):
        embedding_file_path = os.path.join(embeddings_dir, embedding_file)
        if not os.path.exists(embedding_file_path):
            raise FileNotFoundError(f"Embedding file not found at {embedding_file_path}")

        with open(embedding_file_path, 'rb') as f:
            embedding_dictionary = pickle.load(f)

        print('Embedding dictionary loaded successfully.')
        return embedding_dictionary
    
    def vectorize_text(self, text):
        tokens = word_tokenize(text.lower())
        vectors = [
            self.embedding_matrix[token] if token in self.embedding_matrix else np.zeros(
                EMBEDDING_DIM)
            for token in tokens
        ]

        if len(vectors) < self.max_len:
            padding = [np.zeros(EMBEDDING_DIM)] * (self.max_len - len(vectors))
            vectors.extend(padding)
        else:
            vectors = vectors[:self.max_len]

        return np.array(vectors)
    
    def predict_sentiment(self, text):
        vector = self.vectorize_text(text)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(
            0).to(self.device) 

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

        return probabilities
    
    def get_sentiment_probability(self, text):
        probability = self.predict_sentiment(text)
        prediction = "positive" if probability >= 0.5 else "negative"
        confidence = probability if probability >= 0.5 else 1 - probability
        return confidence, prediction
    
    def analyze_sentiment(self, text):
        _, prediction = self.get_sentiment_probability(text)
        return prediction
