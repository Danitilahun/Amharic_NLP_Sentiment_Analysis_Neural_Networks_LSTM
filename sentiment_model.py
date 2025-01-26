# import os
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# from nltk.tokenize import word_tokenize
# import nltk

# # Download NLTK data if not already present
# nltk.download('punkt')

# # Model and training parameters
# embedding_dim = 100
# hidden_dim = 64
# output_dim = 1  # Binary classification: positive vs. negative
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class SentimentLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, output_dim=1, dropout=0.5):
#         super(SentimentLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=embedding_dim,
#                             hidden_size=hidden_dim, batch_first=True)
#         self.batch_norm_lstm = nn.BatchNorm1d(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.batch_norm_fc = nn.BatchNorm1d(output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         _, (hidden, _) = self.lstm(x)
#         hidden = hidden[-1]

#         hidden = self.batch_norm_lstm(hidden)
#         hidden = self.relu(hidden)
#         hidden = self.dropout(hidden)

#         out = self.fc(hidden)
#         out = self.batch_norm_fc(out)

#         return out  # No activation; apply sigmoid during prediction

# def load_trained_model(model_path):
#     model = SentimentLSTM(embedding_dim=embedding_dim,
#                           hidden_dim=hidden_dim, output_dim=1).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     return model

# def load_embedding_dictionary(embeddings_dir, embedding_file='embedding_dictionary.pkl'):
#     embedding_file_path = os.path.join(embeddings_dir, embedding_file)

#     if not os.path.exists(embedding_file_path):
#         raise FileNotFoundError(f"Embedding file not found at {embedding_file_path}")

#     with open(embedding_file_path, 'rb') as f:
#         embedding_dictionary = pickle.load(f)

#     print('Embedding dictionary loaded successfully.')
#     return embedding_dictionary

# # Initialize paths
# current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
# embeddings_dir = os.path.join(current_dir, 'embeddings')
# model_path = os.path.join(current_dir, 'best_sentiment_model.pth')  # Path to the trained sentiment model

# # Load resources
# print(f"Loading model from: {model_path}")
# best_model = load_trained_model(model_path)
# embedding_dictionary = load_embedding_dictionary(current_dir)  # Corrected to embeddings_dir
# print('Trained sentiment model loaded successfully.')

# def vectorize_text(text, embedding_matrix, max_len=50):
#     tokens = word_tokenize(text)
#     vectors = [
#         embedding_matrix[token] if token in embedding_matrix else np.zeros(
#             len(next(iter(embedding_matrix.values()))))
#         for token in tokens
#     ]

#     if len(vectors) < max_len:
#         padding = [np.zeros(
#             len(next(iter(embedding_matrix.values()))))] * (max_len - len(vectors))
#         vectors.extend(padding)
#     else:
#         vectors = vectors[:max_len]

#     return np.array(vectors)

# def predict_sentiment(text, model, embedding_matrix, device, max_len=50):
#     vector = vectorize_text(text, embedding_matrix, max_len=max_len)
#     tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(
#         0).to(device)  # Shape: (1, max_len, embedding_dim)

#     with torch.no_grad():
#         output = model(tensor)  # Shape: (1, output_dim)
#         probabilities = torch.sigmoid(output).squeeze().cpu().numpy()  # Shape: ()

#     # Determine the prediction based on probability
#     prediction = "positive" if probabilities >= 0.5 else "negative"
#     probability = probabilities if probabilities >= 0.5 else 1 - probabilities

#     return probabilities, prediction

# def process_text(text, model, embedding_matrix, device, max_len=50):
#     probability, prediction = predict_sentiment(
#         text, model, embedding_matrix, device, max_len)

#     return {
#         "text": text,
#         "model_sentiment_probability": probability * 100,  # Percentage
#         "model_sentiment_label": prediction
#     }
    
# def analyze_sentiment(text):
#     result = process_text(text, best_model, embedding_dictionary, device, max_len=50)
#     return result["model_sentiment_label"]


# sentiment_model.py
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Model parameters
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
            0).to(self.device)  # Shape: (1, max_len, embedding_dim)

        with torch.no_grad():
            output = self.model(tensor)  # Shape: (1, output_dim)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()  # Shape: ()

        return probabilities
    
    def get_sentiment_probability(self, text):
        probability = self.predict_sentiment(text)
        prediction = "positive" if probability >= 0.5 else "negative"
        confidence = probability if probability >= 0.5 else 1 - probability
        return confidence, prediction
    
    def analyze_sentiment(self, text):
        _, prediction = self.get_sentiment_probability(text)
        return prediction
