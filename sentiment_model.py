def analyze_sentiment(text):
    # Replace this with your model logic
    if "good" in text.lower():
        return "positive"
    elif "bad" in text.lower():
        return "negative"
    else:
        return "neutral"
