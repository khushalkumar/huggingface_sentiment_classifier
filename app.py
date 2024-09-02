from flask import Flask, request, jsonify
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analyze_sentiment(review_text):
    try:
        # Tokenize the input text and prepare tensors
        inputs = tokenizer.encode_plus(
            review_text,
            return_tensors="tf",
            max_length=512,
            truncation=True
        )
        
        # Run the model and get predictions
        outputs = model(inputs['input_ids'])
        
        # Convert logits to probabilities
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        
        # Get the predicted sentiment label and confidence
        predicted_label = tf.argmax(probs, axis=1).numpy()[0]
        confidence = float(np.max(probs.numpy()))  # Convert to float
        
        # Map the label to sentiment
        sentiment_map = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
        sentiment = sentiment_map.get(predicted_label, "unknown")
        
        return {
            "review_text": review_text,
            "sentiment": sentiment,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {
            "error": "An error occurred while processing the review."
        }


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    
    if 'review_text' not in data:
        return jsonify({"error": "No review text provided"}), 400
    
    review_text = data['review_text']
    sentiment_analysis = analyze_sentiment(review_text)
    
    response = {
        "review_text": sentiment_analysis["review_text"],
        "sentiment": sentiment_analysis["sentiment"],
        "confidence": sentiment_analysis["confidence"]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
