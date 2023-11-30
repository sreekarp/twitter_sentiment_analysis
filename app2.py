from flask import Flask, render_template, request
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the tokenizer using pickle
tokenizer_path = "bert_tokenizer.pkl"
with open(tokenizer_path, "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the model using torch.load
model_path = "bert_torch_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define a function to predict the sentiment of a text
def predict_sentiment(text):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = input_ids.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        predicted_class = predict_sentiment(text)
        sentiment = "Positive" if predicted_class == 0 else "Negative"
        return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
