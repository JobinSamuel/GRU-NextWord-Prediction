
#GRU Next Word Prediction (Jane Austen - Emma) + Streamlit App

A deep learning project that predicts the next word in a sentence using a GRU-based Neural Network trained on Jane Austen’s Emma (NLTK Gutenberg dataset). The model is deployed using a simple and interactive Streamlit web app for real-time predictions.

Project Highlights
Built a Next Word Prediction model using TensorFlow/Keras GRU Trained on a real-world text corpus (austen-emma.txt) Created an easy-to-use Streamlit interface for live inference Saved and reused the trained model + tokenizer for deployment

Model Architecture
This project uses an embedding + stacked GRU architecture:
	•	Embedding Layer
	•	GRU Layer (150 units, return_sequences=True)
	•	Dropout (0.1)
	•	GRU Layer (100 units)
	•	Dense Layer (ReLU)
	•	Output Dense Layer (Softmax)

Training Note (Local System Limitation)
The model was trained locally on my personal computer.
	•	Training was limited to 200 epochs
	•	It took approximately 72 minutes to complete training
	•	The accuracy could be improved further by training longer / using stronger hardware (GPU)
This was a tradeoff between performance and practicality while training on local resources.

Streamlit App Demo
The Streamlit app loads the trained model and tokenizer:
	•	Loads the model from: best_model.keras
	•	Loads the tokenizer from: tokenizer.pickle
	•	Takes user input and predicts the next word instantly
App UI Title:
Next Word Prediction with GRU (Title used in the app)
Works with any user-entered sentence.


How Prediction Works
The app uses this pipeline:
	1	Converts the input text into tokens
	2	Pads the sequence to match training input size
	3	Runs prediction using the trained GRU model
	4	Extracts the predicted word index using argmax()
	5	Maps it back into the word using tokenizer vocabulary

Example Usage
Input:
"I was walking down the street when"
Output:
Next word Prediction: i
(Your output depends on the trained vocabulary and patterns learned from Emma)

 Future Improvements
Train for more epochs (300+ / 500+) Use GPU training for faster experimentation Improve accuracy using:
	•	Learning rate scheduling
	•	Better regularization (Dropout tuning)
	•	Larger embeddings
	•	Temperature sampling instead of argmax Add top-k predictions (not just 1 word)

Conclusion
This project demonstrates an end-to-end deep learning workflow:
Dataset → Tokenization → Sequence Generation → GRU Training → Model Saving → Streamlit Deployment
