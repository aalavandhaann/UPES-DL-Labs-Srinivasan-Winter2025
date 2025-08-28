
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

# -------------------------------
# Hyperparameters
# -------------------------------
max_features = 10000   # Number of unique words to consider
max_len = 500          # Sequence length (padding/truncation)
embedding_dim = 128    # Word embedding dimensions
rnn_units = 128        # Number of RNN hidden units
batch_size = 64
epochs = 3

# -------------------------------
# Load the IMDB dataset
# -------------------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# -------------------------------
# Preprocess data: Pad sequences
# -------------------------------
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# -------------------------------
# Build the RNN Model
# -------------------------------
model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(rnn_units, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# -------------------------------
# Compile the Model
# -------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# -------------------------------
# Train the Model
# -------------------------------
print("Training started...")
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")


# Get the word index mapping from IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    """Convert encoded integers back to readable text"""
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# Prepare a tokenizer for new reviews
tokenizer = Tokenizer(num_words=max_features)
tokenizer.word_index = word_index

def predict_review_sentiment(review_text: str):
    """Preprocess and predict sentiment for a single review"""
    # Convert review to sequence of integers
    seq = tokenizer.texts_to_sequences([review_text])
    padded_seq = sequence.pad_sequences(seq, maxlen=max_len)

    # Get prediction
    prediction = model.predict(padded_seq)
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

    print(f"\nReview: {review_text}")
    print(f"Predicted Sentiment: {sentiment} ({prediction[0][0]:.4f})")


# -------------------------------
# Example Predictions, you can change it and undestand how RNN works with this context
# -------------------------------
print("\nTesting custom reviews:")
predict_review_sentiment("The movie was absolutely fantastic and I loved it!")
predict_review_sentiment("This movie was terrible, I hated every moment.")


# -------------------------------
# Visualize Training Results
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
