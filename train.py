import numpy as np
from data_loader import DataLoader
from model import build_bilstm_crf_model
from seqeval.metrics import classification_report

# --- Configuration ---
DATA_FILE = 'ner_dataset.csv' # Replace with your CoNLL-2003 file path
MAX_LEN = 75
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DROPOUT_RATE = 0.1
EPOCHS = 5
BATCH_SIZE = 32

def main():
    # 1. Data Loading and Preprocessing
    print("--- 1. Loading and Preprocessing Data ---")
    data_loader = DataLoader(DATA_FILE, max_len=MAX_LEN)
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.preprocess()
    
    n_words = len(data_loader.words)
    n_tags = len(data_loader.tags)
    index_to_tag = data_loader.index_to_tag

    # 2. Model Building
    print("--- 2. Building BiLSTM-CRF Model ---")
    model = build_bilstm_crf_model(
        max_len=MAX_LEN, 
        n_words=n_words, 
        n_tags=n_tags, 
        embedding_dim=EMBEDDING_DIM, 
        lstm_units=LSTM_UNITS, 
        dropout_rate=DROPOUT_RATE
    )
    model.summary()

    # 3. Model Training
    print("--- 3. Training Model ---")
    history = model.fit(
        X_train, 
        y_train, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        validation_split=0.1,
        verbose=1
    )

    # 4. Evaluation
    print("--- 4. Evaluating Model Performance ---")
    
    # Predict on the test set
    p_test = model.predict(X_test, verbose=0)
    
    # Decode predictions and true values from index to tag label
    y_pred_labels = []
    y_true_labels = []

    for i in range(len(X_test)):
        # Decode predicted sequence
        pred_indices = np.argmax(p_test[i], axis=-1)
        pred_labels = [index_to_tag[idx] for idx in pred_indices if index_to_tag[idx] != "PAD"]
        y_pred_labels.append(pred_labels)
        
        # Decode true sequence
        true_labels = [index_to_tag[idx] for idx in y_test[i] if index_to_tag[idx] != "PAD"]
        y_true_labels.append(true_labels)
        
    # Generate the classification report using seqeval
    # This correctly handles sequence labeling metrics (Precision, Recall, F1-score)
    print("\n--- Classification Report (F1-score) ---")
    print(classification_report(y_true_labels, y_pred_labels, digits=4))


    # 5. Prediction on Custom Text (Conceptual)
    # The actual prediction module would involve:
    # a. Tokenizing and converting custom text to indices
    # b. Padding the sequence
    # c. Calling model.predict()
    # d. Mapping the output indices back to tag strings

if __name__ == '__main__':
    main()