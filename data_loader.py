import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Loads and preprocesses the CoNLL-2003 NER dataset.
    """
    def __init__(self, data_path, max_len=75):
        self.data_path = data_path
        self.max_len = max_len
        self.words = ["PAD", "UNK"]
        self.tags = ["PAD", "O"]
        self.word_to_index = {}
        self.tag_to_index = {}
        self.index_to_tag = {}
        self.sentences = []
        
    def load_data(self):
        # 1. Load data and fill NaN sentence IDs
        data = pd.read_csv(self.data_path, encoding="latin1")
        data = data.fillna(method="ffill")
        
        # 2. Group words by sentence
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        grouped = data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in grouped]
        
        # 3. Build vocabularies
        all_words = list(set(data["Word"].values))
        all_tags = list(set(data["Tag"].values))
        
        self.words.extend(all_words)
        self.tags.extend(all_tags)
        
        self.word_to_index = {w: i for i, w in enumerate(self.words)}
        self.tag_to_index = {t: i for i, t in enumerate(self.tags)}
        self.index_to_tag = {i: t for t, i in self.tag_to_index.items()}
        
        print(f"Total sentences: {len(self.sentences)}")
        print(f"Total words in vocab: {len(self.words)}")
        print(f"Total tags: {len(self.tags)}")
        
    def preprocess(self):
        # 1. Convert sentences (words) to sequences of word indices
        X = [[self.word_to_index.get(w[0], self.word_to_index["UNK"]) for w in s] for s in self.sentences]
        
        # 2. Convert tags to sequences of tag indices
        y = [[self.tag_to_index.get(w[1]) for w in s] for s in self.sentences]
        
        # 3. Padding sequences to MAX_LEN
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word_to_index["PAD"])
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tag_to_index["PAD"])
        
        # 4. Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # 5. One-hot encode the tags for CRF (Keras's CRF layer handles its own loss, 
        #    but standard practice often involves one-hot for simpler models. 
        #    For Keras-CRF, often just the indices are used; however, we'll keep the indices 
        #    and let the CRF layer manage the loss internally on the index sequence).
        # We will use the integer index sequence 'y' directly for Keras-Contrib's CRF loss function.
        
        return X_train, X_test, y_train, y_test

# Example of expected file path: 'ner_dataset.csv' from the entity-annotated-corpus on Kaggle
# The CoNLL-2003 is often found in this format online.