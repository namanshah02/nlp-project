"""
BiLSTM-CRF for Named Entity Recognition and Shallow Parsing
Author: NLP Project Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# For metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print("=" * 60)
print("BiLSTM-CRF for NER and Shallow Parsing")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING MODULE
# ============================================================================

class CoNLLDataLoader:
    """Loads and processes CoNLL-2003 format data"""
    
    def __init__(self):
        self.sentences = []
        self.labels = []
        self.pos_tags = []
        
    def load_conll_format(self, filepath):
        """Load data in CoNLL format"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sentence = []
                tags = []
                pos = []
                
                for line in f:
                    line = line.strip()
                    if line == "" or line.startswith("-DOCSTART-"):
                        if sentence:
                            self.sentences.append(sentence)
                            self.labels.append(tags)
                            self.pos_tags.append(pos)
                            sentence = []
                            tags = []
                            pos = []
                    else:
                        parts = line.split()
                        if len(parts) >= 4:
                            sentence.append(parts[0])
                            pos.append(parts[1])
                            tags.append(parts[3])
                
                if sentence:
                    self.sentences.append(sentence)
                    self.labels.append(tags)
                    self.pos_tags.append(pos)
            
            print(f"✓ Loaded {len(self.sentences)} sentences from {filepath}")
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return False
    
    def create_sample_data(self):
        """Create sample training data for demonstration"""
        print("\n📊 Creating sample dataset (CoNLL-2003 format simulation)...")
        
        # Sample sentences with NER tags
        sample_data = [
            (["Apple", "Inc.", "is", "looking", "at", "buying", "U.K.", "startup"],
             ["B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "O"],
             ["NNP", "NNP", "VBZ", "VBG", "IN", "VBG", "NNP", "NN"]),
            
            (["Steve", "Jobs", "founded", "Apple", "in", "California"],
             ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
             ["NNP", "NNP", "VBD", "NNP", "IN", "NNP"]),
            
            (["Google", "announced", "new", "features", "in", "Mountain", "View"],
             ["B-ORG", "O", "O", "O", "O", "B-LOC", "I-LOC"],
             ["NNP", "VBD", "JJ", "NNS", "IN", "NNP", "NNP"]),
            
            (["Microsoft", "CEO", "Satya", "Nadella", "visited", "India"],
             ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC"],
             ["NNP", "NN", "NNP", "NNP", "VBD", "NNP"]),
            
            (["The", "European", "Union", "meets", "in", "Brussels", "today"],
             ["O", "B-ORG", "I-ORG", "O", "O", "B-LOC", "O"],
             ["DT", "NNP", "NNP", "VBZ", "IN", "NNP", "NN"]),
            
            (["Amazon", "opened", "new", "offices", "in", "Seattle"],
             ["B-ORG", "O", "O", "O", "O", "B-LOC"],
             ["NNP", "VBD", "JJ", "NNS", "IN", "NNP"]),
            
            (["President", "Biden", "met", "with", "Chinese", "officials"],
             ["O", "B-PER", "O", "O", "B-MISC", "O"],
             ["NN", "NNP", "VBD", "IN", "JJ", "NNS"]),
            
            (["Tesla", "factory", "in", "Texas", "produces", "electric", "vehicles"],
             ["B-ORG", "O", "O", "B-LOC", "O", "O", "O"],
             ["NNP", "NN", "IN", "NNP", "VBZ", "JJ", "NNS"]),
        ]
        
        # Replicate data to have more training samples
        for _ in range(25):
            for sent, tags, pos in sample_data:
                self.sentences.append(sent)
                self.labels.append(tags)
                self.pos_tags.append(pos)
        
        print(f"✓ Created {len(self.sentences)} sample sentences")
        return True

# ============================================================================
# 2. TEXT PREPROCESSING MODULE
# ============================================================================

class TextPreprocessor:
    """Handles all text preprocessing tasks"""
    
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0}
        self.pos2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2tag = {0: "<PAD>"}
        self.vocab_size = 2
        self.tag_size = 1
        self.pos_size = 2
        
    def build_vocab(self, sentences, labels, pos_tags):
        """Build vocabulary from sentences"""
        print("\n🔤 Building vocabulary...")
        
        # Build word vocabulary
        word_freq = Counter()
        for sent in sentences:
            word_freq.update(sent)
        
        for word, freq in word_freq.items():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.vocab_size += 1
        
        # Build tag vocabulary
        for tags in labels:
            for tag in tags:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = self.tag_size
                    self.idx2tag[self.tag_size] = tag
                    self.tag_size += 1
        
        # Build POS vocabulary
        for pos_seq in pos_tags:
            for pos in pos_seq:
                if pos not in self.pos2idx:
                    self.pos2idx[pos] = self.pos_size
                    self.pos_size += 1
        
        print(f"✓ Vocabulary size: {self.vocab_size}")
        print(f"✓ Tag size: {self.tag_size}")
        print(f"✓ POS size: {self.pos_size}")
        
    def encode_sentence(self, sentence):
        """Convert sentence to indices"""
        return [self.word2idx.get(word, 1) for word in sentence]
    
    def encode_tags(self, tags):
        """Convert tags to indices"""
        return [self.tag2idx.get(tag, 0) for tag in tags]
    
    def encode_pos(self, pos_tags):
        """Convert POS tags to indices"""
        return [self.pos2idx.get(pos, 1) for pos in pos_tags]
    
    def decode_tags(self, tag_indices):
        """Convert tag indices back to tags"""
        return [self.idx2tag.get(idx, "<PAD>") for idx in tag_indices]
    
    def pad_sequence(self, sequences, max_len):
        """Pad sequences to max length"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded.append(seq)
        return padded

# ============================================================================
# 3. DATASET CLASS
# ============================================================================

class NERDataset(Dataset):
    """PyTorch Dataset for NER"""
    
    def __init__(self, sentences, labels, pos_tags):
        self.sentences = sentences
        self.labels = labels
        self.pos_tags = pos_tags
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            'sentence': torch.tensor(self.sentences[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'pos': torch.tensor(self.pos_tags[idx], dtype=torch.long)
        }

# ============================================================================
# 4. BiLSTM-CRF MODEL
# ============================================================================

class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF model for sequence tagging"""
    
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, 
                           batch_first=True, dropout=0.3)
        
        # Linear layer to project to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
        
        # Initialize transitions
        self.transitions.data[0, :] = -10000  # No transition to PAD
        self.transitions.data[:, 0] = -10000  # No transition from PAD
        
    def forward(self, sentences, labels=None):
        """Forward pass"""
        # Get LSTM features
        embeds = self.word_embeds(sentences)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            # Training: return negative log-likelihood
            return self._neg_log_likelihood(emissions, labels)
        else:
            # Inference: return best path
            return self._viterbi_decode(emissions)
    
    def _neg_log_likelihood(self, emissions, tags):
        """Calculate negative log-likelihood for training"""
        batch_size, seq_len, tag_size = emissions.size()
        
        # Calculate score of the correct path
        score = torch.zeros(batch_size).to(emissions.device)
        
        for i in range(batch_size):
            emission = emissions[i]
            tag = tags[i]
            
            # Emission scores
            for j in range(seq_len):
                if tag[j].item() != 0:  # Not padding
                    score[i] += emission[j, tag[j]]
                    if j > 0 and tag[j-1].item() != 0:
                        score[i] += self.transitions[tag[j-1], tag[j]]
        
        # Calculate partition function (all possible paths)
        forward_var = torch.full((batch_size, tag_size), -10000.).to(emissions.device)
        forward_var[:, 0] = 0.  # Start from PAD
        
        for i in range(seq_len):
            emit_score = emissions[:, i, :].unsqueeze(2)
            trans_score = self.transitions.unsqueeze(0)
            next_tag_var = forward_var.unsqueeze(1) + trans_score + emit_score
            forward_var = torch.logsumexp(next_tag_var, dim=2)
        
        partition = torch.logsumexp(forward_var, dim=1)
        
        return (partition - score).mean()
    
    def _viterbi_decode(self, emissions):
        """Viterbi algorithm for decoding best path"""
        batch_size, seq_len, tag_size = emissions.size()
        
        # Initialize
        viterbi = torch.full((batch_size, seq_len, tag_size), -10000.).to(emissions.device)
        viterbi[:, 0, :] = emissions[:, 0, :]
        backpointers = torch.zeros(batch_size, seq_len, tag_size, dtype=torch.long).to(emissions.device)
        
        # Forward pass
        for i in range(1, seq_len):
            for t in range(tag_size):
                scores = viterbi[:, i-1, :] + self.transitions[:, t].unsqueeze(0) + emissions[:, i, t].unsqueeze(1)
                viterbi[:, i, t], backpointers[:, i, t] = torch.max(scores, dim=1)
        
        # Backward pass to find best path
        best_paths = []
        for b in range(batch_size):
            best_path = [torch.argmax(viterbi[b, -1, :]).item()]
            for i in range(seq_len - 1, 0, -1):
                best_path.insert(0, backpointers[b, i, best_path[0]].item())
            best_paths.append(best_path)
        
        return best_paths

# ============================================================================
# 5. TRAINING MODULE
# ============================================================================

class ModelTrainer:
    """Handles model training"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            sentences = batch['sentence'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            loss = self.model(sentences, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sentences = batch['sentence'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(sentences, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Full training loop"""
        print("\n🏋️ Training BiLSTM-CRF model...")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        print("✓ Training completed!")

# ============================================================================
# 6. EVALUATION MODULE
# ============================================================================

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, preprocessor, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        
    def predict(self, sentences):
        """Predict tags for sentences"""
        self.model.eval()
        
        encoded = [self.preprocessor.encode_sentence(sent) for sent in sentences]
        max_len = max(len(s) for s in encoded)
        padded = self.preprocessor.pad_sequence(encoded, max_len)
        
        tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # Decode predictions
        decoded_preds = []
        for i, pred in enumerate(predictions):
            pred_tags = self.preprocessor.decode_tags(pred[:len(sentences[i])])
            decoded_preds.append(pred_tags)
        
        return decoded_preds
    
    def evaluate_dataset(self, dataloader):
        """Evaluate on entire dataset"""
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                sentences = batch['sentence'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(sentences)
                
                for pred, label in zip(predictions, labels):
                    all_preds.extend(pred)
                    all_labels.extend(label.cpu().numpy())
        
        # Remove padding
        filtered_preds = []
        filtered_labels = []
        for p, l in zip(all_preds, all_labels):
            if l != 0:
                filtered_preds.append(p)
                filtered_labels.append(l)
        
        return filtered_preds, filtered_labels
    
    def print_metrics(self, predictions, labels):
        """Print evaluation metrics"""
        print("\n📊 Model Evaluation Metrics:")
        print("=" * 60)
        
        # Accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        unique_labels = sorted(set(labels))
        label_names = [self.preprocessor.idx2tag[l] for l in unique_labels]
        
        print("\nClassification Report:")
        print(classification_report([self.preprocessor.idx2tag[l] for l in labels],
                                   [self.preprocessor.idx2tag[p] for p in predictions],
                                   zero_division=0))

# ============================================================================
# 7. PREDICTION MODULE
# ============================================================================

class NERPredictor:
    """Handle predictions on custom text"""
    
    def __init__(self, model, preprocessor, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.evaluator = ModelEvaluator(model, preprocessor, device)
        
    def predict_text(self, text):
        """Predict entities in text"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Predict
        predictions = self.evaluator.predict([tokens])
        
        return list(zip(tokens, predictions[0]))
    
    def visualize_entities(self, text):
        """Visualize entities in text"""
        results = self.predict_text(text)
        
        print("\n🔍 Named Entity Recognition Results:")
        print("=" * 60)
        print(f"Input: {text}\n")
        print("Token".ljust(20) + "Entity Type")
        print("-" * 60)
        
        for token, tag in results:
            if tag != 'O' and tag != '<PAD>':
                print(f"{token.ljust(20)} {tag}")
        
        print("=" * 60)

# ============================================================================
# 8. VISUALIZATION MODULE
# ============================================================================

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BiLSTM-CRF Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training plot saved as 'training_history.png'")
    plt.show()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 60)
    print("STARTING BiLSTM-CRF NER PIPELINE")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load Data
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)
    
    loader = CoNLLDataLoader()
    
    # Try to load real CoNLL data, else use sample
    if not loader.load_conll_format('train.txt'):
        loader.create_sample_data()
    
    # 2. Preprocess
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(loader.sentences, loader.labels, loader.pos_tags)
    
    # Encode data
    encoded_sentences = [preprocessor.encode_sentence(s) for s in loader.sentences]
    encoded_labels = [preprocessor.encode_tags(l) for l in loader.labels]
    encoded_pos = [preprocessor.encode_pos(p) for p in loader.pos_tags]
    
    # Pad sequences
    max_len = max(len(s) for s in encoded_sentences)
    padded_sentences = preprocessor.pad_sequence(encoded_sentences, max_len)
    padded_labels = preprocessor.pad_sequence(encoded_labels, max_len)
    padded_pos = preprocessor.pad_sequence(encoded_pos, max_len)
    
    # Split data
    split = int(0.8 * len(padded_sentences))
    train_data = NERDataset(padded_sentences[:split], padded_labels[:split], padded_pos[:split])
    val_data = NERDataset(padded_sentences[split:], padded_labels[split:], padded_pos[split:])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Validation samples: {len(val_data)}")
    
    # 3. Build Model
    print("\n" + "=" * 60)
    print("STEP 3: MODEL ARCHITECTURE")
    print("=" * 60)
    
    model = BiLSTM_CRF(
        vocab_size=preprocessor.vocab_size,
        tag_size=preprocessor.tag_size,
        embedding_dim=100,
        hidden_dim=128
    )
    
    print(f"✓ Model created")
    print(f"  - Embedding dim: 100")
    print(f"  - Hidden dim: 128")
    print(f"  - Vocab size: {preprocessor.vocab_size}")
    print(f"  - Tag size: {preprocessor.tag_size}")
    
    # 4. Train Model
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)
    
    trainer = ModelTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=20, lr=0.001)
    
    # 5. Evaluate
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)
    
    evaluator = ModelEvaluator(model, preprocessor, device)
    predictions, labels = evaluator.evaluate_dataset(val_loader)
    evaluator.print_metrics(predictions, labels)
    
    # 6. Visualize
    print("\n" + "=" * 60)
    print("STEP 6: VISUALIZATION")
    print("=" * 60)
    
    plot_training_history(trainer.history)
    
    # 7. Test on custom text
    print("\n" + "=" * 60)
    print("STEP 7: CUSTOM TEXT PREDICTION")
    print("=" * 60)
    
    predictor = NERPredictor(model, preprocessor, device)
    
    test_sentences = [
        "Apple Inc. is planning to open new stores in California.",
        "Steve Jobs was the CEO of Apple.",
        "Microsoft announced new products in Seattle."
    ]
    
    for sent in test_sentences:
        predictor.visualize_entities(sent)
    
    # 8. Save model
    print("\n" + "=" * 60)
    print("STEP 8: SAVING MODEL")
    print("=" * 60)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': preprocessor,
    }, 'bilstm_crf_model.pth')
    
    print("✓ Model saved as 'bilstm_crf_model.pth'")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, preprocessor, predictor

if __name__ == "__main__":
    model, preprocessor, predictor = main()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to analyze (or 'quit' to exit):")
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_input.strip():
            predictor.visualize_entities(user_input)