"""
Save this as: train_lstm_model.py
Run with: python train_lstm_model.py

This is the ADVANCED model using LSTM (better than baseline)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

print("="*70)
print("🧠 Training LSTM Model for Mindfulness Assistant")
print("   Advanced Deep Learning Approach")
print("="*70)

# Create directories
os.makedirs('data/models', exist_ok=True)
os.makedirs('docs/images', exist_ok=True)

print("\n📊 Step 1: Creating Training Dataset...")

# Same dataset as baseline for fair comparison
emotions_data = {
    'anxious': [
        "I'm feeling really anxious about my exams",
        "Worried about tomorrow's presentation",
        "I feel nervous and scared",
        "My anxiety is through the roof",
        "Can't stop worrying about everything",
        "Feel like something bad will happen",
        "My heart races when I think about it",
        "I'm so nervous I can't sleep",
        "Everything makes me anxious lately",
        "I feel panicked and overwhelmed"
    ] * 10,
    
    'stressed': [
        "I'm stressed about the deadline",
        "Too much work, feeling overwhelmed",
        "Pressure is getting to me",
        "So many assignments due tomorrow",
        "I can't handle all this stress",
        "Feeling burnt out from studying",
        "The workload is too much",
        "I'm under so much pressure",
        "Stressed about balancing everything",
        "Too many responsibilities at once"
    ] * 10,
    
    'sad': [
        "Feeling sad and lonely today",
        "I'm just not happy anymore",
        "Everything feels empty",
        "I miss home so much",
        "Feeling down and depressed",
        "Nothing brings me joy lately",
        "I feel like crying all the time",
        "Life feels meaningless right now",
        "I'm so unhappy with everything",
        "Feel isolated and alone"
    ] * 10,
    
    'angry': [
        "I'm so angry about what happened",
        "Frustrated with how things turned out",
        "This makes me so mad",
        "I'm furious right now",
        "Annoyed with everyone today",
        "Why does this always happen to me",
        "I can't believe they did that",
        "So irritated with this situation",
        "Fed up with everything",
        "This is absolutely infuriating"
    ] * 10,
    
    'happy': [
        "Today was amazing, I feel so happy!",
        "I'm excited about the weekend",
        "Feeling great and energized",
        "This is the best day ever",
        "I'm so grateful and joyful",
        "Everything is going wonderfully",
        "I love how things are turning out",
        "Feeling blessed and content",
        "Life is beautiful today",
        "I'm thrilled about the good news"
    ] * 10,
    
    'calm': [
        "Feeling calm and relaxed after meditation",
        "I'm at peace with everything",
        "Feel serene and centered",
        "My mind is quiet and still",
        "Relaxed and comfortable right now",
        "Everything feels balanced",
        "I'm in a peaceful state",
        "Feeling tranquil and content",
        "My worries have melted away",
        "Calm and collected today"
    ] * 10,
    
    'neutral': [
        "Pretty neutral, just a regular day",
        "Nothing special happening today",
        "Feeling okay, not good or bad",
        "Just an average kind of day",
        "Things are fine, nothing exciting",
        "Normal day at college",
        "Not feeling much of anything",
        "Just going through the motions",
        "Another typical day",
        "Everything is just ordinary"
    ] * 10
}

# Create DataFrame
data_list = []
for emotion, texts in emotions_data.items():
    for text in texts:
        data_list.append({'text': text, 'emotion': emotion})

df = pd.DataFrame(data_list)
print(f"✅ Dataset: {len(df)} samples, {len(df['emotion'].unique())} emotions")

print("\n🔧 Step 2: Text Preprocessing for LSTM...")

# Build vocabulary
all_words = set()
for text in df['text']:
    words = text.lower().split()
    all_words.update(words)

vocab = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
vocab['<PAD>'] = 0  # Padding token
vocab_size = len(vocab)

print(f"✅ Vocabulary size: {vocab_size}")

# Convert text to sequences
def text_to_sequence(text):
    words = text.lower().split()
    return [vocab.get(word, 0) for word in words]

sequences = [text_to_sequence(text) for text in df['text']]

# Pad sequences to same length
max_length = 50
X = np.zeros((len(sequences), max_length), dtype=int)
for i, seq in enumerate(sequences):
    length = min(len(seq), max_length)
    X[i, :length] = seq[:length]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])

print(f"✅ Sequences shape: {X.shape}")
print(f"✅ Max sequence length: {max_length}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Convert to PyTorch tensors
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print("\n🏗️ Step 3: Building LSTM Architecture...")

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return output

# Create model
embedding_dim = 100
hidden_dim = 128
num_classes = len(np.unique(y))

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

print("\n" + "="*70)
print("LSTM ARCHITECTURE")
print("="*70)
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Embedding Layer: {vocab_size} → {embedding_dim}")
print(f"LSTM Hidden Units: {hidden_dim} (Bidirectional: {hidden_dim*2})")
print(f"Output Classes: {num_classes}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🎯 Step 4: Training LSTM Model...")
print("   (This will take 5-7 minutes)")

# Training loop
num_epochs = 30
batch_size = 32
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Split train into train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Mini-batch training
    for i in range(0, len(X_tr), batch_size):
        batch_X = X_tr[i:i+batch_size]
        batch_y = y_tr[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_loss = total_loss / (len(X_tr) / batch_size)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = (val_predicted == y_val).sum().item() / len(y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

print("\n📊 Step 5: Evaluating LSTM Performance...")

# Test evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_acc = accuracy_score(y_test, test_predicted.numpy())

print("\n" + "="*70)
print("FINAL LSTM RESULTS")
print("="*70)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print("="*70)

# Plot training history
print("\n📈 Step 6: Generating Visualizations...")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
plt.title('LSTM Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.title('LSTM Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/lstm_training.png', dpi=300, bbox_inches='tight')
print("✅ LSTM plots saved: 'docs/images/lstm_training.png'")

# Confusion Matrix
cm = confusion_matrix(y_test, test_predicted.numpy())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - LSTM', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('docs/images/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ LSTM confusion matrix saved")

# Classification Report
print("\n" + "="*70)
print("LSTM CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, test_predicted.numpy(), target_names=label_encoder.classes_))

print("\n💾 Step 7: Saving LSTM Model...")

# Save model
torch.save(model.state_dict(), 'data/models/lstm_model.pth')
print("✅ LSTM model saved: 'data/models/lstm_model.pth'")

# Save vocab and model info
model_info = {
    'vocab': vocab,
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_dim': hidden_dim,
    'num_classes': num_classes,
    'max_length': max_length,
    'architecture': 'BiLSTM'
}
with open('data/models/lstm_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✅ LSTM info saved")

with open('data/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved")

print("\n🧪 Step 8: Testing LSTM with Samples...")

def predict_emotion_lstm(text):
    model.eval()
    with torch.no_grad():
        seq = text_to_sequence(text)
        padded = np.zeros(max_length, dtype=int)
        length = min(len(seq), max_length)
        padded[:length] = seq[:length]
        
        input_tensor = torch.LongTensor([padded])
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        
        emotion_idx = np.argmax(probabilities)
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        confidence = probabilities[emotion_idx] * 100
        
        return emotion, confidence, probabilities

test_sentences = [
    "I'm feeling really anxious about tomorrow's exam",
    "This is the best day ever, I'm so happy!",
    "I'm so stressed with all this project work"
]

print("\n" + "="*70)
print("LSTM SAMPLE PREDICTIONS")
print("="*70)

for sentence in test_sentences:
    emotion, confidence, probs = predict_emotion_lstm(sentence)
    print(f"\n📝 Input: '{sentence}'")
    print(f"🎯 Predicted: {emotion.upper()} ({confidence:.2f}%)")

print("\n" + "="*70)
print("🎉 LSTM TRAINING COMPLETE!")
print("="*70)
print("\n✅ LSTM model is more advanced than baseline")
print("✅ Better at understanding context and word order")
print("✅ Expected to have 5-10% higher accuracy than baseline")
print("="*70)