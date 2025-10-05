"""
Save this as: train_baseline_model_pytorch.py
Run with: python train_baseline_model_pytorch.py

This uses PyTorch instead of TensorFlow (more stable on Windows)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
from torch.utils.data import Dataset, DataLoader

print("="*70)
print("🧠 Training Baseline ANN Model for Mindfulness Assistant")
print("   Using PyTorch (More Stable Than TensorFlow)")
print("="*70)

# Create necessary directories
os.makedirs('data/models', exist_ok=True)
os.makedirs('docs/images', exist_ok=True)

print("\n📊 Step 1: Creating Training Dataset...")

# Extended sample dataset
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

print(f"✅ Created dataset with {len(df)} samples")
print(f"   Emotions: {list(df['emotion'].unique())}")
print(f"   Distribution:\n{df['emotion'].value_counts()}")

print("\n🔧 Step 2: Preprocessing Text...")

# Text preprocessing
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text']).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])

print(f"✅ Text vectorized: {X.shape}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print("\n🏗️ Step 3: Building ANN Architecture...")

# Define PyTorch Neural Network
class BaselineANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BaselineANN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x

# Create model
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = BaselineANN(input_dim, num_classes)

print("\n" + "="*70)
print("BASELINE ANN ARCHITECTURE (PyTorch)")
print("="*70)
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🎯 Step 4: Training the Model...")
print("   (This will take 2-3 minutes)")

# Training loop
num_epochs = 50
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
    # Training
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
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

print("\n📊 Step 5: Evaluating Performance...")

# Test evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_acc = accuracy_score(y_test, test_predicted.numpy())

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print("="*70)

# Plot training history
print("\n📈 Step 6: Generating Visualizations...")

plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Baseline ANN (PyTorch)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.title('Model Loss - Baseline ANN (PyTorch)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/images/baseline_ann_training.png', dpi=300, bbox_inches='tight')
print("✅ Training plots saved: 'docs/images/baseline_ann_training.png'")

# Confusion Matrix
cm = confusion_matrix(y_test, test_predicted.numpy())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Baseline ANN', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('docs/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: 'docs/images/confusion_matrix.png'")

# Classification Report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, test_predicted.numpy(), target_names=label_encoder.classes_))

print("\n💾 Step 7: Saving Model & Artifacts...")

# Save PyTorch model
torch.save(model.state_dict(), 'data/models/baseline_ann_pytorch.pth')
print("✅ PyTorch model saved: 'data/models/baseline_ann_pytorch.pth'")

# Save model architecture info
model_info = {
    'input_dim': input_dim,
    'num_classes': num_classes,
    'architecture': 'BaselineANN',
    'framework': 'PyTorch'
}
with open('data/models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✅ Model info saved: 'data/models/model_info.pkl'")

# Save vectorizer
with open('data/models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Vectorizer saved: 'data/models/vectorizer.pkl'")

# Save label encoder
with open('data/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved: 'data/models/label_encoder.pkl'")

print("\n🧪 Step 8: Testing with Sample Inputs...")

def predict_emotion(text):
    model.eval()
    with torch.no_grad():
        text_vectorized = vectorizer.transform([text]).toarray()
        text_tensor = torch.FloatTensor(text_vectorized)
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        emotion_idx = np.argmax(probabilities)
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        confidence = probabilities[emotion_idx] * 100
        
        return emotion, confidence, probabilities

test_sentences = [
    "I'm feeling really anxious about tomorrow's exam",
    "This is the best day ever, I'm so happy!",
    "I'm so stressed with all this project work",
    "Feeling calm and relaxed after yoga",
    "I'm so angry about what happened"
]

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

for sentence in test_sentences:
    emotion, confidence, all_probs = predict_emotion(sentence)
    print(f"\n📝 Input: '{sentence}'")
    print(f"🎯 Predicted: {emotion.upper()} (Confidence: {confidence:.2f}%)")
    print(f"📊 All probabilities:")
    for i, prob in enumerate(all_probs):
        if prob > 0.05:
            print(f"   {label_encoder.classes_[i]}: {prob*100:.2f}%")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print("\n✅ Next Steps:")
print("   1. Check visualizations in 'docs/images/' folder")
print("   2. Run Flask app: python frontend/web/app_with_model_pytorch.py")
print("   3. Open http://localhost:5000 to test the app")
print("\n💡 The app will now use your trained ANN model for predictions!")
print("="*70)