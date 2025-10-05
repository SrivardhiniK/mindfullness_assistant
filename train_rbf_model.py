"""
RBF Network Implementation (OLD MODEL from syllabus Unit 4)
Save as: train_rbf_model.py
Run: python train_rbf_model.py

This uses Radial Basis Function Networks (mentioned in your syllabus)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import os

print("="*70)
print("🧠 Training RBF Network (OLD MODEL from Syllabus Unit 4)")
print("="*70)

# Create dataset (same as before)
emotions_data = {
    'anxious': ["I'm feeling anxious", "Worried and nervous", "I feel scared"] * 30,
    'stressed': ["I'm stressed", "Too much pressure", "Feeling overwhelmed"] * 30,
    'sad': ["Feeling sad", "I'm unhappy", "Everything feels empty"] * 30,
    'angry': ["I'm so angry", "This frustrates me", "I'm furious"] * 30,
    'happy': ["I feel happy!", "This is amazing", "I'm excited"] * 30,
    'calm': ["Feeling calm", "I'm at peace", "Everything is balanced"] * 30,
    'neutral': ["Just regular", "Nothing special", "Feeling okay"] * 30
}

data_list = []
for emotion, texts in emotions_data.items():
    for text in texts:
        data_list.append({'text': text, 'emotion': emotion})

df = pd.DataFrame(data_list)
print(f"✅ Dataset: {len(df)} samples")

# Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data prepared: {X_train.shape}")

# RBF Network Implementation
class RBFNetwork:
    def __init__(self, num_centers, num_classes):
        self.num_centers = num_centers
        self.num_classes = num_classes
        self.centers = None
        self.sigmas = None
        self.weights = None
    
    def _gaussian_rbf(self, x, center, sigma):
        """Gaussian RBF kernel"""
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))
    
    def _calculate_activation(self, X):
        """Calculate RBF activations"""
        activations = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                activations[i, j] = self._gaussian_rbf(x, center, self.sigmas[j])
        return activations
    
    def fit(self, X, y):
        """Train RBF network"""
        print(f"\n🎯 Training RBF Network with {self.num_centers} centers...")
        
        # Step 1: Find RBF centers using K-Means clustering
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        # Step 2: Calculate sigma (spread) for each center
        self.sigmas = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            distances = np.linalg.norm(X - self.centers[i], axis=1)
            self.sigmas[i] = np.mean(distances)
        
        print(f"✅ RBF centers identified")
        print(f"✅ Sigma values calculated")
        
        # Step 3: Calculate RBF activations
        phi = self._calculate_activation(X)
        
        # Step 4: Train output layer using pseudo-inverse
        y_one_hot = np.zeros((y.shape[0], self.num_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        
        self.weights = np.linalg.pinv(phi) @ y_one_hot
        print(f"✅ Output weights trained")
    
    def predict(self, X):
        """Make predictions"""
        phi = self._calculate_activation(X)
        y_pred = phi @ self.weights
        return np.argmax(y_pred, axis=1)

# Train RBF Network
print("\n" + "="*70)
print("TRAINING RBF NETWORK")
print("="*70)

num_centers = 30  # Number of RBF centers
num_classes = len(np.unique(y))

rbf_net = RBFNetwork(num_centers=num_centers, num_classes=num_classes)
rbf_net.fit(X_train, y_train)

# Evaluate
y_pred_train = rbf_net.predict(X_train)
y_pred_test = rbf_net.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("\n" + "="*70)
print("RBF NETWORK RESULTS")
print("="*70)
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Save model
os.makedirs('data/models', exist_ok=True)

rbf_model_data = {
    'centers': rbf_net.centers,
    'sigmas': rbf_net.sigmas,
    'weights': rbf_net.weights,
    'num_centers': num_centers,
    'num_classes': num_classes
}

with open('data/models/rbf_model.pkl', 'wb') as f:
    pickle.dump(rbf_model_data, f)

with open('data/models/vectorizer_rbf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('data/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✅ RBF model saved: data/models/rbf_model.pkl")

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(['RBF Network'], [test_acc*100], color='#667eea', width=0.5)
plt.title('RBF Network Performance', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.text(0, test_acc*100 + 2, f'{test_acc*100:.2f}%', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('docs/images/rbf_results.png', dpi=300)
print("✅ Plot saved: docs/images/rbf_results.png")

print("\n" + "="*70)
print("RBF NETWORK TRAINING COMPLETE!")
print("="*70)
print("\n📚 RBF Networks (from Syllabus Unit 4):")
print("   • Uses Gaussian Radial Basis Functions")
print("   • K-Means clustering for center selection")
print("   • Pseudo-inverse for output weight training")
print("   • Suitable for classification problems")
print("="*70)