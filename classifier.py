import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def main():
    print("========================================")
    print("  DecodeLabs AI - Project 2 Pipeline")
    print("  Supervised Learning: Data Classification")
    print("========================================\n")

    # 1. INPUT: Raw Material - The Iris Benchmark (Slide 8)
    print("[1] Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"    Samples: {X.shape[0]} | Classes: {len(np.unique(y))} | Dimensions: {X.shape[1]}")
    
    # 2. SCALING: The Gatekeeper Rule (Slide 9)
    print("\n[2] Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("    StandardScaler applied: Mean ~ 0, Variance ~ 1.")

    # 3. STRUCTURAL INTEGRITY: The Split (Slide 10 & 17)
    # 80% Training, 20% Testing
    print("\n[3] Splitting Data into Training and Test Sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Training Set: {X_train.shape[0]} samples (Pattern Recognition)")
    print(f"    Test Set: {X_test.shape[0]} samples (Validation)")

    # 4. PROCESS: The Algorithm (K-Nearest Neighbors) (Slide 11 & 13)
    print("\n[4] Instantiating and Training KNN Model...")
    # K=5 as specified in Slide 11 & 13
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print("    Model instantiated and fit to training data.")

    # 5. PREDICT: Applying Logic (Slide 13)
    print("\n[5] Making Predictions on Test Set...")
    predictions = model.predict(X_test)
    
    # 6. OUTPUT VALIDATION: Diagnostic Tools (Slide 14, 15, 16)
    print("\n[6] Output Validation (Avoiding the Accuracy Mirage)...")
    
    acc = accuracy_score(y_test, predictions)
    print(f"    Accuracy: {acc * 100:.2f}%")
    
    print("\n    Confusion Matrix (Slide 15):")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # F1 Score (Harmonic Mean of Precision and Recall) (Slide 16)
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f"\n    F1 Score (Weighted): {f1:.4f}")
    print("\nProject 2 pipeline execution complete.")

if __name__ == '__main__':
    main()
