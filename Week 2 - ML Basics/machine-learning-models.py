# Simple Baseline Manual Classifier Logic
import numpy as np

# Features: [Anomalies Found, Surface Texture Score]
X_train = np.array([[0, 1.2], [4, 3.1], [1, 0.9], [6, 4.5], [0, 1.1]])
y_train = np.array([1, 0, 1, 0, 1])  # 1 = Pass, 0 = Fail

class ManualThresholdClassifier:
    """A minimal classifier simulating early decision tree splits."""
    def __init__(self, anomaly_limit=2):
        self.anomaly_limit = anomaly_limit
        
    def predict(self, features):
        # Fail if total anomalies cross our strict limit boundary
        return [1 if sample[0] < self.anomaly_limit else 0 for sample in features]

# Instantiate and check predictive outcome matches
clf = ManualThresholdClassifier(anomaly_limit=3)
predictions = clf.predict(X_train)

correct_hits = sum(1 for p, actual in zip(predictions, y_train) if p == actual)
print(f"Calculated Mock Classifier Accuracy: {(correct_hits / len(y_train)) * 100:.1f}%")
