import numpy as np

mock_database = {
    "Surface crack detected on steel sheet": np.array([0.15, 0.88, 0.03]),
    "Minor discoloration on casting shell": np.array([0.02, 0.11, 0.94]),
    "Major thermal leak around power unit": np.array([0.91, 0.05, 0.12])
}

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    return dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) > 0 else 0

new_anomaly_embedding = np.array([0.12, 0.85, 0.05])
print("--- Vector Matching Query Results ---")

for log_text, stored_embedding in mock_database.items():
    similarity = calculate_cosine_similarity(new_anomaly_embedding, stored_embedding)
    print(f"Match Score: {similarity:.4f} | Source: {log_text}")
