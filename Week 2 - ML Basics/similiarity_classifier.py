import numpy as np

training_features = np.array([[10, 5], [11, 4], [30, 25], [32, 28]])
training_labels = np.array(["pass", "pass", "fail", "fail"])

def predict_nearest_class(new_sample, features, labels):
    best_distance = float('inf')
    best_index = -1
    
    for i in range(len(features)):
        distance = np.sqrt(np.sum((new_sample - features[i]) ** 2))
        if distance < best_distance:
            best_distance = distance
            best_index = i
            
    return labels[best_index]

untested_sensor_point = np.array([12, 6])
prediction = predict_nearest_class(untested_sensor_point, training_features, training_labels)

print(f"Predicted quality category: {prediction}")
