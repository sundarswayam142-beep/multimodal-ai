
import torch


camera_feed = torch.tensor([[
    [45.0, 50.0, 48.0, 52.0],
    [40.0, 240.0, 245.0, 41.0], 
    [42.0, 238.0, 250.0, 46.0],
    [39.0, 41.0, 44.0, 40.0]
]])

print(f"Initial Shape: {camera_feed.shape} | Data Typing: {camera_feed.dtype}")

scaled_tensor = camera_feed / 255.0
print("\n--- Scaled Input Vector Matrix ---")
print(scaled_tensor)
