import torch
import torch.nn as nn

class CustomEdgeFilter(nn.Module):
    def __init__(self):
        super(CustomEdgeFilter, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        sobel_kernel = torch.tensor([[-1.0, 0.0, 1.0],
                                     [-2.0, 0.0, 2.0],
                                     [-1.0, 0.0, 1.0]])
        
        self.conv.weight = nn.Parameter(sobel_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return self.conv(x)

detector = CustomEdgeFilter()
mock_image = torch.ones(1, 1, 5, 5) * 10.0
mock_image[:, :, :, 3:] = 255.0

output = detector(mock_image)
print("--- Extracted Vertical Edge Map ---")
print(output.squeeze().detach().numpy())
