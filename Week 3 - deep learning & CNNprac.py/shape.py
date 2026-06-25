import numpy as np

imageGrid = np.array([,
 ,
 ,
    [0, 10, 0, 0]
])

kernel = np.array([,
    [0, -1]
])

def run_manual_conv2d(matrix, filter_kernel, stride=1):
    m_h, m_w = matrix.shape
    k_h, k_w = filter_kernel.shape
    
    out_h = int((m_h - k_h) / stride) + 1
    out_w = int((m_w - k_w) / stride) + 1
    output = np.zeros((out_h, out_w))
    
    for r in range(out_h):
        for c in range(out_w):
            r_start = r * stride
            c_start = c * stride
            region = matrix[r_start:r_start+k_h, c_start:c_start+k_w]
            output[r, c] = np.sum(region * filter_kernel)
            
    return output

feature_map = run_manual_conv2d(imageGrid, kernel, stride=1)
print("--- Calculated Spatial Output Feature Grid ---")
print(feature_map)
