import numpy as np
import csv

def generate_data_to_csv(filename='data.csv', m=1_000_000, n=30, seed=53, noise_level=0.1, flip_prob=0.05):
    print("seed:", seed)
    print("noise_level:", noise_level)
    print("flip_prob:", flip_prob)
    print("m:", m)
    print("n:", n)
    print("filename:", filename)
    
    np.random.seed(seed)
     
    X = np.random.randn(m, n)
    
    true_weights = np.random.randn(n)
    true_bias = np.random.randn()
    
    y_model = X @ true_weights + true_bias
    
    probs = 1 / (1 + np.exp(-y_model))
    
    y = (probs >= 0.5).astype(int)
    
    X += np.random.normal(0, noise_level, X.shape)
    
    flip_mask = np.random.rand(m) < flip_prob
    y[flip_mask] = 1 - y[flip_mask]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f'x{i+1}' for i in range(n)] + ['y']
        writer.writerow(header)
        for i in range(m):
            writer.writerow(list(X[i]) + [y[i]])
    
    print(f"Saved to '{filename}'")
    print("Original parameters:")
    print("Weights:")
    for i, w in enumerate(true_weights):
        print(f"  w{i+1}: {w:.5f}")
    print(f"\nBias: {true_bias:.5f}")
print("Generating data with the following parameters:")
generate_data_to_csv(filename="data.csv", m=1_000_000, n=30, seed=344, noise_level=0.08, flip_prob=0.01)

