import pandas as pd
import matplotlib.pyplot as plt

# Read the first 1000 rows of data.csv
data = pd.read_csv('data1.csv', nrows=1000)

# Extract x1, x2, and y
x1 = data['x1']
x2 = data['x2']
y = data['y']

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x1[y == 0], x2[y == 0], c='blue', label='y=0', alpha=0.6, s=50)
plt.scatter(x1[y == 1], x2[y == 1], c='red', label='y=1', alpha=0.6, s=50)
plt.title('Диаграмма рассеяния')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png')
plt.close()

# Create histogram
plt.figure(figsize=(10, 6))
# Histogram for x1
plt.subplot(1, 2, 1)
plt.hist(x1[y == 0], bins=30, alpha=0.5, color='blue', label='y=0', density=True)
plt.hist(x1[y == 1], bins=30, alpha=0.5, color='red', label='y=1', density=True)
plt.title('Гистограмма x1 по меткам')
plt.xlabel('x1')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

# Histogram for x2
plt.subplot(1, 2, 2)
plt.hist(x2[y == 0], bins=30, alpha=0.5, color='blue', label='y=0', density=True)
plt.hist(x2[y == 1], bins=30, alpha=0.5, color='red', label='y=1', density=True)
plt.title('Гистограмма x2 по меткам')
plt.xlabel('x2')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram.png')
plt.close()