import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filepath = "outputs/evaluation_results.csv"
try:
    df = pd.read_csv(filepath)
except FileNotFoundError as e:
    print("Error: Run evaluate.py first to generate the CSV.")
    exit()

# Metrics to be plotted
metrics = ['MAP', 'nDCG@10', 'Recall@10']

x = np.arange(len(df['System']))
width = 0.25  # Thin bars to fit 3 metrics
fig, ax = plt.subplots(figsize=(12, 6))

# Dynamic Plotting Loop
for i, metric in enumerate(metrics):
    # This calculates where to put the bar so they stand side-by-side rather than on top of each other.
    # It shifts the bars slightly left or right based on which metric it is.
    position = x + (i * width) - (width * len(metrics) / 2) + (width / 2)

    bars = ax.bar(position, df[metric], width, label=metric)

    # Add Value Labels on top of bars
    for bar in bars:
        height = bar.get_height()

        # This acts like a label maker. It reads the height of the bar (e.g., 0.763) and writes that
        # number directly on top of the bar so you don't have to guess the value by looking at the axis.
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

# Formatting
ax.set_ylabel('Score')
ax.set_title('Performance Comparison: BM25 vs Dense vs LLMs')
ax.set_xticks(x)
ax.set_xticklabels(df['System'])
ax.set_ylim(0, 1.1)  # Sets the graph height slightly above 1.0 (110%) to make room for the text labels.
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/results_chart.png', dpi=300)
print(f"Chart saved at: {'outputs/results_chart.png'}")