import matplotlib.pyplot as plt
import pandas as pd

# Data from your final evaluation
data = {
    'System': ['BM25', 'S-BERT', 'Gemini (Zero)', 'Gemini (Few)'],
    'MAP': [0.7126, 0.5275, 0.7633, 0.7447],
    'nDCG@10': [0.7500, 0.5713, 0.8040, 0.7866]
}

df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))
width = 0.35

# Bars
ax.bar([i - width/2 for i in x], df['MAP'], width, label='MAP', color='#4c72b0')
ax.bar([i + width/2 for i in x], df['nDCG@10'], width, label='nDCG@10', color='#dd8452')

# Labels and Title
ax.set_ylabel('Score')
ax.set_title('Comparison of Retrieval Models on Turkish Law Dataset')
ax.set_xticks(x)
ax.set_xticklabels(df['System'])
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save
plt.savefig('results_chart.png', dpi=300)
print("Chart saved as 'results_chart.png'")