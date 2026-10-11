import os
import matplotlib.pyplot as plt

os.makedirs('assets', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')

# 1. نمودار Recall به تفکیک اندازه
def plot_recall():
    sizes = ['Tiny (<20px)', 'Small (20-32px)', 'Medium (32-64px)', 'Large (>64px)']
    recalls = [1.000, 0.962, 0.984, 0.990]
    samples = [22, 52, 244, 767]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sizes, recalls, color='#d62728', alpha=0.8)
    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel('Recall Score', fontweight='bold')
    ax.set_title('Model Recall by Balloon Size (Validation Set)', fontweight='bold')

    for bar, sample in zip(bars, samples):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.002, 
                f'{y:.3f}\n(n={sample})', ha='center', va='bottom', fontsize=10)
    
    plt.savefig('assets/recall_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: assets/recall_benchmark.png")

# 2. نمودار خطای تخمین فاصله
def plot_distance_error():
    distances = [150, 250, 350, 500, 650, 800]
    errors = [-0.7, 2.0, -2.1, -5.9, -2.4, 9.0]

    plt.figure(figsize=(8, 5))
    plt.plot(distances, errors, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.fill_between(distances, -6, 6, color='gray', alpha=0.1, label='Acceptable Range (±6%)')
    plt.xlabel('True Distance (meters)', fontweight='bold')
    plt.ylabel('Estimation Error (%)', fontweight='bold')
    plt.title('Monocular Distance Estimation Accuracy', fontweight='bold')
    plt.legend()
    
    plt.savefig('assets/distance_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: assets/distance_error.png")

if __name__ == "__main__":
    plot_recall()
    plot_distance_error()