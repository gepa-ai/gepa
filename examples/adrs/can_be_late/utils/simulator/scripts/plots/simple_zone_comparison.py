import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def load_trace_data(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as f:
        data = json.load(f)
        return np.array(data['data'])

def main():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 10))
    
    single_zone_file = pathlib.Path('data/real/16-nodes/us-west-2a_v100_1.json')
    single_zone_data = load_trace_data(str(single_zone_file))
    
    single_total = 16
    single_binary = np.sum((single_zone_data == 0) | (single_zone_data == 16))
    single_binary_ratio = single_binary / len(single_zone_data)
    
    multi_zone_files = [
        'data/real/4-nodes-real-preemption/us-east-1f_v100_1.json',
        'data/real/4-nodes-real-preemption/us-east-2a_v100_1.json',
        'data/real/4-nodes-real-preemption/us-west-2c_v100_1.json'
    ]
    
    zone_data = []
    for f in multi_zone_files:
        data = load_trace_data(f)
        zone_data.append(data)
    
    min_len = min(len(d) for d in zone_data)
    zone_data = [d[:min_len] for d in zone_data]
    
    total_available = np.sum(zone_data, axis=0)
    multi_total = 12  # Three zones × four nodes each
    multi_binary = np.sum((total_available == 0) | (total_available == 12))
    multi_binary_ratio = multi_binary / len(total_available)
    
    ax1 = plt.subplot(2, 3, 1)
    scenarios = ['Single Zone\n(16 nodes)', 'Cross Zones\n(3×4 nodes)']
    binary_ratios = [single_binary_ratio * 100, multi_binary_ratio * 100]
    colors = ['#ff4444', '#44ff44']
    
    bars = ax1.bar(scenarios, binary_ratios, color=colors, alpha=0.7, width=0.6)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('All-or-Nothing Probability\n(Higher = More Risky)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)  # Increased to give more space for text
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, ratio in zip(bars, binary_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    unique, counts = np.unique(single_zone_data, return_counts=True)
    ax2.bar(unique, counts/len(single_zone_data)*100, color='#ff4444', alpha=0.7)
    ax2.set_xlabel('Available Nodes', fontsize=11)
    ax2.set_ylabel('Frequency (%)', fontsize=11)
    ax2.set_title('Single Zone Distribution\n(Extreme Values)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.5, 16.5)
    
    ax3 = plt.subplot(2, 3, 3)
    unique, counts = np.unique(total_available, return_counts=True)
    ax3.bar(unique, counts/len(total_available)*100, color='#44ff44', alpha=0.7)
    ax3.set_xlabel('Available Nodes', fontsize=11)
    ax3.set_ylabel('Frequency (%)', fontsize=11)
    ax3.set_title('Cross-Zone Distribution\n(More Balanced)', fontsize=12, fontweight='bold')
    ax3.set_xlim(-0.5, 12.5)
    
    ax4 = plt.subplot(2, 3, (4, 5))
    time_points = 200
    time_range = range(time_points)
    
    single_normalized = single_zone_data[:time_points] / 16
    multi_normalized = total_available[:time_points] / 12
    
    ax4.plot(time_range, single_normalized, 'r-', alpha=0.7, linewidth=1.5, label='Single Zone (16 nodes)')
    ax4.plot(time_range, multi_normalized, 'g-', alpha=0.7, linewidth=1.5, label='Cross Zones (3×4 nodes)')
    
    ax4.set_xlabel('Time (5-minute intervals)', fontsize=11)
    ax4.set_ylabel('Availability Ratio (0=none, 1=all)', fontsize=11)
    ax4.set_title('Availability Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)
    
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    plt.suptitle('Why Cross-Zone Deployment Matters: Single Zone vs Multi-Zone Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = 'simple_zone_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("Key Findings:")
    print("="*60)
    print(f"Single Zone (16 nodes): {single_binary_ratio:.1%} of time: ALL or NONE")
    print(f"Cross Zones (3×4 nodes): {multi_binary_ratio:.1%} of time: ALL or NONE")
    print(f"Risk Reduction: {(single_binary_ratio - multi_binary_ratio)/single_binary_ratio:.1%}")

if __name__ == "__main__":
    main()
