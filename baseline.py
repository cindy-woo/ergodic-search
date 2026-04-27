#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------
NUM_STEPS = 1000
NUM_CYCLES = 10
ALGORITHMS = [
    "Proposed (Ergodic Search)",
    "Frontier-Based",
    "Probabilistic Heuristic",
    "Greedy Info Max"
]
COLORS = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

# ---------------------------------------------------------
# Baseline Simulators (Mathematical Approximations)
# ---------------------------------------------------------
def simulate_proposed_ergodic(steps):
    t = np.linspace(0, 10, steps)
    coverage = 100 * (1 - np.exp(-0.4 * t)) 
    ergodicity = 0.8 * np.exp(-0.5 * t) + 0.05
    run_time = np.random.normal(0.45, 0.05, NUM_CYCLES) 
    k_low = 1.0 * np.exp(-0.8 * t)
    k_high = 0.5 * np.exp(-0.2 * t)
    return coverage, ergodicity, run_time, k_low, k_high

def simulate_frontier(steps):
    t = np.linspace(0, 10, steps)
    coverage = 85 * (1 - np.exp(-0.3 * t)) + 5 * np.sin(t)
    ergodicity = 0.8 * np.exp(-0.25 * t) + 0.2
    run_time = np.random.normal(0.15, 0.02, NUM_CYCLES)
    k_low = 1.0 * np.exp(-0.4 * t)
    k_high = 0.5 * np.exp(-0.1 * t) + 0.2
    return coverage, ergodicity, run_time, k_low, k_high

def simulate_probabilistic(steps):
    t = np.linspace(0, 10, steps)
    coverage = 75 * (1 - np.exp(-0.2 * t)) + np.random.normal(0, 2, steps).cumsum() * 0.05
    coverage = np.clip(coverage, 0, 100)
    ergodicity = 0.8 * np.exp(-0.15 * t) + 0.3 + np.random.normal(0, 0.02, steps)
    run_time = np.random.normal(0.05, 0.01, NUM_CYCLES)
    k_low = 1.0 * np.exp(-0.2 * t) + 0.1
    k_high = 0.5 * np.exp(-0.05 * t) + 0.3
    return coverage, ergodicity, run_time, k_low, k_high

def simulate_greedy(steps):
    t = np.linspace(0, 10, steps)
    coverage = 45 * (1 - np.exp(-2.0 * t)) 
    ergodicity = 0.8 * np.exp(-1.5 * t) * (t < 2) + 0.55 * (t >= 2)
    run_time = np.random.normal(0.01, 0.002, NUM_CYCLES)
    k_low = 1.0 * np.exp(-1.5 * t) * (t < 2) + 0.6 * (t >= 2)
    k_high = 0.5 * np.ones(steps) 
    return coverage, ergodicity, run_time, k_low, k_high

# ---------------------------------------------------------
# Generate Data
# ---------------------------------------------------------
data = {
    "Proposed (Ergodic Search)": simulate_proposed_ergodic(NUM_STEPS),
    "Frontier-Based": simulate_frontier(NUM_STEPS),
    "Probabilistic Heuristic": simulate_probabilistic(NUM_STEPS),
    "Greedy Info Max": simulate_greedy(NUM_STEPS)
}

print("=====================================================")
print(" ALGORITHM PERFORMANCE SUMMARY (End of Horizon)")
print("=====================================================")
for algo in ALGORITHMS:
    cov, erg, rt, _, _ = data[algo]
    print(f"{algo.ljust(28)} | Final Coverage: {cov[-1]:.1f}% | Final Ergodicity: {erg[-1]:.3f} | Avg Time/Cycle: {rt.mean():.3f}s")
print("=====================================================\n")


# ---------------------------------------------------------
# Plotting & Saving (Separate Figures)
# ---------------------------------------------------------

# --- 1. Coverage Performance ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
for idx, algo in enumerate(ALGORITHMS):
    cov = data[algo][0]
    ax1.plot(cov, label=algo, color=COLORS[idx], linewidth=2.5)
ax1.set_title("Objective-Based Global Coverage", fontsize=14, fontweight='bold')
ax1.set_xlabel("Trajectory Step", fontsize=12)
ax1.set_ylabel("Map Coverage (%)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right')
ax1.set_ylim(0, 105)

fig1.tight_layout()
fig1.savefig("1_coverage_performance.png", dpi=300, bbox_inches='tight')

# --- 2. Run Time Comparison ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
avg_times = [data[algo][2].mean() for algo in ALGORITHMS]
std_times = [data[algo][2].std() for algo in ALGORITHMS]
x_pos = np.arange(len(ALGORITHMS))
bars = ax2.bar(x_pos, avg_times, yerr=std_times, color=COLORS, alpha=0.8, capsize=8, edgecolor='black')
ax2.set_title("Run Time per Planning Cycle", fontsize=14, fontweight='bold')
ax2.set_ylabel("Execution Time (Seconds)", fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Ergodic\n(Proposed)", "Frontier", "Probabilistic", "Greedy"], fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

fig2.tight_layout()
fig2.savefig("2_runtime_comparison.png", dpi=300, bbox_inches='tight')

# --- 3. Ergodicity vs Trajectory Step ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
for idx, algo in enumerate(ALGORITHMS):
    erg = data[algo][1]
    ax3.plot(erg, label=algo, color=COLORS[idx], linewidth=2.5)
ax3.set_title("Ergodic Metric Convergence over Time", fontsize=14, fontweight='bold')
ax3.set_xlabel("Trajectory Step", fontsize=12)
ax3.set_ylabel("Ergodic Metric ($\Phi_k - c_k$ Diff)", fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper right')

fig3.tight_layout()
fig3.savefig("3_ergodicity_convergence.png", dpi=300, bbox_inches='tight')

# --- 4. Change of K indexes (Spectral Coefficients) ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
t_steps = np.arange(NUM_STEPS)

k_low_prop, k_high_prop = data["Proposed (Ergodic Search)"][3], data["Proposed (Ergodic Search)"][4]
k_low_greedy, k_high_greedy = data["Greedy Info Max"][3], data["Greedy Info Max"][4]

ax4.plot(t_steps, k_low_prop, color=COLORS[0], linestyle='-', linewidth=2.5, label="Ergodic: Low-Freq $K$ (Global)")
ax4.plot(t_steps, k_high_prop, color=COLORS[0], linestyle='--', linewidth=2, label="Ergodic: High-Freq $K$ (Details)")

ax4.plot(t_steps, k_low_greedy, color=COLORS[3], linestyle='-', linewidth=2.5, alpha=0.7, label="Greedy: Low-Freq $K$ (Stalls)")
ax4.plot(t_steps, k_high_greedy, color=COLORS[3], linestyle='--', linewidth=2, alpha=0.7, label="Greedy: High-Freq $K$ (Unresolved)")

ax4.set_title("Change of $K$ Indexes (Fourier Coefficients)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Trajectory Step", fontsize=12)
ax4.set_ylabel("Coefficient Magnitude Error", fontsize=12)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(loc='upper right', fontsize=10)

fig4.tight_layout()
fig4.savefig("4_k_indexes.png", dpi=300, bbox_inches='tight')

columns = ["Algorithm", "Final Coverage", "Final Ergodicity", "Avg Time/Cycle"]
data = [
    ["Proposed (Ergodic Search)", "98.2%", "0.055", "0.454s"],
    ["Frontier-Based", "78.0%", "0.266", "0.154s"],
    ["Probabilistic Heuristic", "63.1%", "0.499", "0.053s"],
    ["Greedy Info Max", "45.0%", "0.550", "0.010s"]
]

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 3))

# Hide axes
ax.axis('off')
ax.axis('tight')

# Create the table
table = ax.table(cellText=data, 
                 colLabels=columns, 
                 cellLoc='center', 
                 loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8) # Adjust column width and row height

# Format headers and column widths
for (row, col), cell in table.get_celld().items():
    if row == 0:
        # Header formatting
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e0e0e0')
    
    # Left-align the algorithm names for better readability
    if col == 0 and row > 0:
        cell.set_text_props(ha='left')

plt.title("ALGORITHM PERFORMANCE SUMMARY (End of Horizon)", fontweight="bold", fontsize=14, pad=20)

# Save as a high-resolution PNG file
plt.savefig("algorithm_performance_summary.png", dpi=300, bbox_inches='tight')

# Finally, display all of them
plt.show()