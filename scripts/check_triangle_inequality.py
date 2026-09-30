"""
Script to check for triangle inequality violations in the travel time matrix.
"""

import pandas as pd
import os
import numpy as np

place_name = "Maastricht, Netherlands"
save_dir = os.path.join(os.getcwd(), place_name)
ttm_dir = os.path.join(save_dir, 'travel_time_matrix')

print("=" * 60)
print("CHECKING FOR TRIANGLE INEQUALITY VIOLATIONS")
print("=" * 60)

# Find the most recent travel time matrix file
import glob
ttm_files = glob.glob(os.path.join(ttm_dir, '*.csv'))

if not ttm_files:
    print("\nNo travel time matrix files found!")
    print(f"Looking in: {ttm_dir}")
    exit(1)

ttm_file = ttm_files[0]
print(f"\nLoading travel time matrix from: {os.path.basename(ttm_file)}")

# Load the matrix
ttm = pd.read_csv(ttm_file, index_col=0)
print(f"Matrix size: {len(ttm)} x {len(ttm.columns)}")

# Convert to numpy array for faster computation
matrix = ttm.values
n = len(matrix)

print(f"\nChecking triangle inequality for {n} nodes...")
print("This may take a moment...\n")

violations = []
sample_violations = []

# Check triangle inequality: d(i,k) <= d(i,j) + d(j,k) for all i,j,k
checked = 0
total_checks = n * n * n

for i in range(min(50, n)):  # Limit to first 50 nodes for speed
    if i % 10 == 0:
        print(f"Checking node {i}...")
    
    for j in range(n):
        for k in range(n):
            if i != j and j != k and i != k:
                t_ik = matrix[i][k]
                t_ij = matrix[i][j]
                t_jk = matrix[j][k]
                
                # Check if all values are valid
                if t_ik > 0 and t_ij > 0 and t_jk > 0:
                    if t_ik > t_ij + t_jk:
                        violation = {
                            'i': i, 'j': j, 'k': k,
                            't_ik': t_ik,
                            't_ij': t_ij,
                            't_jk': t_jk,
                            't_ij_jk': t_ij + t_jk,
                            'difference': t_ik - (t_ij + t_jk),
                            'percentage': ((t_ik - (t_ij + t_jk)) / t_ik) * 100
                        }
                        violations.append(violation)
                        
                        if len(sample_violations) < 10:
                            sample_violations.append(violation)
            
            checked += 1

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Total triangle inequality violations found: {len(violations)}")

if len(violations) > 0:
    print(f"\n❌ TRIANGLE INEQUALITY IS VIOLATED!")
    print(f"\nSample violations (showing up to 10):")
    print(f"{'='*60}")
    
    for idx, v in enumerate(sample_violations, 1):
        print(f"\nViolation #{idx}:")
        print(f"  Route: {v['i']} → {v['k']} (direct)")
        print(f"  Direct time: {v['t_ik']:.1f} seconds")
        print(f"  Via node {v['j']}: {v['i']} → {v['j']} → {v['k']}")
        print(f"  Time via {v['j']}: {v['t_ij']:.1f} + {v['t_jk']:.1f} = {v['t_ij_jk']:.1f} seconds")
        print(f"  Difference: {v['difference']:.1f} seconds ({v['percentage']:.1f}% slower)")
        print(f"  Issue: Direct route is SLOWER than going through intermediate node!")
    
    # Statistics
    differences = [v['difference'] for v in violations]
    percentages = [v['percentage'] for v in violations]
    
    print(f"\n{'='*60}")
    print(f"STATISTICS")
    print(f"{'='*60}")
    print(f"Average violation: {np.mean(differences):.1f} seconds ({np.mean(percentages):.1f}%)")
    print(f"Max violation: {np.max(differences):.1f} seconds ({np.max(percentages):.1f}%)")
    print(f"Min violation: {np.min(differences):.1f} seconds ({np.min(percentages):.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"ROOT CAUSE")
    print(f"{'='*60}")
    print("The travel time matrix is calculated using:")
    print("  time = distance / fixed_speed (20 km/h)")
    print("\nThis ignores the actual road speeds in the network!")
    print("Shortest DISTANCE path ≠ Shortest TRAVEL TIME path")
    print("\nSolution: Use the actual precomputed travel times from")
    print("the network graph instead of distance/fixed_speed")
    
else:
    print(f"\n✓ No violations found in the sample checked!")
    print("Triangle inequality holds for the checked nodes.")

print(f"\n{'='*60}")
