#!/usr/bin/env python3
"""
Generate Latin Hypercube Sampling (LHS) points for DiskANN parameter sensitivity analysis.

Usage:
    python gen_lhs.py --n_samples 200 --output lhs_samples.csv --seed 42 --k 10
"""

import argparse
import csv
import numpy as np
from scipy.stats import qmc

def generate_lhs_samples(n_samples=200, seed=42, k_value=10):
    """
    Generate LHS samples for DiskANN search parameters with external K.
    
    Parameters (normalized to [0,1]):
    1. L (search_list):           [K, 400]      → exploration breadth (lower bound follows K)
    2. W (beamwidth):             [1, 16]       → candidate beam width
    3. num_nodes_to_cache:        [0, 50000]    → DRAM prefetch budget
    4. search_io_limit:           [10, 500]     → I/O budget ceiling
    5. num_threads (T):           [1, 16]       → parallelism level
    K is fixed externally (input argument) and emitted per sample.
    
    Returns:
        List of dictionaries with parameter combinations
    """
    
    # Define bounds in original units
    bounds = {
        'L': (max(5, k_value), 400),
        'W': (1, 16),
        'num_nodes_to_cache': (0, 50000),
        'search_io_limit': (10, 500),
        'num_threads': (1, 16)
    }
    
    # Generate LHS in [0,1]^5
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed, optimization=None)
    lhs_samples = sampler.random(n_samples)
    
    # Map to original parameter ranges
    param_names = list(bounds.keys())
    samples = []
    
    for i, sample in enumerate(lhs_samples):
        params = {'sample_id': i + 1, 'K': k_value}
        
        for j, param_name in enumerate(param_names):
            lo, hi = bounds[param_name]
            
            # Handle discrete parameters differently
            if param_name in ['L', 'W', 'num_threads', 'search_io_limit']:
                # Round to nearest integer
                value = int(np.round(lo + sample[j] * (hi - lo)))
                value = max(lo, min(hi, value))  # Clamp to bounds
            elif param_name == 'num_nodes_to_cache':
                # Round to nearest 1000
                value = int(np.round((lo + sample[j] * (hi - lo)) / 1000) * 1000)
                value = max(lo, min(hi, value))
            
            params[param_name] = value
        samples.append(params)
    
    return samples

def save_samples_to_csv(samples, output_file):
    """Save LHS samples to CSV file."""
    if not samples:
        print("ERROR: No samples generated")
        return False
    
    fieldnames = ['sample_id', 'L', 'W', 'num_nodes_to_cache', 'search_io_limit', 'num_threads', 'K']
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)
        
        print(f"✓ Generated {len(samples)} LHS samples → {output_file}")
        
        # Print summary statistics
        print("\nParameter ranges in sample set:")
        for param in fieldnames[1:]:
            values = [s[param] for s in samples]
            print(f"  {param:20s}: [{min(values):6d}, {max(values):6d}]  (mean: {np.mean(values):6.1f})")
        
        return True
    except IOError as e:
        print(f"ERROR writing to {output_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Generate LHS samples for DiskANN parameter sensitivity analysis'
    )
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of LHS samples (default: 200)')
    parser.add_argument('--output', type=str, default='lhs_samples.csv',
                        help='Output CSV file (default: lhs_samples.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--k', type=int, default=10,
                        help='Fixed K (top-K) value; also used as the lower bound for L (default: 10)')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} LHS samples with seed={args.seed}, K={args.k}...")
    samples = generate_lhs_samples(n_samples=args.n_samples, seed=args.seed, k_value=args.k)
    
    if save_samples_to_csv(samples, args.output):
        print(f"\nReady for batch execution. Next step:")
        print(f"  bash run_batch.sh {args.output}")

if __name__ == '__main__':
    main()
