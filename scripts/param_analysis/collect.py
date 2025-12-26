#!/usr/bin/env python3
"""
Aggregate results from batch search execution into a unified CSV for analysis.

Parses per-query CSV files and summary logs to consolidate all parameter + metric pairs.

Usage:
    python collect.py --input_dir ./batch_results --output results_all.csv
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

def parse_lhs_sample_id(filename):
    """Extract sample_id from filename like result_sample_42_search.csv"""
    match = re.search(r'result_sample_(\d+)_search\.csv$', filename)
    return int(match.group(1)) if match else None

def parse_summary_log(log_file):
    """
    Parse search_disk_index summary statistics from .log file.
    Falls back to summary_stats.csv if available.
    
    Looks for lines like:
      Recall@10: 0.95
      Mean latency: 1234.5 µs
      Overall QPS: 800.5
    """
    stats = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract recall@10
            recall_match = re.search(r'Recall@10:\s*([\d.]+)', content, re.IGNORECASE)
            if recall_match:
                stats['recall'] = float(recall_match.group(1))
            
            # Extract mean latency (in microseconds)
            latency_match = re.search(r'Mean latency:\s*([\d.]+)\s*µ?s', content, re.IGNORECASE)
            if latency_match:
                stats['mean_latency_us'] = float(latency_match.group(1))
            
            # Extract QPS
            qps_match = re.search(r'Overall QPS:\s*([\d.]+)', content, re.IGNORECASE)
            if qps_match:
                stats['qps'] = float(qps_match.group(1))
            
            # Extract average I/O count
            io_match = re.search(r'Average #IOs:\s*([\d.]+)', content, re.IGNORECASE)
            if io_match:
                stats['mean_ios'] = float(io_match.group(1))
            
            # Extract average hops
            hops_match = re.search(r'Average #hops:\s*([\d.]+)', content, re.IGNORECASE)
            if hops_match:
                stats['hop_mean'] = float(hops_match.group(1))
    
    except IOError:
        pass
    
    return stats

def parse_csv_file(csv_file):
    """
    Parse per-query CSV or summary_stats.csv to extract metrics.
    
    Supports both per-query format and summary format.
    """
    stats = {
        'recall': [],
        'mean_latency_us': [],
        'latency_p50_us': [],
        'latency_p90_us': [],
        'latency_p95_us': [],
        'latency_p99_us': [],
        'latency_p999_us': [],
        'latency_max_us': [],
        'mean_ios': [],
        'hop_mean': [],
        'mean_io_us': [],
        'mean_cpu_us': [],
        'mean_sort_us': [],
        'mean_reorder_cpu_us': [],
        'visited_mean': []
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Try to parse recall (might be 'recall', 'recall@10', etc.)
                for col in ['recall', 'recall@10']:
                    if col in row and row[col]:
                        try:
                            val = float(row[col])
                            # Normalize if it's in [0, 100] range (percentage)
                            if val > 1.0:
                                val = val / 100.0
                            stats['recall'].append(val)
                            break
                        except ValueError:
                            pass
                
                # Parse mean latency
                if 'mean_latency_us' in row and row['mean_latency_us']:
                    try:
                        stats['mean_latency_us'].append(float(row['mean_latency_us']))
                    except ValueError:
                        pass
                
                # Parse latency percentiles
                for latency_col in ['latency_p50_us', 'latency_p90_us', 'latency_p95_us', 'latency_p99_us', 'latency_p999_us', 'latency_max_us']:
                    if latency_col in row and row[latency_col]:
                        try:
                            stats[latency_col].append(float(row[latency_col]))
                        except ValueError:
                            pass
                
                # Parse I/O metrics
                if 'mean_ios' in row and row['mean_ios']:
                    try:
                        stats['mean_ios'].append(float(row['mean_ios']))
                    except ValueError:
                        pass
                
                # Hops metrics
                if 'hop_mean' in row and row['hop_mean']:
                    try:
                        stats['hop_mean'].append(float(row['hop_mean']))
                    except ValueError:
                        pass
                
                # Percentile latency
                if 'latency_p999_us' in row and row['latency_p999_us']:
                    try:
                        stats['latency_p999_us'].append(float(row['latency_p999_us']))
                    except ValueError:
                        pass
                
                # Time breakdowns
                if 'mean_io_us' in row and row['mean_io_us']:
                    try:
                        stats['mean_io_us'].append(float(row['mean_io_us']))
                    except ValueError:
                        pass
                
                if 'mean_cpu_us' in row and row['mean_cpu_us']:
                    try:
                        stats['mean_cpu_us'].append(float(row['mean_cpu_us']))
                    except ValueError:
                        pass
                
                if 'mean_sort_us' in row and row['mean_sort_us']:
                    try:
                        stats['mean_sort_us'].append(float(row['mean_sort_us']))
                    except ValueError:
                        pass
                
                if 'mean_reorder_cpu_us' in row and row['mean_reorder_cpu_us']:
                    try:
                        stats['mean_reorder_cpu_us'].append(float(row['mean_reorder_cpu_us']))
                    except ValueError:
                        pass
                
                # Visited nodes
                if 'visited_mean' in row and row['visited_mean']:
                    try:
                        stats['visited_mean'].append(float(row['visited_mean']))
                    except ValueError:
                        pass
    
    except (IOError, ValueError) as e:
        print(f"  WARNING: Error parsing {csv_file}: {e}")
    
    # Compute means
    result = {}
    for key, values in stats.items():
        if values:
            result[key] = sum(values) / len(values)
    
    return result

def read_lhs_parameters(lhs_file):
    """
    Read LHS sample parameters from original CSV.
    Returns dict: {sample_id -> {L, W, num_nodes_to_cache, ...}}
    """
    params = {}
    
    try:
        with open(lhs_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = int(row['sample_id'])
                params[sample_id] = {
                    'L': int(row['L']),
                    'W': int(row['W']),
                    'num_nodes_to_cache': int(row['num_nodes_to_cache']),
                    'search_io_limit': int(row['search_io_limit']),
                    'num_threads': int(row['num_threads'])
                }
    except (IOError, ValueError) as e:
        print(f"WARNING: Could not read LHS parameters: {e}")
    
    return params

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate batch search results into unified CSV'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing batch results')
    parser.add_argument('--output', type=str, default='results_all.csv',
                        help='Output CSV file (default: results_all.csv)')
    parser.add_argument('--lhs_file', type=str, default='lhs_samples.csv',
                        help='Original LHS samples file for parameter mapping')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return False
    
    # Read LHS parameters
    lhs_params = {}
    if os.path.exists(args.lhs_file):
        lhs_params = read_lhs_parameters(args.lhs_file)
        print(f"✓ Loaded {len(lhs_params)} LHS parameter sets from {args.lhs_file}")
    
    # Find all per-query CSV files
    result_files = sorted(glob.glob(os.path.join(args.input_dir, 'result_sample_*_search.csv')))
    
    # If no per-query CSV, try summary stats files
    if not result_files:
        result_files = sorted(glob.glob(os.path.join(args.input_dir, 'result_sample_*_summary_stats.csv')))
    
    if not result_files:
        print(f"ERROR: No result files (*_search.csv or *_summary_stats.csv) found in {args.input_dir}")
        return False
    
    print(f"Found {len(result_files)} result files")
    
    # Aggregate results
    all_results = []
    
    for result_file in result_files:
        sample_id = parse_lhs_sample_id(os.path.basename(result_file))
        if sample_id is None:
            continue
        
        # Get parameters from LHS
        params = lhs_params.get(sample_id, {})
        if not params:
            print(f"WARNING: No parameters found for sample {sample_id}")
            continue
        
        # Try to find summary_stats file (priority over search.csv)
        summary_file = result_file.replace('_search.csv', '')
        # Find the actual summary file
        summary_matches = glob.glob(os.path.join(os.path.dirname(result_file), f'result_sample_{sample_id}_*_summary_stats.csv'))
        if summary_matches:
            summary_file = summary_matches[0]
        else:
            # Fall back to log
            summary_file = result_file.replace('_search.csv', '.log')
        
        # Parse CSV/summary file
        csv_stats = parse_csv_file(summary_file)
        
        # If summary didn't work, try log
        if not csv_stats:
            log_file = result_file.replace('_search.csv', '.log')
            log_stats = parse_summary_log(log_file)
        else:
            log_stats = {}
        
        # Combine all data
        row = {
            'sample_id': sample_id,
            **params,
            **log_stats,
            **csv_stats
        }
        
        all_results.append(row)
        print(f"  Sample {sample_id:3d}: recall={row.get('recall', 0):.4f}, "
              f"latency={row.get('mean_latency_us', 0):.1f}µs, "
              f"IOs={row.get('mean_ios', 0):.1f}")
    
    if not all_results:
        print("ERROR: No results to aggregate")
        return False
    
    # Write consolidated CSV
    fieldnames = ['sample_id', 'L', 'W', 'num_nodes_to_cache', 'search_io_limit', 'num_threads',
                  'recall', 'qps', 'mean_latency_us', 'latency_p50_us', 'latency_p90_us', 'latency_p95_us',
                  'latency_p99_us', 'latency_p999_us', 'latency_max_us',
                  'mean_ios', 'mean_io_us', 'mean_cpu_us', 'mean_sort_us', 'mean_reorder_cpu_us',
                  'hop_mean', 'visited_mean']
    
    try:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Aggregated {len(all_results)} results → {args.output}")
        print(f"\nNext step: Analyze with XGBoost:")
        print(f"  jupyter notebook analyze_xgb.ipynb")
        return True
    
    except IOError as e:
        print(f"ERROR writing output: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
