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
    """Extract sample_id from batch artifact filenames.
    Supports:
      - result_sample_42_search.csv
      - result_sample_42_<suffix>_summary_stats.csv
      - result_sample_42.log
    """
    patterns = [
        r'result_sample_(\d+)_search\.csv$',
        r'result_sample_(\d+)_.*_summary_stats\.csv$',
        r'result_sample_(\d+)\.log$'
    ]
    for pat in patterns:
        match = re.search(pat, filename)
        if match:
            return int(match.group(1))
    return None

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
            
            # Extract recall@K and K value
            recall_match = re.search(r'Recall@(\d+):\s*([\d.]+)', content, re.IGNORECASE)
            if recall_match:
                stats['search_k'] = int(recall_match.group(1))
                stats['recall'] = float(recall_match.group(2))
            
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

def parse_build_search_params_from_filename(filename):
        """Parse build/search parameters from summary filename if present.
        Expected pattern example:
            result_sample_3_siftsmall_R64_L100_B0.2_M1_W4_cache42000_T8_summary_stats.csv
        Returns dict with keys: build_R, build_L, build_B, build_M, beamwidth_from_name,
            configured_cache_nodes, search_threads, dataset (optional).
        """
        name = os.path.basename(filename)
        params = {}
        # Dataset (between sample id and R)
        ds_match = re.search(r'result_sample_\d+_([^_]+)_R', name)
        if ds_match:
                params['dataset'] = ds_match.group(1)
        # Build/Search params
        match = re.search(
                r'_R(\d+)_L(\d+)_B([0-9.]+)_M(\d+)_W(\d+)_cache(\d+)_T(\d+)_summary_stats\.csv$',
                name
        )
        if match:
                params['build_R'] = int(match.group(1))
                params['build_L'] = int(match.group(2))
                params['build_B'] = float(match.group(3))
                params['build_M'] = int(match.group(4))
                params['beamwidth_from_name'] = int(match.group(5))
                params['configured_cache_nodes'] = int(match.group(6))
                params['search_threads'] = int(match.group(7))
        return params

def parse_csv_file(csv_file):
    """
    Parse per-query CSV or summary_stats.csv to extract metrics.
    
    Supports both per-query format and summary format.
    """
    stats = {
        'build_R': [],
        'build_L': [],
        'build_B': [],
        'build_M': [],
        'search_K': [],
        'search_L': [],
        'search_W': [],
        'search_T': [],
        'search_io_limit': [],
        'num_queries': [],
        'dataset_size': [],
        'vector_dim': [],
        'actual_cached_nodes': [],
        'qps': [],
        'out_degree_mean': [],
        'out_degree_p0': [],
        'out_degree_p1': [],
        'out_degree_p5': [],
        'out_degree_p10': [],
        'out_degree_p25': [],
        'out_degree_p50': [],
        'out_degree_p75': [],
        'out_degree_p90': [],
        'out_degree_p95': [],
        'out_degree_p99': [],
        'out_degree_max': [],
        'mean_latency_us': [],
        'latency_p50_us': [],
        'latency_p75_us': [],
        'latency_p90_us': [],
        'latency_p95_us': [],
        'latency_p99_us': [],
        'latency_p999_us': [],
        'latency_max_us': [],
        'ios_mean': [],
        'ios_p50': [],
        'ios_p75': [],
        'ios_p90': [],
        'ios_p95': [],
        'ios_p99': [],
        'ios_max': [],
        'io_us_mean': [],
        'io_us_p50': [],
        'io_us_p75': [],
        'io_us_p90': [],
        'io_us_p95': [],
        'io_us_p99': [],
        'io_us_max': [],
        'cpu_us_mean': [],
        'cpu_us_p50': [],
        'cpu_us_p75': [],
        'cpu_us_p90': [],
        'cpu_us_p95': [],
        'cpu_us_p99': [],
        'cpu_us_max': [],
        'sort_us_mean': [],
        'sort_us_p50': [],
        'sort_us_p75': [],
        'sort_us_p90': [],
        'sort_us_p95': [],
        'sort_us_p99': [],
        'sort_us_max': [],
        'read_size_mean': [],
        'read_size_p50': [],
        'read_size_p75': [],
        'read_size_p90': [],
        'read_size_p95': [],
        'read_size_p99': [],
        'read_size_max': [],
        'compares_mean': [],
        'compares_p50': [],
        'compares_p75': [],
        'compares_p90': [],
        'compares_p95': [],
        'compares_p99': [],
        'compares_max': [],
        'recall_mean': [],
        'recall_p0': [],
        'recall_p1': [],
        'recall_p5': [],
        'recall_p10': [],
        'recall_p25': [],
        'recall_p50': [],
        'recall_p75': [],
        'recall_p90': [],
        'recall_max': [],
        'cache_hit_rate_mean': [],
        'cache_hit_rate_p0': [],
        'cache_hit_rate_p1': [],
        'cache_hit_rate_p5': [],
        'cache_hit_rate_p10': [],
        'cache_hit_rate_p25': [],
        'cache_hit_rate_p50': [],
        'cache_hit_rate_p75': [],
        'cache_hit_rate_p90': [],
        'cache_hit_rate_max': [],
        'hop_mean': [],
        'hop_p50': [],
        'hop_p75': [],
        'hop_p90': [],
        'hop_p95': [],
        'hop_p99': [],
        'hop_max': [],
        'visited_mean': [],
        'visited_p50': [],
        'visited_p75': [],
        'visited_p90': [],
        'visited_p95': [],
        'visited_p99': [],
        'visited_max': []
    }
    meta = {}

    def add_value(key, value):
        if value is None or value == '':
            return
        try:
            stats[key].append(float(value))
        except (ValueError, KeyError):
            pass
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if 'dataset_name' in row and row['dataset_name']:
                    meta.setdefault('dataset_name', row['dataset_name'])
                    meta.setdefault('dataset', row['dataset_name'])
                if 'data_type' in row and row['data_type']:
                    meta.setdefault('data_type', row['data_type'])

                # Try to parse recall (legacy + new summary columns).
                for col in ['recall_mean', 'recall', 'recall@10']:
                    if col in row and row[col]:
                        try:
                            val = float(row[col])
                            if val > 1.0:
                                val = val / 100.0
                            stats['recall_mean'].append(val)
                            break
                        except ValueError:
                            pass
                for col in ['recall_mean', 'recall_p0', 'recall_p1', 'recall_p5', 'recall_p10', 'recall_p25',
                            'recall_p50', 'recall_p75', 'recall_p90', 'recall_max']:
                    add_value(col, row.get(col))
                
                # Parse QPS (support alternative column names)
                for qps_col in ['qps', 'overall_qps']:
                    if qps_col in row and row[qps_col]:
                        try:
                            stats['qps'].append(float(row[qps_col]))
                            break
                        except ValueError:
                            pass

                # Parse mean latency
                add_value('mean_latency_us', row.get('mean_latency_us'))
                add_value('mean_latency_us', row.get('total_us'))
                
                # Parse latency percentiles
                for latency_col in ['latency_p50_us', 'latency_p75_us', 'latency_p90_us', 'latency_p95_us',
                                    'latency_p99_us', 'latency_p999_us', 'latency_max_us']:
                    add_value(latency_col, row.get(latency_col))
                
                # Parse I/O metrics
                add_value('ios_mean', row.get('ios_mean'))
                add_value('ios_mean', row.get('mean_ios'))
                add_value('ios_mean', row.get('n_ios'))
                for ios_col in ['ios_p50', 'ios_p75', 'ios_p90', 'ios_p95', 'ios_p99', 'ios_max']:
                    add_value(ios_col, row.get(ios_col))
                
                # Hops metrics
                add_value('hop_mean', row.get('hop_mean'))
                add_value('hop_mean', row.get('n_hops'))
                for hop_col in ['hop_p50', 'hop_p75', 'hop_p90', 'hop_p95', 'hop_p99', 'hop_max']:
                    add_value(hop_col, row.get(hop_col))
                
                # Percentile latency
                for hit_col in ['cache_hit_rate_mean', 'cache_hit_rate_p0', 'cache_hit_rate_p1', 'cache_hit_rate_p5',
                                'cache_hit_rate_p10', 'cache_hit_rate_p25', 'cache_hit_rate_p50', 'cache_hit_rate_p75',
                                'cache_hit_rate_p90', 'cache_hit_rate_max']:
                    add_value(hit_col, row.get(hit_col))
                
                # Time breakdowns
                add_value('io_us_mean', row.get('io_us_mean'))
                add_value('io_us_mean', row.get('mean_io_us'))
                add_value('io_us_mean', row.get('io_us'))
                for io_col in ['io_us_p50', 'io_us_p75', 'io_us_p90', 'io_us_p95', 'io_us_p99', 'io_us_max']:
                    add_value(io_col, row.get(io_col))
                
                add_value('cpu_us_mean', row.get('cpu_us_mean'))
                add_value('cpu_us_mean', row.get('mean_cpu_us'))
                add_value('cpu_us_mean', row.get('cpu_us'))
                for cpu_col in ['cpu_us_p50', 'cpu_us_p75', 'cpu_us_p90', 'cpu_us_p95', 'cpu_us_p99', 'cpu_us_max']:
                    add_value(cpu_col, row.get(cpu_col))
                
                add_value('sort_us_mean', row.get('sort_us_mean'))
                add_value('sort_us_mean', row.get('mean_sort_us'))
                add_value('sort_us_mean', row.get('sort_us'))
                for sort_col in ['sort_us_p50', 'sort_us_p75', 'sort_us_p90', 'sort_us_p95', 'sort_us_p99',
                                 'sort_us_max']:
                    add_value(sort_col, row.get(sort_col))
                
                if 'mean_reorder_cpu_us' in row and row['mean_reorder_cpu_us']:
                    try:
                        stats['mean_reorder_cpu_us'].append(float(row['mean_reorder_cpu_us']))
                    except ValueError:
                        pass

                for read_col in ['read_size_mean', 'read_size_p50', 'read_size_p75', 'read_size_p90',
                                 'read_size_p95', 'read_size_p99', 'read_size_max']:
                    add_value(read_col, row.get(read_col))
                add_value('read_size_mean', row.get('read_size'))

                for cmp_col in ['compares_mean', 'compares_p50', 'compares_p75', 'compares_p90', 'compares_p95',
                                'compares_p99', 'compares_max']:
                    add_value(cmp_col, row.get(cmp_col))
                add_value('compares_mean', row.get('n_cmps'))

                add_value('dataset_size', row.get('dataset_size'))
                add_value('vector_dim', row.get('vector_dim'))
                add_value('actual_cached_nodes', row.get('actual_cached_nodes'))
                add_value('build_R', row.get('build_R'))
                add_value('build_L', row.get('build_L'))
                add_value('build_B', row.get('build_B'))
                add_value('build_M', row.get('build_M'))
                add_value('search_K', row.get('search_K'))
                add_value('search_L', row.get('search_L'))
                add_value('search_W', row.get('search_W'))
                add_value('search_T', row.get('search_T'))
                add_value('search_io_limit', row.get('search_io_limit'))
                add_value('num_queries', row.get('num_queries'))
                for out_col in ['out_degree_mean', 'out_degree_p0', 'out_degree_p1', 'out_degree_p5',
                                'out_degree_p10', 'out_degree_p25', 'out_degree_p50', 'out_degree_p75',
                                'out_degree_p90', 'out_degree_p95', 'out_degree_p99', 'out_degree_max']:
                    add_value(out_col, row.get(out_col))
                
                # Visited nodes
                add_value('visited_mean', row.get('visited_mean'))
                add_value('visited_mean', row.get('visited_nodes'))
                for v_col in ['visited_p50', 'visited_p75', 'visited_p90', 'visited_p95', 'visited_p99',
                              'visited_max']:
                    add_value(v_col, row.get(v_col))
    
    except (IOError, ValueError) as e:
        print(f"  WARNING: Error parsing {csv_file}: {e}")
    
    # Compute means
    result = {}
    for key, values in stats.items():
        if values:
            result[key] = sum(values) / len(values)
    result.update(meta)
    
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
                # Optional K in LHS file
                search_k = None
                for k_col in ['search_k', 'K']:
                    if k_col in row and row[k_col]:
                        try:
                            search_k = int(row[k_col])
                        except ValueError:
                            pass
                        break

                params[sample_id] = {
                    'L': int(row['L']),
                    'beamwidth': int(row.get('beamwidth', row['W'])),
                    'num_nodes_to_cache': int(row['num_nodes_to_cache']),
                    'search_io_limit': int(row['search_io_limit']),
                    'num_threads': int(row['num_threads']),
                    **({'search_k': search_k} if search_k is not None else {})
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
    
    fieldnames = [
        'dataset_name', 'data_type', 'build_R', 'build_L', 'build_B', 'build_M', 'search_K', 'search_L', 'search_W',
        'search_T', 'search_io_limit', 'num_queries', 'dataset_size', 'vector_dim', 'actual_cached_nodes', 'qps',
        'out_degree_mean', 'out_degree_p0', 'out_degree_p1', 'out_degree_p5', 'out_degree_p10', 'out_degree_p25',
        'out_degree_p50', 'out_degree_p75', 'out_degree_p90', 'out_degree_p95', 'out_degree_p99', 'out_degree_max',
        'mean_latency_us', 'latency_p50_us', 'latency_p75_us', 'latency_p90_us', 'latency_p95_us', 'latency_p99_us',
        'latency_p999_us', 'latency_max_us', 'ios_mean', 'ios_p50', 'ios_p75', 'ios_p90', 'ios_p95', 'ios_p99',
        'ios_max', 'io_us_mean', 'io_us_p50', 'io_us_p75', 'io_us_p90', 'io_us_p95', 'io_us_p99', 'io_us_max',
        'cpu_us_mean', 'cpu_us_p50', 'cpu_us_p75', 'cpu_us_p90', 'cpu_us_p95', 'cpu_us_p99', 'cpu_us_max',
        'sort_us_mean', 'sort_us_p50', 'sort_us_p75', 'sort_us_p90', 'sort_us_p95', 'sort_us_p99', 'sort_us_max',
        'read_size_mean', 'read_size_p50', 'read_size_p75', 'read_size_p90', 'read_size_p95', 'read_size_p99',
        'read_size_max', 'compares_mean', 'compares_p50', 'compares_p75', 'compares_p90', 'compares_p95',
        'compares_p99', 'compares_max', 'recall_mean', 'recall_p0', 'recall_p1', 'recall_p5', 'recall_p10',
        'recall_p25', 'recall_p50', 'recall_p75', 'recall_p90', 'recall_max', 'cache_hit_rate_mean',
        'cache_hit_rate_p0', 'cache_hit_rate_p1', 'cache_hit_rate_p5', 'cache_hit_rate_p10', 'cache_hit_rate_p25',
        'cache_hit_rate_p50', 'cache_hit_rate_p75', 'cache_hit_rate_p90', 'cache_hit_rate_max', 'hop_mean',
        'hop_p50', 'hop_p75', 'hop_p90', 'hop_p95', 'hop_p99', 'hop_max', 'visited_mean', 'visited_p50',
        'visited_p75', 'visited_p90', 'visited_p95', 'visited_p99', 'visited_max'
    ]

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

        # Parse build/search params from filename if present
        fn_params = parse_build_search_params_from_filename(summary_file)
        
        # Combine all data
        row = {
            **params,
            **log_stats,
            **csv_stats,
            **fn_params
        }

        if 'dataset_name' not in row and 'dataset' in row:
            row['dataset_name'] = row['dataset']
        if 'dataset' not in row and 'dataset_name' in row:
            row['dataset'] = row['dataset_name']
        if 'search_K' not in row and 'search_k' in row:
            row['search_K'] = row['search_k']
        if 'search_L' not in row and 'L' in row:
            row['search_L'] = row['L']
        if 'search_W' not in row and 'beamwidth' in row:
            row['search_W'] = row['beamwidth']
        if 'search_T' not in row and 'num_threads' in row:
            row['search_T'] = row['num_threads']
        if 'recall_mean' not in row and 'recall' in row:
            row['recall_mean'] = row['recall']
        if 'ios_mean' not in row and 'mean_ios' in row:
            row['ios_mean'] = row['mean_ios']
        if 'io_us_mean' not in row and 'mean_io_us' in row:
            row['io_us_mean'] = row['mean_io_us']
        if 'cpu_us_mean' not in row and 'mean_cpu_us' in row:
            row['cpu_us_mean'] = row['mean_cpu_us']
        
        filtered_row = {key: row.get(key, '') for key in fieldnames}
        all_results.append(filtered_row)
        print(f"  Sample {sample_id:3d}: recall={row.get('recall_mean', 0):.4f}, "
              f"latency={row.get('mean_latency_us', 0):.1f}µs, "
              f"IOs={row.get('ios_mean', 0):.1f}")
    
    if not all_results:
        print("ERROR: No results to aggregate")
        return False
    
    # Write consolidated CSV
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
