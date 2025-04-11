#!/usr/bin/env python
"""
Visualization script for dominating set solver results.
Creates various comparative graphs from the CSV result files.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import re

# Set plot style
plt.style.use('ggplot')
sns.set_palette("Set2")

# Set font sizes for better readability
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def parse_graph_size(graph_name):
    """Extract the graph size from the graph name."""
    match = re.search(r'bremen_subgraph_(\d+)', graph_name)
    if match:
        return int(match.group(1))
    return 0

def load_data(results_dir):
    """
    Load all CSV data files from the results directory structure.
    Returns a dictionary with graph names as keys and solver data as values.
    """
    data = {}
    
    # Find all graph directories
    graph_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    graph_dirs.sort(key=parse_graph_size)  # Sort by graph size
    
    for graph_dir in graph_dirs:
        graph_path = os.path.join(results_dir, graph_dir)
        graph_data = {}
        
        # Find all solver directories
        solver_dirs = [d for d in os.listdir(graph_path) if os.path.isdir(os.path.join(graph_path, d))]
        
        for solver_dir in solver_dirs:
            solver_path = os.path.join(graph_path, solver_dir)
            csv_path = os.path.join(solver_path, "data.csv")
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # Process the dataframe
                    graph_data[solver_dir] = df
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
        
        if graph_data:
            data[graph_dir] = graph_data
    
    return data

def plot_solution_size_comparison(data, output_dir):
    """
    Create a bar chart comparing solution sizes across different solvers for each graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for graph_name, solvers_data in data.items():
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        solver_names = []
        avg_solution_sizes = []
        std_solution_sizes = []
        min_solution_sizes = []
        expected_sizes = []
        
        for solver_name, df in solvers_data.items():
            if "Solution Size" in df.columns or "Number of Vertices" in df.columns:
                size_col = "Solution Size" if "Solution Size" in df.columns else "Number of Vertices"
                expected_col = "Expected Size" if "Expected Size" in df.columns else "Number of vertices expected"
                
                solver_names.append(solver_name)
                avg_solution_sizes.append(df[size_col].mean())
                std_solution_sizes.append(df[size_col].std())
                min_solution_sizes.append(df[size_col].min())
                
                # Get expected size (should be the same for all runs)
                if not expected_sizes and expected_col in df.columns:
                    expected_sizes.append(df[expected_col].iloc[0])
        
        # Create the bar chart
        x = np.arange(len(solver_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, avg_solution_sizes, width, yerr=std_solution_sizes, 
                      label='Average Solution Size', alpha=0.7)
        
        # Add min solution sizes as points
        ax.scatter(x, min_solution_sizes, color='red', s=50, label='Minimum Solution Size')
        
        # Add expected solution size as a horizontal line if available
        if expected_sizes:
            ax.axhline(y=expected_sizes[0], color='green', linestyle='-', label=f'Expected Size ({expected_sizes[0]})')
        
        # Add labels and titles
        ax.set_xlabel('Solver')
        ax.set_ylabel('Solution Size (Number of Vertices)')
        ax.set_title(f'Dominating Set Solution Size Comparison - {graph_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(solver_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{graph_name}_solution_sizes.png"))
        plt.close()

def plot_execution_time_comparison(data, output_dir):
    """
    Create a bar chart comparing execution times across different solvers for each graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for graph_name, solvers_data in data.items():
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        solver_names = []
        avg_times = []
        std_times = []
        min_times = []
        
        for solver_name, df in solvers_data.items():
            if "Time" in df.columns:
                solver_names.append(solver_name)
                avg_times.append(df["Time"].mean())
                std_times.append(df["Time"].std())
                min_times.append(df["Time"].min())
        
        # Create the bar chart
        x = np.arange(len(solver_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, avg_times, width, yerr=std_times, 
                      label='Average Execution Time (s)', alpha=0.7)
        
        # Add min times as points
        ax.scatter(x, min_times, color='red', s=50, label='Minimum Execution Time (s)')
        
        # Add labels and titles
        ax.set_xlabel('Solver')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Dominating Set Execution Time Comparison - {graph_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(solver_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{graph_name}_execution_times.png"))
        plt.close()

def plot_scaling_comparison(data, output_dir):
    """
    Create line charts showing how different solvers scale with graph size.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by solver and graph size
    solver_data = {}
    graph_sizes = []
    
    for graph_name, solvers_data in data.items():
        graph_size = parse_graph_size(graph_name)
        graph_sizes.append(graph_size)
        
        for solver_name, df in solvers_data.items():
            if solver_name not in solver_data:
                solver_data[solver_name] = {"sizes": [], "times": [], "solution_sizes": []}
            
            if "Time" in df.columns:
                solver_data[solver_name]["sizes"].append(graph_size)
                solver_data[solver_name]["times"].append(df["Time"].mean())
                
                size_col = "Solution Size" if "Solution Size" in df.columns else "Number of Vertices"
                if size_col in df.columns:
                    solver_data[solver_name]["solution_sizes"].append(df[size_col].min())
    
    # Sort data by graph size
    for solver_name in solver_data:
        size_order = np.argsort(solver_data[solver_name]["sizes"])
        solver_data[solver_name]["sizes"] = np.array(solver_data[solver_name]["sizes"])[size_order]
        solver_data[solver_name]["times"] = np.array(solver_data[solver_name]["times"])[size_order]
        if solver_data[solver_name]["solution_sizes"]:
            solver_data[solver_name]["solution_sizes"] = np.array(solver_data[solver_name]["solution_sizes"])[size_order]
    
    # Plot execution time scaling
    plt.figure(figsize=(12, 6))
    
    for solver_name, solver_info in solver_data.items():
        if len(solver_info["sizes"]) > 1:  # Need at least 2 points for a line
            plt.plot(solver_info["sizes"], solver_info["times"], marker='o', label=solver_name)
    
    plt.xlabel('Graph Size (Number of Vertices)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Execution Time Scaling with Graph Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_time_scaling.png"))
    plt.close()
    
    # Plot solution size scaling
    plt.figure(figsize=(12, 6))
    
    for solver_name, solver_info in solver_data.items():
        if len(solver_info["sizes"]) > 1 and len(solver_info["solution_sizes"]) > 1:
            plt.plot(solver_info["sizes"], solver_info["solution_sizes"], marker='o', label=solver_name)
    
    plt.xlabel('Graph Size (Number of Vertices)')
    plt.ylabel('Best Solution Size (Number of Vertices)')
    plt.title('Solution Size Scaling with Graph Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "solution_size_scaling.png"))
    plt.close()

def plot_solution_quality_vs_time(data, output_dir):
    """
    Create scatter plots showing the trade-off between solution quality and execution time.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for graph_name, solvers_data in data.items():
        plt.figure(figsize=(10, 6))
        
        for solver_name, df in solvers_data.items():
            if "Time" in df.columns:
                size_col = "Solution Size" if "Solution Size" in df.columns else "Number of Vertices"
                if size_col in df.columns:
                    plt.scatter(df["Time"], df[size_col], label=solver_name, alpha=0.7, s=50)
        
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Solution Size (Number of Vertices)')
        plt.title(f'Solution Quality vs. Execution Time - {graph_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{graph_name}_quality_vs_time.png"))
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_directory> [output_directory]")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualization_output"
    
    print(f"Loading data from {results_dir}...")
    data = load_data(results_dir)
    
    if not data:
        print("No data found!")
        sys.exit(1)
    
    print(f"Found data for {len(data)} graphs")
    for graph_name in data:
        print(f"  - {graph_name}: {list(data[graph_name].keys())}")
    
    print("\nGenerating visualizations...")
    
    print("1. Solution size comparison...")
    plot_solution_size_comparison(data, output_dir)
    
    print("2. Execution time comparison...")
    plot_execution_time_comparison(data, output_dir)
    
    print("3. Scaling comparison...")
    plot_scaling_comparison(data, output_dir)
    
    print("4. Solution quality vs. time...")
    plot_solution_quality_vs_time(data, output_dir)
    
    print(f"\nVisualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
