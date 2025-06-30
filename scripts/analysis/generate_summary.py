"""
Generate comprehensive summary of calibration results
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import configparser

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utilities.post_processor import PostProcessor
from utilities.visualization import CalibrationVisualizer

def main():
    parser = argparse.ArgumentParser(description='Generate calibration summary')
    parser.add_argument('--config', default='config/calibration_config.ini', 
                       help='Configuration file path')
    args = parser.parse_args()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    
    output_dir = config.get('PATHS', 'calibration_output')
    results_dir = os.path.join(output_dir, 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Ensure directories exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find results file
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not results_files:
        print("No results files found!")
        return
    
    results_file = os.path.join(results_dir, results_files[0])
    best_params_file = os.path.join(results_dir, 'best_parameters.json')
    
    # Initialize processors
    post_processor = PostProcessor(output_dir)
    visualizer = CalibrationVisualizer(plots_dir)
    
    # Generate all plots and analysis
    print("Generating parameter evolution plot...")
    post_processor.analyze_parameter_evolution(results_file)
    
    print("Generating objective function plot...")
    post_processor.plot_objective_function(results_file)
    
    print("Generating parameter correlation matrix...")
    post_processor.create_parameter_correlation_matrix(results_file)
    
    if os.path.exists(best_params_file):
        print("Generating summary report...")
        post_processor.generate_summary_report(results_file, best_params_file)
    
    print(f"Summary generation completed! Check {output_dir}")

if __name__ == "__main__":
    main()