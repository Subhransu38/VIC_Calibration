"""
Post-processing utilities for calibration results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from typing import Dict, List, Optional
import os

class PostProcessor:
    """Post-process and analyze calibration results"""
    
    def __init__(self, output_dir: str = 'output/calibration'):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_parameter_evolution(self, results_file: str) -> None:
        """Plot parameter evolution during calibration"""
        try:
            results = pd.read_csv(results_file)
            param_cols = [col for col in results.columns if col.startswith('par')]
            
            if not param_cols:
                self.logger.warning("No parameter columns found in results")
                return
            
            n_params = len(param_cols)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            param_names = ['binfilt', 'Ws', 'Ds', 'Dsmax', 'soil_d1', 'soil_d2', 'soil_d3']
            
            for i, (param_col, param_name) in enumerate(zip(param_cols, param_names)):
                if i < len(axes):
                    axes[i].plot(results[param_col], alpha=0.7)
                    axes[i].set_title(f'{param_name}')
                    axes[i].set_xlabel('Iteration')
                    axes[i].set_ylabel('Parameter Value')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(param_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'parameter_evolution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Parameter evolution plot saved")
            
        except Exception as e:
            self.logger.error(f"Error creating parameter evolution plot: {e}")
    
    def plot_objective_function(self, results_file: str) -> None:
        """Plot objective function evolution"""
        try:
            results = pd.read_csv(results_file)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot all values
            ax1.plot(results['like1'], alpha=0.7, color='blue')
            ax1.set_title('Objective Function Evolution')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.grid(True, alpha=0.3)
            
            # Plot running maximum
            running_max = results['like1'].cummax()
            ax2.plot(running_max, color='red', linewidth=2)
            ax2.set_title('Best Objective Value')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Objective Value')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'objective_function.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Objective function plot saved")
            
        except Exception as e:
            self.logger.error(f"Error creating objective function plot: {e}")
    
    def create_parameter_correlation_matrix(self, results_file: str) -> None:
        """Create parameter correlation matrix"""
        try:
            results = pd.read_csv(results_file)
            param_cols = [col for col in results.columns if col.startswith('par')]
            
            if len(param_cols) < 2:
                self.logger.warning("Not enough parameters for correlation matrix")
                return
            
            param_data = results[param_cols]
            correlation_matrix = param_data.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'parameter_correlation.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Parameter correlation matrix saved")
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {e}")
    
    def generate_summary_report(self, results_file: str, best_params_file: str) -> None:
        """Generate a summary report of calibration results"""
        try:
            # Load data
            results = pd.read_csv(results_file)
            with open(best_params_file, 'r') as f:
                best_params = json.load(f)
            
            # Generate report
            report_file = os.path.join(self.output_dir, 'calibration_summary_report.md')
            
            with open(report_file, 'w') as f:
                f.write("# VIC Calibration Summary Report\n\n")
                f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Calibration Settings\n")
                f.write(f"- **Total Iterations:** {len(results)}\n")
                f.write(f"- **Best Iteration:** {best_params['iteration']}\n")
                f.write(f"- **Best Objective Value:** {best_params['objective_value']:.6f}\n\n")
                
                f.write("## Best Parameters\n")
                f.write("| Parameter | Value | Unit |\n")
                f.write("|-----------|-------|------|\n")
                param_units = {
                    'binfilt': '-',
                    'Ws': 'fraction',
                    'Ds': 'fraction', 
                    'Dsmax': 'mm/day',
                    'soil_d1': 'm',
                    'soil_d2': 'm',
                    'soil_d3': 'm'
                }
                
                for param, value in best_params.items():
                    if param in param_units:
                        f.write(f"| {param} | {value:.6f} | {param_units[param]} |\n")
                
                f.write("\n## Calibration Statistics\n")
                f.write(f"- **Mean Objective Value:** {results['like1'].mean():.6f}\n")
                f.write(f"- **Standard Deviation:** {results['like1'].std():.6f}\n")
                f.write(f"- **Minimum Value:** {results['like1'].min():.6f}\n")
                f.write(f"- **Maximum Value:** {results['like1'].max():.6f}\n")
                
                # Calculate convergence info
                last_100 = results['like1'].tail(100)
                f.write(f"- **Last 100 iterations mean:** {last_100.mean():.6f}\n")
                f.write(f"- **Last 100 iterations std:** {last_100.std():.6f}\n")
                
                f.write("\n## Files Generated\n")
                f.write("- `best_parameters.json`: Best parameter set\n")
                f.write("- `parameter_evolution.png`: Parameter evolution plots\n")
                f.write("- `objective_function.png`: Objective function evolution\n")
                f.write("- `parameter_correlation.png`: Parameter correlation matrix\n")
                f.write("- `calibration.log`: Detailed calibration log\n")
            
            self.logger.info(f"Summary report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
