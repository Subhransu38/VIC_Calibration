"""
Visualization utilities for calibration results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import os

class CalibrationVisualizer:
    """Visualization tools for calibration analysis"""
    
    def __init__(self, output_dir: str = 'output/calibration/plots'):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("tab10")
    
    def plot_hydrographs(self, observed: Dict[str, np.ndarray], 
                        simulated: Dict[str, np.ndarray],
                        dates: pd.DatetimeIndex,
                        station_subset: Optional[List[str]] = None) -> None:
        """Plot observed vs simulated hydrographs for selected stations"""
        
        stations_to_plot = station_subset or list(observed.keys())[:4]  # Plot first 4 if no subset
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, station in enumerate(stations_to_plot):
            if i >= len(axes):
                break
                
            if station in observed and station in simulated:
                obs_data = observed[station]
                sim_data = simulated[station]
                
                # Remove NaN values for plotting
                valid_mask = ~(np.isnan(obs_data) | np.isnan(sim_data))
                
                axes[i].plot(dates[valid_mask], obs_data[valid_mask], 
                           label='Observed', color='blue', alpha=0.7)
                axes[i].plot(dates[valid_mask], sim_data[valid_mask], 
                           label='Simulated', color='red', alpha=0.7)
                
                axes[i].set_title(f'{station}')
                axes[i].set_ylabel('Discharge (cumecs)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Format x-axis
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axes[i].xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hydrographs_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Hydrographs comparison plot saved")
    
    def plot_scatter_comparison(self, observed: Dict[str, np.ndarray], 
                               simulated: Dict[str, np.ndarray]) -> None:
        """Create scatter plots for observed vs simulated values"""
        
        # Combine all station data
        all_obs = []
        all_sim = []
        
        for station in observed.keys():
            if station in simulated:
                obs_data = observed[station]
                sim_data = simulated[station]
                
                valid_mask = ~(np.isnan(obs_data) | np.isnan(sim_data)) & (obs_data > 0)
                
                all_obs.extend(obs_data[valid_mask])
                all_sim.extend(sim_data[valid_mask])
        
        if len(all_obs) == 0:
            self.logger.warning("No valid data for scatter plot")
            return
        
        all_obs = np.array(all_obs)
        all_sim = np.array(all_sim)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.scatter(all_obs, all_sim, alpha=0.5, s=1)
        max_val = max(np.max(all_obs), np.max(all_sim))
        ax1.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
        ax1.set_xlabel('Observed (cumecs)')
        ax1.set_ylabel('Simulated (cumecs)')
        ax1.set_title('Linear Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        positive_mask = (all_obs > 0) & (all_sim > 0)
        if np.sum(positive_mask) > 0:
            ax2.scatter(all_obs[positive_mask], all_sim[positive_mask], alpha=0.5, s=1)
            ax2.plot([np.min(all_obs[positive_mask]), np.max(all_obs[positive_mask])], 
                    [np.min(all_obs[positive_mask]), np.max(all_obs[positive_mask])], 
                    'r--', label='1:1 line')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel('Observed (cumecs)')
            ax2.set_ylabel('Simulated (cumecs)')
            ax2.set_title('Log Scale')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scatter_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Scatter comparison plot saved")