"""
Data validation utilities for VIC calibration
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import os

class DataValidator:
    """ Validate input data for VIC calibration """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_streamflow_data(self, data: Dict[str, np.ndarray], 
                                min_valid_ratio: float = 0.7) -> Dict[str, Dict]:
        """Validate streamflow data quality"""
        validation_results = {}
        
        for station_name, flow_data in data.items():
            results = {
                'total_days': len(flow_data),
                'valid_days': np.sum(~np.isnan(flow_data) & (flow_data >= 0)),
                'negative_values': np.sum(flow_data < 0),
                'zero_values': np.sum(flow_data == 0),
                'missing_values': np.sum(np.isnan(flow_data)),
                'outliers': 0,
                'valid_ratio': 0,
                'quality_flag': 'POOR'
            }
            
            # Calculate valid ratio
            results['valid_ratio'] = results['valid_days'] / results['total_days']
            
            # Check for outliers (values > 3 std from mean)
            valid_data = flow_data[~np.isnan(flow_data) & (flow_data >= 0)]
            if len(valid_data) > 10:
                mean_flow = np.mean(valid_data)
                std_flow = np.std(valid_data)
                outlier_threshold = mean_flow + 3 * std_flow
                results['outliers'] = np.sum(valid_data > outlier_threshold)
            
            # Assign quality flag
            if results['valid_ratio'] >= 0.9:
                results['quality_flag'] = 'EXCELLENT'
            elif results['valid_ratio'] >= 0.8:
                results['quality_flag'] = 'GOOD'
            elif results['valid_ratio'] >= min_valid_ratio:
                results['quality_flag'] = 'ACCEPTABLE'
            else:
                results['quality_flag'] = 'POOR'
            
            validation_results[station_name] = results
            
            self.logger.info(f"{station_name}: {results['quality_flag']} "
                           f"({results['valid_ratio']:.1%} valid data)")
        
        return validation_results
    
    def validate_parameter_ranges(self, parameters: Dict[str, float]) -> bool:
        """Validate parameter values are within acceptable ranges"""
        param_ranges = {
            'binfilt': (0.001, 0.4),
            'Ws': (0.7, 1.0),
            'Ds': (0.001, 1.0),
            'Dsmax': (5.0, 50.0),
            'soil_d1': (0.1, 0.3),
            'soil_d2': (0.5, 2.0),
            'soil_d3': (0.5, 2.0)
        }
        
        all_valid = True
        
        for param_name, value in parameters.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                if not (min_val <= value <= max_val):
                    self.logger.warning(f"Parameter {param_name}={value} outside "
                                      f"acceptable range [{min_val}, {max_val}]")
                    all_valid = False
        
        return all_valid
    
    def check_file_existence(self, file_paths: Dict[str, str]) -> Dict[str, bool]:
        """Check if required files exist"""
        file_status = {}
        
        for file_type, file_path in file_paths.items():
            exists = os.path.exists(file_path)
            file_status[file_type] = exists
            
            if exists:
                self.logger.debug(f"✓ {file_type}: {file_path}")
            else:
                self.logger.warning(f"✗ {file_type}: {file_path} (not found)")
        
        return file_status