"""
Improved VIC Calibration with Multi-Station Support
Enhanced with better logging, configuration management, and error handling
"""

from __future__ import print_function
import os
import sys
import json
import logging
import datetime
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
from typing import Dict, List, Optional, Tuple
import spotpy
from format_soil_params_final import format_soil_params

# Configure logging
def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class CalibrationConfig:
    """Configuration management for VIC calibration"""
    
    def __init__(self, config_file='config/calibration_config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            logging.info(f"Loaded configuration from {self.config_file}")
        else:
            self.create_default_config()
            logging.info(f"Created default configuration at {self.config_file}")
    
    def create_default_config(self):
        """Create default configuration file"""
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Default configuration
        self.config['PATHS'] = {
            'base_dir': '.',
            'vic_executable': '../input/vicNl',
            'routing_executable': './rout',
            'input_dir': '../input',
            'output_dir': '../output',
            'observed_data': '../input/observed_data/Mahanadi_discharge_14stations.xlsx',
            'global_params': '../input/global.params',
            'soil_params': '../input/soil.param',
            'grid_raster': '../input/gis/Mahanadi_grid.tif',
            'elevation_raster': '../input/gis/Mahanadi_basin_ElvAvg.tif',
            'slope_raster': '../input/gis/Mahanadi_basin_SlopeAvg.tif',
            'precip_raster': '../input/gis/Mahanadi_basin_Precip.tif',
            'soil_raster': '../input/gis/Mahanadi_basin_SoilsAgg.tif',
            'flux_output': '../output/fluxes/calibrated/',
            'routing_output': './output/',
            'calibration_output': '../output/calibration'
        }
        
        self.config['SIMULATION'] = {
            'start_date': '2005-01-01',
            'end_date': '2009-12-31',
            'calibration_start': '2005-03-01',
            'calibration_end': '2009-12-31'
        }
        
        self.config['STATIONS'] = {
            'station_file': 'station.txt',
            'station_optimized_file': 'station_optimized.txt',
            'uh_directory': './src/'
        }
        
        self.config['PARAMETERS'] = {
            'binfilt_min': '0.001',
            'binfilt_max': '0.4',
            'ws_min': '0.7',
            'ws_max': '1.0',
            'ds_min': '0.001',
            'ds_max': '1.0',
            'dsmax_min': '5.0',
            'dsmax_max': '50.0',
            'soil_d1_min': '0.1',
            'soil_d1_max': '0.3',
            'soil_d2_min': '0.5',
            'soil_d2_max': '2.0',
            'soil_d3_min': '0.5',
            'soil_d3_max': '2.0',
            'velocity_fixed': '1.5',
            'diffusion_fixed': '800'
        }
        
        self.config['CALIBRATION'] = {
            'algorithm': 'sceua',
            'max_iterations': '100',
            'ngs': '5',
            'processes': '1',
            'objective_function': 'nse',
            'min_valid_days': '50'
        }
        
        # Save default configuration
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section, key, fallback=None):
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getfloat(self, section, key, fallback=None):
        """Get configuration value as float"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getint(self, section, key, fallback=None):
        """Get configuration value as integer"""
        return self.config.getint(section, key, fallback=fallback)

class MetricsCalculator:
    """Calculate various performance metrics"""
    
    @staticmethod
    def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency"""
        if len(observed) == 0 or len(simulated) == 0:
            return -999.0
        
        valid_mask = ~(np.isnan(observed) | np.isnan(simulated)) & (observed > 0)
        if np.sum(valid_mask) < 10:
            return -999.0
        
        obs_valid = observed[valid_mask]
        sim_valid = simulated[valid_mask]
        
        numerator = np.sum((obs_valid - sim_valid) ** 2)
        denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
        
        if denominator == 0:
            return -999.0
        
        return 1 - (numerator / denominator)
    
    @staticmethod
    def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        valid_mask = ~(np.isnan(observed) | np.isnan(simulated))
        if np.sum(valid_mask) == 0:
            return 999.0
        
        obs_valid = observed[valid_mask]
        sim_valid = simulated[valid_mask]
        
        return np.sqrt(np.mean((obs_valid - sim_valid) ** 2))
    
    @staticmethod
    def correlation_coefficient(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        valid_mask = ~(np.isnan(observed) | np.isnan(simulated))
        if np.sum(valid_mask) < 2:
            return 0.0
        
        obs_valid = observed[valid_mask]
        sim_valid = simulated[valid_mask]
        
        corr_matrix = np.corrcoef(obs_valid, sim_valid)
        return corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
    
    @staticmethod
    def percent_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Percent Bias"""
        valid_mask = ~(np.isnan(observed) | np.isnan(simulated))
        if np.sum(valid_mask) == 0:
            return 999.0
        
        obs_valid = observed[valid_mask]
        sim_valid = simulated[valid_mask]
        
        if np.sum(obs_valid) == 0:
            return 999.0
        
        return 100.0 * np.sum(sim_valid - obs_valid) / np.sum(obs_valid)
    
    @classmethod
    def calculate_all_metrics(cls, observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
        """Calculate all available metrics"""
        return {
            'NSE': cls.nash_sutcliffe_efficiency(observed, simulated),
            'RMSE': cls.rmse(observed, simulated),
            'R': cls.correlation_coefficient(observed, simulated),
            'PBIAS': cls.percent_bias(observed, simulated)
        }

class VICMultiStationModel:
    """Enhanced VIC multi-station model with improved error handling"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Parse dates
        self.start_date = datetime.datetime.strptime(
            config.get('SIMULATION', 'start_date'), '%Y-%m-%d').date()
        self.end_date = datetime.datetime.strptime(
            config.get('SIMULATION', 'end_date'), '%Y-%m-%d').date()
        
        self.observations = {}
        self.station_areas = {}
        
        # Station names
        self.station_names = [
            'KANTAMAL', 'KESINGA', 'SALEBHATA', 'ANDHIYARKHORE', 
            'BAMNIDHI', 'BARONDA', 'BASANTPUR', 'GHATORA', 
            'JONDHRA', 'KOTNI', 'KURUBHATA', 'RAJIM', 'RAMPUR', 'SIMGA'
        ]
        
        # Initialize directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.get('PATHS', 'output_dir'),
            self.config.get('PATHS', 'flux_output'),
            self.config.get('PATHS', 'routing_output'),
            self.config.get('PATHS', 'calibration_output'),
            self.config.get('STATIONS', 'uh_directory')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def load_observations(self) -> bool:
        """Load observed streamflow data with improved error handling"""
        obs_file = self.config.get('PATHS', 'observed_data')
        
        if not os.path.exists(obs_file):
            self.logger.warning(f"Observed data file not found: {obs_file}")
            self._create_synthetic_observations()
            return False
        
        try:
            self.logger.info(f"Loading observations from {obs_file}")
            obs_data = pd.ExcelFile(obs_file)
            obs_times = pd.date_range(
                self.config.get('SIMULATION', 'calibration_start'),
                self.config.get('SIMULATION', 'calibration_end'),
                freq='D'
            )
            
            successful_loads = 0
            
            for station_name in self.station_names:
                try:
                    if station_name in obs_data.sheet_names:
                        obs_sheet = obs_data.parse(station_name)
                        obs_series = obs_sheet.iloc[:, 1].values
                    else:
                        obs_sheet = obs_data.parse('Daily')
                        if station_name in obs_sheet.columns:
                            obs_series = obs_sheet[station_name].values
                        else:
                            raise ValueError(f"No data found for {station_name}")
                    
                    # Ensure correct length
                    if len(obs_series) > len(obs_times):
                        obs_series = obs_series[:len(obs_times)]
                    elif len(obs_series) < len(obs_times):
                        obs_series = np.pad(obs_series, (0, len(obs_times) - len(obs_series)), 
                                          constant_values=np.nan)
                    
                    # Remove negative values and replace with NaN
                    obs_series = np.where(obs_series < 0, np.nan, obs_series)
                    
                    self.observations[station_name] = obs_series
                    successful_loads += 1
                    
                    self.logger.debug(f"Loaded {station_name}: {len(obs_series)} days, "
                                    f"valid: {np.sum(~np.isnan(obs_series))}")
                    
                except Exception as e:
                    self.logger.warning(f"Error loading data for {station_name}: {e}")
                    self._create_synthetic_station_data(station_name, len(obs_times))
            
            self.logger.info(f"Successfully loaded observations for {successful_loads}/{len(self.station_names)} stations")
            return successful_loads > 0
            
        except Exception as e:
            self.logger.error(f"Error reading observation file: {e}")
            self._create_synthetic_observations()
            return False
    
    def _create_synthetic_observations(self):
        """Create synthetic observations for testing"""
        self.logger.warning("Creating synthetic observations for testing")
        obs_times = pd.date_range(
            self.config.get('SIMULATION', 'calibration_start'),
            self.config.get('SIMULATION', 'calibration_end'),
            freq='D'
        )
        
        for station_name in self.station_names:
            self._create_synthetic_station_data(station_name, len(obs_times))
    
    def _create_synthetic_station_data(self, station_name: str, n_days: int):
        """Create synthetic data for a single station"""
        # Create realistic synthetic data with seasonal patterns
        base_flow = np.random.normal(50, 20, n_days)
        seasonal_component = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        synthetic_data = base_flow + seasonal_component
        synthetic_data = np.maximum(synthetic_data, 0.1)  # Minimum flow
        
        self.observations[station_name] = synthetic_data
        self.logger.debug(f"Created synthetic data for {station_name}")
    
    def load_station_areas(self) -> bool:
        """Load station catchment areas"""
        station_file = self.config.get('STATIONS', 'station_optimized_file')
        if not os.path.exists(station_file):
            station_file = self.config.get('STATIONS', 'station_file')
        
        if not os.path.exists(station_file):
            self.logger.warning(f"Station file not found: {station_file}")
            self._create_default_station_areas()
            return False
        
        try:
            self.logger.info(f"Loading station areas from {station_file}")
            with open(station_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip() and not line.startswith('NONE') and not line.startswith('/'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        station_name = parts[1]
                        try:
                            catchment_area = float(parts[4])
                            self.station_areas[station_name] = catchment_area
                            self.logger.debug(f"{station_name}: {catchment_area} km²")
                        except ValueError:
                            self.logger.warning(f"Invalid area for {station_name}: {parts[4]}")
            
            self.logger.info(f"Loaded areas for {len(self.station_areas)} stations")
            return len(self.station_areas) > 0
            
        except Exception as e:
            self.logger.error(f"Error reading station areas: {e}")
            self._create_default_station_areas()
            return False
    
    def _create_default_station_areas(self):
        """Create default station areas"""
        self.logger.warning("Using default station areas (1000 km² each)")
        for station_name in self.station_names:
            self.station_areas[station_name] = 1000.0
    
    def ensure_uh_files_exist(self) -> bool:
        """Ensure UH_s files exist for routing"""
        uh_dir = self.config.get('STATIONS', 'uh_directory')
        
        if os.path.exists(uh_dir):
            uh_files = [f for f in os.listdir(uh_dir) if f.endswith('.uh_s')]
            if len(uh_files) >= len(self.station_names):
                self.logger.info(f"Found {len(uh_files)} existing UH_s files")
                return True
        
        self.logger.info("UH_s files not found. Generating them...")
        return self._generate_uh_files()
    
    def _generate_uh_files(self) -> bool:
        """Generate UH_s files using routing model"""
        try:
            # Create routing input file
            routing_input_file = self._create_routing_input_file(
                velocity=self.config.getfloat('PARAMETERS', 'velocity_fixed'),
                diffusion=self.config.getfloat('PARAMETERS', 'diffusion_fixed'),
                use_optimized_stations=False
            )
            
            # Run routing model
            routing_executable = self.config.get('PATHS', 'routing_executable')
            result = subprocess.run(
                [routing_executable, routing_input_file],
                capture_output=True, text=True, cwd='.'
            )
            
            if result.returncode != 0:
                self.logger.error(f"Error generating UH_s files: {result.stderr}")
                return False
            
            # Check if files were created
            uh_dir = self.config.get('STATIONS', 'uh_directory')
            if os.path.exists(uh_dir):
                uh_files = [f for f in os.listdir(uh_dir) if f.endswith('.uh_s')]
                if len(uh_files) >= len(self.station_names):
                    self.logger.info(f"Successfully generated {len(uh_files)} UH_s files")
                    self._create_optimized_station_file()
                    return True
            
            self.logger.error("Failed to generate UH_s files")
            return False
            
        except Exception as e:
            self.logger.error(f"Exception while generating UH_s files: {e}")
            return False
    
    def _create_routing_input_file(self, velocity: float, diffusion: float, 
                                 use_optimized_stations: bool = True) -> str:
        """Create routing input file"""
        routing_input_file = 'routing_input_calibration.txt'
        
        # Updated flux path to match your file naming convention
        # Your files are like: fluxes_19.2871_82.7093, fluxes_19.3371_82.8093
        # So the prefix should be the path + "fluxes_" without the trailing "fluxes_"
        flux_base_path = self.config.get('PATHS', 'flux_output')
        flux_prefix = os.path.join(flux_base_path, 'fluxes_')
        
        station_file = (self.config.get('STATIONS', 'station_optimized_file') 
                       if use_optimized_stations and os.path.exists(self.config.get('STATIONS', 'station_optimized_file'))
                       else self.config.get('STATIONS', 'station_file'))
        
        with open(routing_input_file, 'w') as f:
            f.write(f"ROUT_PARAM              {os.path.abspath('fdr.txt')}\n")
            f.write(f"STATIONS                {os.path.abspath(station_file)}\n")
            f.write(f"FRACT_PARAM             {os.path.abspath('fraction.txt')}\n")
            f.write(f"IN_RUNOFF               {flux_prefix}\n")  # This will be: /path/to/fluxes/fluxes_
            f.write(f"OUT_STREAMFLOW          {self.config.get('PATHS', 'routing_output')}calibration_routing_output\n")
            f.write(f"START_YEAR              {self.start_date.year}\n")
            f.write(f"START_MONTH             {self.start_date.month}\n")
            f.write(f"START_DAY               {self.start_date.day}\n")
            f.write(f"N_DAYS                  {(self.end_date - self.start_date).days + 1}\n")
            f.write(f"VELOCITY                {velocity}\n")
            f.write(f"DIFFUSION               {diffusion}\n")
        
        self.logger.debug(f"Created routing input file with flux prefix: {flux_prefix}")
        return routing_input_file
    
    def _create_optimized_station_file(self):
        """Create optimized station file with UH_s paths"""
        self.logger.info("Creating optimized station file...")
        
        base_path = os.path.abspath(self.config.get('STATIONS', 'uh_directory'))
        optimized_file = self.config.get('STATIONS', 'station_optimized_file')
        
        with open(optimized_file, 'w') as f:
            for station_name in self.station_names:
                uh_file_path = os.path.join(base_path, f'{station_name}.uh_s')
                f.write(f"1\t{station_name}\t20.0\t85.0\t{self.station_areas.get(station_name, 1000)}\n")
                f.write(f"{uh_file_path}\n")
        
        self.logger.info(f"Created {optimized_file}")
    
    def run_simulation(self, parameters: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Run VIC and routing simulation with given parameters"""
        try:
            self.logger.info("Starting VIC simulation")
            self.logger.debug(f"Parameters: {parameters}")
            
            # Run VIC model
            if not self._run_vic_model(parameters):
                self.logger.warning("VIC model failed, returning dummy outputs")
                return self._create_dummy_outputs()
            
            # Run routing model
            station_flows = self._run_routing_model(parameters)
            
            if not station_flows:
                self.logger.warning("Routing model failed, returning dummy outputs")
                return self._create_dummy_outputs()
            
            self.logger.info(f"Simulation completed successfully for {len(station_flows)} stations")
            return station_flows
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return self._create_dummy_outputs()
    
    def _run_vic_model(self, parameters: Dict[str, float]) -> bool:
        """Run VIC model with given parameters"""
        try:
            # Create soil parameter file
            format_soil_params(
                self.config.get('PATHS', 'grid_raster'),
                self.config.get('PATHS', 'soil_raster'),
                self.config.get('PATHS', 'elevation_raster'),
                self.config.get('PATHS', 'precip_raster'),
                self.config.get('PATHS', 'slope_raster'),
                self.config.get('PATHS', 'soil_params'),
                parameters['binfilt'], parameters['Ws'], parameters['Ds'],
                parameters['Dsmax'], parameters['soil_d2'], parameters['soil_d3']
            )
            
            # Run VIC
            vic_executable = self.config.get('PATHS', 'vic_executable')
            global_file = self.config.get('PATHS', 'global_params')
            
            result = os.system(f'{vic_executable} -g {global_file} 2> {self.config.get("PATHS", "output_dir")}/vic.log')
            
            if result != 0:
                self.logger.warning("VIC execution returned non-zero status")
                return False
            
            # Check for output files
            flux_dir = self.config.get('PATHS', 'flux_output')
            flux_files = [f for f in os.listdir(flux_dir) if f.startswith('fluxes_')]
            
            if not flux_files:
                self.logger.warning("No VIC flux files found")
                return False
            
            self.logger.debug(f"VIC produced {len(flux_files)} flux files")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running VIC model: {e}")
            return False
    
    def _run_routing_model(self, parameters: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Run routing model and parse outputs"""
        try:
            # Create routing input file
            routing_input_file = self._create_routing_input_file(
                velocity=parameters.get('velocity', self.config.getfloat('PARAMETERS', 'velocity_fixed')),
                diffusion=parameters.get('diffusion', self.config.getfloat('PARAMETERS', 'diffusion_fixed'))
            )
            
            # Run routing
            routing_executable = self.config.get('PATHS', 'routing_executable')
            result = subprocess.run(
                [routing_executable, routing_input_file],
                capture_output=True, text=True, cwd='.'
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Routing execution failed: {result.stderr}")
                return {}
            
            # Parse outputs
            return self._parse_routing_outputs()
            
        except Exception as e:
            self.logger.error(f"Error running routing model: {e}")
            return {}
    
    def _parse_routing_outputs(self) -> Dict[str, np.ndarray]:
        """Parse routing model outputs and convert to cumecs"""
        station_flows = {}
        output_dir = self.config.get('PATHS', 'routing_output')
        
        for station_name in self.station_names:
            try:
                day_mm_file = os.path.join(output_dir, f'{station_name}.day_mm')
                
                if os.path.exists(day_mm_file):
                    discharge_cumecs = self._parse_day_mm_file(day_mm_file, station_name)
                    if len(discharge_cumecs) > 0:
                        station_flows[station_name] = discharge_cumecs
                        self.logger.debug(f"{station_name}: {len(discharge_cumecs)} days, "
                                        f"range: {np.min(discharge_cumecs):.2f}-{np.max(discharge_cumecs):.2f} cumecs")
                    else:
                        station_flows[station_name] = self._create_dummy_station_output()
                else:
                    self.logger.warning(f"Output file not found: {day_mm_file}")
                    station_flows[station_name] = self._create_dummy_station_output()
                    
            except Exception as e:
                self.logger.error(f"Error parsing output for {station_name}: {e}")
                station_flows[station_name] = self._create_dummy_station_output()
        
        return station_flows
    
    def _parse_day_mm_file(self, day_mm_file: str, station_name: str) -> np.ndarray:
        """Parse .day_mm file and convert to cumecs"""
        try:
            catchment_area_km2 = self.station_areas.get(station_name, 1000.0)
            discharge_mm_data = []
            
            with open(day_mm_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            try:
                                streamflow_mm = float(parts[3])
                                discharge_mm_data.append(streamflow_mm)
                            except ValueError:
                                continue
            
            if len(discharge_mm_data) == 0:
                return np.array([])
            
            # Convert mm/day to cumecs: mm/day * area_km2 * 1000 / 86400
            discharge_mm_array = np.array(discharge_mm_data)
            discharge_cumecs = discharge_mm_array * catchment_area_km2 * 1000.0 / 86400.0
            
            return discharge_cumecs
            
        except Exception as e:
            self.logger.error(f"Error parsing {day_mm_file}: {e}")
            return np.array([])
    
    def _create_dummy_outputs(self) -> Dict[str, np.ndarray]:
        """Create dummy outputs when simulation fails"""
        dummy_outputs = {}
        for station_name in self.station_names:
            dummy_outputs[station_name] = self._create_dummy_station_output()
        return dummy_outputs
    
    def _create_dummy_station_output(self) -> np.ndarray:
        """Create dummy output for a single station"""
        n_days = (self.end_date - self.start_date).days + 1
        return np.full(n_days, 10.0)

class SPOTPYCalibrationSetup:
    """Enhanced SPOTPY calibration setup"""
    
    def __init__(self, config_file: str = 'config/calibration_config.ini'):
        self.config = CalibrationConfig(config_file)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize VIC model
        self.vic_model = VICMultiStationModel(self.config)
        
        # Load observations and station areas
        self.vic_model.load_observations()
        self.vic_model.load_station_areas()
        
        # Ensure UH files exist
        if not self.vic_model.ensure_uh_files_exist():
            self.logger.error("Failed to ensure UH_s files exist")
            raise RuntimeError("Cannot initialize calibration without UH_s files")
        
        # Define parameters
        self.params = self._define_parameters()
        
        self.logger.info("Calibration setup initialized successfully")
    
    def _define_parameters(self) -> List[spotpy.parameter.Base]:
        """Define calibration parameters"""
        params = [
            spotpy.parameter.Uniform('binfilt', 
                                   self.config.getfloat('PARAMETERS', 'binfilt_min'),
                                   self.config.getfloat('PARAMETERS', 'binfilt_max')),
            spotpy.parameter.Uniform('Ws',
                                   self.config.getfloat('PARAMETERS', 'ws_min'),
                                   self.config.getfloat('PARAMETERS', 'ws_max')),
            spotpy.parameter.Uniform('Ds',
                                   self.config.getfloat('PARAMETERS', 'ds_min'),
                                   self.config.getfloat('PARAMETERS', 'ds_max')),
            spotpy.parameter.Uniform('Dsmax',
                                   self.config.getfloat('PARAMETERS', 'dsmax_min'),
                                   self.config.getfloat('PARAMETERS', 'dsmax_max')),
            spotpy.parameter.Uniform('soil_d1',
                                   self.config.getfloat('PARAMETERS', 'soil_d1_min'),
                                   self.config.getfloat('PARAMETERS', 'soil_d1_max')),
            spotpy.parameter.Uniform('soil_d2',
                                   self.config.getfloat('PARAMETERS', 'soil_d2_min'),
                                   self.config.getfloat('PARAMETERS', 'soil_d2_max')),
            spotpy.parameter.Uniform('soil_d3',
                                   self.config.getfloat('PARAMETERS', 'soil_d3_min'),
                                   self.config.getfloat('PARAMETERS', 'soil_d3_max'))
        ]
        
        return params
    
    def parameters(self):
        """Return parameter generators for SPOTPY"""
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, vector):
        """Run simulation with parameter vector"""
        # Create parameter dictionary
        param_dict = {
            'binfilt': vector[0],
            'Ws': vector[1],
            'Ds': vector[2],
            'Dsmax': vector[3],
            'soil_d1': vector[4],
            'soil_d2': vector[5],
            'soil_d3': vector[6]
        }
        
        # Run simulation
        station_flows = self.vic_model.run_simulation(param_dict)
        
        # Combine all station flows into single array for SPOTPY
        combined_sim = []
        station_names = sorted(station_flows.keys())
        
        for station_name in station_names:
            flow_data = station_flows[station_name]
            combined_sim.extend(flow_data)
        
        return np.array(combined_sim)
    
    def evaluation(self, evaldates=False):
        """Get observed data for all stations"""
        combined_obs = []
        station_names = sorted(self.vic_model.observations.keys())
        
        for station_name in station_names:
            obs_data = self.vic_model.observations[station_name]
            combined_obs.extend(obs_data)
        
        return np.array(combined_obs)
    
    def objectivefunction(self, simulation, evaluation):
        """Multi-objective function considering all stations"""
        # Split combined arrays back to individual stations
        station_names = sorted(self.vic_model.observations.keys())
        n_days = len(evaluation) // len(station_names)
        
        total_objective = 0
        valid_stations = 0
        station_metrics = {}
        
        objective_func = self.config.get('CALIBRATION', 'objective_function', 'nse')
        min_valid_days = self.config.getint('CALIBRATION', 'min_valid_days', 50)
        
        for i, station_name in enumerate(station_names):
            start_idx = i * n_days
            end_idx = (i + 1) * n_days
            
            sim_station = simulation[start_idx:end_idx]
            obs_station = evaluation[start_idx:end_idx]
            
            # Calculate metrics for this station
            metrics = MetricsCalculator.calculate_all_metrics(obs_station, sim_station)
            station_metrics[station_name] = metrics
            
            # Check if station has enough valid data
            valid_mask = ~(np.isnan(sim_station) | np.isnan(obs_station)) & (obs_station > 0)
            
            if np.sum(valid_mask) >= min_valid_days:
                if objective_func.lower() == 'nse':
                    objective_value = metrics['NSE']
                elif objective_func.lower() == 'rmse':
                    objective_value = -metrics['RMSE']  # Negative because SPOTPY maximizes
                elif objective_func.lower() == 'r':
                    objective_value = metrics['R']
                else:
                    objective_value = metrics['NSE']  # Default to NSE
                
                if objective_value > -900:  # Valid metric
                    total_objective += objective_value
                    valid_stations += 1
        
        # Log station-wise performance
        if hasattr(self, '_iteration_count'):
            self._iteration_count += 1
        else:
            self._iteration_count = 1
        
        if self._iteration_count % 10 == 0:  # Log every 10 iterations
            self.logger.info(f"Iteration {self._iteration_count}: Valid stations: {valid_stations}")
            for station, metrics in station_metrics.items():
                if metrics['NSE'] > -900:
                    self.logger.debug(f"  {station}: NSE={metrics['NSE']:.3f}, R={metrics['R']:.3f}")
        
        # Return average objective value across all valid stations
        if valid_stations > 0:
            avg_objective = total_objective / valid_stations
        else:
            avg_objective = -999
        
        return avg_objective

class CalibrationRunner:
    """Main calibration runner with enhanced features"""
    
    def __init__(self, config_file: str = 'config/calibration_config.ini', log_level: str = 'INFO'):
        self.config = CalibrationConfig(config_file)
        
        # Setup logging
        log_file = os.path.join(self.config.get('PATHS', 'calibration_output'), 'calibration.log')
        self.logger = setup_logging(log_level, log_file)
        
        self.logger.info("=" * 60)
        self.logger.info("VIC Multi-Station Calibration for Mahanadi Basin")
        self.logger.info("Enhanced Version with Improved Error Handling")
        self.logger.info("=" * 60)
    
    def run_calibration(self, n_iterations: Optional[int] = None, 
                       algorithm: Optional[str] = None,
                       processes: Optional[int] = None):
        """Run the calibration process"""
        try:
            # Get parameters
            n_iterations = n_iterations or self.config.getint('CALIBRATION', 'max_iterations')
            algorithm = algorithm or self.config.get('CALIBRATION', 'algorithm')
            processes = processes or self.config.getint('CALIBRATION', 'processes')
            
            self.logger.info(f"Starting calibration with {n_iterations} iterations using {algorithm}")
            self.logger.info(f"Using {processes} process(es)")
            
            # Initialize calibration setup
            cal_setup = SPOTPYCalibrationSetup(self.config.config_file)
            
            # Output file
            output_file = os.path.join(
                self.config.get('PATHS', 'calibration_output'),
                f'{algorithm.upper()}_VIC_Mahanadi_enhanced'
            )
            
            # Initialize algorithm
            if algorithm.lower() == 'sceua':
                sampler = spotpy.algorithms.sceua(
                    cal_setup, 
                    dbname=output_file, 
                    dbformat='csv'
                )
            elif algorithm.lower() == 'mcmc':
                sampler = spotpy.algorithms.mcmc(
                    cal_setup,
                    dbname=output_file,
                    dbformat='csv'
                )
            elif algorithm.lower() == 'dream':
                sampler = spotpy.algorithms.dream(
                    cal_setup,
                    dbname=output_file,
                    dbformat='csv'
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Run calibration
            if processes > 1:
                self.logger.info("Running parallel calibration")
                sampler.sample(
                    n_iterations, 
                    ngs=self.config.getint('CALIBRATION', 'ngs', 5),
                    processes=processes
                )
            else:
                self.logger.info("Running sequential calibration")
                sampler.sample(
                    n_iterations,
                    ngs=self.config.getint('CALIBRATION', 'ngs', 5)
                )
            
            self.logger.info("Calibration completed successfully!")
            
            # Analyze results
            self._analyze_results(output_file + '.csv')
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _analyze_results(self, results_file: str):
        """Analyze calibration results"""
        try:
            if not os.path.exists(results_file):
                self.logger.warning(f"Results file not found: {results_file}")
                return
            
            self.logger.info("Analyzing calibration results...")
            
            # Load results
            results = pd.read_csv(results_file)
            
            # Find best parameters
            best_idx = results['like1'].idxmax()
            best_params = results.iloc[best_idx]
            
            self.logger.info("Best parameter set:")
            param_names = ['binfilt', 'Ws', 'Ds', 'Dsmax', 'soil_d1', 'soil_d2', 'soil_d3']
            for i, param_name in enumerate(param_names):
                param_col = f'par{param_name}' if f'par{param_name}' in results.columns else f'par{i}'
                if param_col in results.columns:
                    self.logger.info(f"  {param_name}: {best_params[param_col]:.6f}")
            
            self.logger.info(f"  Objective value: {best_params['like1']:.6f}")
            
            # Save best parameters
            best_params_file = os.path.join(
                self.config.get('PATHS', 'calibration_output'),
                'best_parameters.json'
            )
            
            best_params_dict = {}
            for i, param_name in enumerate(param_names):
                param_col = f'par{param_name}' if f'par{param_name}' in results.columns else f'par{i}'
                if param_col in results.columns:
                    best_params_dict[param_name] = float(best_params[param_col])
            
            best_params_dict['objective_value'] = float(best_params['like1'])
            best_params_dict['iteration'] = int(best_idx)
            
            with open(best_params_file, 'w') as f:
                json.dump(best_params_dict, f, indent=2)
            
            self.logger.info(f"Best parameters saved to: {best_params_file}")
            
            # Basic statistics
            self.logger.info(f"Calibration statistics:")
            self.logger.info(f"  Total iterations: {len(results)}")
            self.logger.info(f"  Best objective: {results['like1'].max():.6f}")
            self.logger.info(f"  Mean objective: {results['like1'].mean():.6f}")
            self.logger.info(f"  Std objective: {results['like1'].std():.6f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced VIC Multi-Station Calibration')
    parser.add_argument('--iterations', '-n', type=int, default=None,
                       help='Number of calibration iterations')
    parser.add_argument('--algorithm', '-a', type=str, default=None,
                       choices=['sceua', 'mcmc', 'dream'],
                       help='Optimization algorithm')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='Number of parallel processes')
    parser.add_argument('--config', '-c', type=str, default='config/calibration_config.ini',
                       help='Configuration file path')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run calibration
        runner = CalibrationRunner(args.config, args.log_level)
        runner.run_calibration(
            n_iterations=args.iterations,
            algorithm=args.algorithm,
            processes=args.processes
        )
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()