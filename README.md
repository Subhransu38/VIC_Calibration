Enhanced VIC Multi-Station Calibration System
A comprehensive, production-ready calibration system for the Variable Infiltration Capacity (VIC) hydrological model with multi-station support for the Mahanadi Basin.
Features
🚀 Enhanced Capabilities

Multi-station calibration across 14 streamflow stations
Robust error handling with fallback mechanisms
Configurable parameters via INI files
Comprehensive logging with multiple levels
Parallel processing support
Automatic result analysis and visualization
Data validation and quality checks

📊 Performance Metrics

Nash-Sutcliffe Efficiency (NSE)
Root Mean Square Error (RMSE)
Correlation Coefficient (R)
Percent Bias (PBIAS)

🔧 Optimization Algorithms

SCE-UA (Shuffled Complex Evolution)
MCMC (Markov Chain Monte Carlo)
DREAM (Differential Evolution Adaptive Metropolis)

Installation
Prerequisites

Python 3.7+
VIC model executable
Fortran routing model executable
Required GIS data and observed streamflow data

Setup
bash# Clone or download the project
cd VIC_Calibration_Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_calibration.sh
chmod +x routing_model/rout
chmod +x input/vicNl

# Create necessary directories
mkdir -p {config,src/utilities,input/{observed_data,gis,forcing},output/{fluxes,calibration/{logs,results,plots},routing},scripts/{data_preparation,analysis,validation},tests,docs,backup}
Configuration
Main Configuration File: config/calibration_config.ini
The system uses a comprehensive configuration file that controls all aspects of the calibration:
ini[PATHS]
vic_executable = ../input/vicNl
routing_executable = ./routing_model/rout
observed_data = ../input/observed_data/Mahanadi_discharge_14stations.xlsx

[PARAMETERS]
binfilt_min = 0.001
binfilt_max = 0.4
# ... other parameters

[CALIBRATION]
algorithm = sceua
max_iterations = 500
processes = 4
Key Configuration Sections:

PATHS: File and directory locations
SIMULATION: Time periods and settings
PARAMETERS: Calibration parameter bounds
CALIBRATION: Optimization settings
VALIDATION: Validation criteria
LOGGING: Log configuration

Usage
Quick Start
bash# Basic calibration with default settings
./run_calibration.sh

# Custom calibration
./run_calibration.sh 1000 sceua 4  # 1000 iterations, SCE-UA, 4 processes

# Using Python directly
python calibration_enhanced.py --iterations 500 --algorithm sceua --processes 4
Command Line Options
bashpython calibration_enhanced.py \
    --iterations 500 \           # Number of iterations
    --algorithm sceua \          # Optimization algorithm
    --processes 4 \              # Parallel processes
    --config config/calibration_config.ini \  # Config file
    --log-level INFO             # Logging level
Advanced Usage
bash# Run with custom configuration
python calibration_enhanced.py --config my_custom_config.ini

# Debug mode with detailed logging
python calibration_enhanced.py --log-level DEBUG

# Generate summary after calibration
python scripts/analysis/generate_summary.py --config config/calibration_config.ini
Project Structure
VIC_Calibration_Project/
├── calibration_enhanced.py          # Main calibration script
├── format_soil_params_final.py      # Soil parameter formatting
├── run_calibration.sh               # Execution script
├── config/
│   └── calibration_config.ini       # Main configuration
├── src/utilities/                   # Utility modules
│   ├── data_validator.py           # Data validation
│   ├── post_processor.py           # Result processing
│   └── visualization.py            # Plotting utilities
├── input/                          # Input data
│   ├── observed_data/              # Streamflow observations
│   ├── gis/                        # GIS datasets
│   └── forcing/                    # Meteorological data
├── output/                         # Results
│   ├── calibration/                # Calibration outputs
│   │   ├── logs/                   # Log files
│   │   ├── results/                # Parameter results
│   │   └── plots/                  # Visualization
│   └── fluxes/                     # VIC model outputs
└── scripts/                        # Analysis scripts
    └── analysis/
        └── generate_summary.py      # Summary generation
Key Improvements Over Original
1. Configuration Management

✅ Centralized configuration via INI files
✅ No hardcoded paths
✅ Easy parameter modification

2. Error Handling

✅ Comprehensive try-catch blocks
✅ Graceful degradation with dummy data
✅ Detailed error logging
✅ File existence checks

3. Logging System

✅ Multi-level logging (DEBUG, INFO, WARNING, ERROR)
✅ File and console output
✅ Rotation and size management
✅ Timestamped entries

4. Performance Metrics

✅ Multiple objective functions
✅ Station-wise performance tracking
✅ Comprehensive metric calculation
✅ Quality assessment

5. Visualization & Analysis

✅ Automated plot generation
✅ Parameter evolution tracking
✅ Correlation analysis
✅ Summary report generation

6. Data Validation

✅ Input data quality checks
✅ Parameter range validation
✅ File existence verification
✅ Data consistency checks

7. Parallel Processing

✅ Multi-core support via SPOTPY
✅ Configurable process count
✅ Improved performance

Output Files
Calibration Results

best_parameters.json: Optimal parameter set
SCEUA_VIC_Mahanadi_enhanced.csv: Complete calibration history
calibration_summary_report.md: Summary report

Visualizations

parameter_evolution.png: Parameter progression
objective_function.png: Optimization progress
parameter_correlation.png: Parameter relationships
hydrographs_comparison.png: Observed vs simulated flows

Logs

calibration.log: Detailed execution log
vic.log: VIC model execution log

Troubleshooting
Common Issues

UH_s files not found
bash# The system automatically generates these, but if issues persist:
# Check routing model executable permissions
chmod +x routing_model/rout

VIC executable not found
bash# Ensure VIC is compiled and path is correct in config
# Check permissions
chmod +x input/vicNl

Observed data file missing
bash# System will create synthetic data for testing
# Check path in config file

Memory issues with large datasets
bash# Reduce number of parallel processes
python calibration_enhanced.py --processes 1


Debug Mode
bash# Run with maximum verbosity
python calibration_enhanced.py --log-level DEBUG
Performance Tips

Parallel Processing: Use multiple cores for faster calibration
bash./run_calibration.sh 500 sceua 4  # 4 processes

Parameter Ranges: Tighten parameter bounds based on prior knowledge
Iteration Count: Start with fewer iterations for testing
bash./run_calibration.sh 50 sceua 1  # Quick test run

Data Quality: Ensure high-quality observed data for better results

Validation
The system includes comprehensive validation capabilities:

Split-period validation: Calibrate on one period, validate on another
Cross-validation: Multiple calibration/validation splits
Performance thresholds: Automated quality assessment

Contributing

Follow Python PEP 8 style guidelines
Add comprehensive logging to new functions
Include error handling and validation
Update configuration files as needed
Add unit tests for new functionality