# Enhanced VIC Multi-Station Calibration System

A comprehensive, production-ready calibration system for the Variable Infiltration Capacity (VIC) hydrological model with multi-station support for the Mahanadi Basin.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![VIC](https://img.shields.io/badge/VIC-4.2+-green.svg)](https://vic.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start

### Prerequisites Checklist
- [ ] Python 3.7 or higher
- [ ] VIC model executable (`vicNl`)
- [ ] Fortran routing model executable (`rout`)
- [ ] Forcing files in correct format
- [ ] Observed streamflow data
- [ ] GIS datasets (grid, elevation, slope, etc.)

### 1-Minute Setup
```bash
# 1. Clone/download and navigate to project directory
cd VIC_Calibration_Project

# 2. Run the setup script
bash quick_setup.sh

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy your data files (see detailed instructions below)

# 5. Verify everything is ready
python verify_forcing_files.py --forcing-dir input/forcing/ --verbose

# 6. Start calibration
./run_calibration.sh
```

## 📁 Project Structure Overview

```
VIC_Calibration_Project/
├── 🔧 calibration_enhanced.py          # Main calibration script
├── 🔧 verify_forcing_files.py          # Data verification tool
├── 🔧 run_calibration.sh               # Easy execution script
├── 📁 config/                          # Configuration files
├── 📁 input/                           # Your input data
├── 📁 output/                          # Results and outputs
├── 📁 routing_model/                   # Routing model files
└── 📁 scripts/                         # Analysis utilities
```

## 📋 Detailed Setup Instructions

### Step 1: Initial Setup
```bash
# Create project directory
mkdir VIC_Calibration_Project
cd VIC_Calibration_Project

# Download/copy all the enhanced calibration files
# (calibration_enhanced.py, verify_forcing_files.py, etc.)

# Run setup script to create directory structure
bash quick_setup.sh
```

### Step 2: Install Dependencies
```bash
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using conda
conda create -n vic_calibration python=3.8
conda activate vic_calibration
pip install -r requirements.txt

# Option C: Using virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Copy Your Data Files

#### 3.1 Forcing Files
```bash
# Your forcing files should be named like: forcing_19.2871_82.7093
# Each file contains 4 columns: rain tmax tmin wind (tab/space separated)
cp /path/to/your/forcing_* input/forcing/

# Verify forcing files
ls input/forcing/
# Should show: forcing_19.2871_82.7093, forcing_19.2871_82.7593, etc.
```

#### 3.2 Observed Streamflow Data
```bash
# Copy your observed discharge data (Excel file with 14 stations)
cp /path/to/Mahanadi_discharge_14stations.xlsx input/observed_data/
```

#### 3.3 GIS Data
```bash
# Copy all required GIS files
cp /path/to/gis_data/* input/gis/
# Required files:
# - Mahanadi_grid.tif
# - Mahanadi_basin_ElvAvg.tif  
# - Mahanadi_basin_SlopeAvg.tif
# - Mahanadi_basin_Precip.tif
# - Mahanadi_basin_SoilsAgg.tif
```

#### 3.4 Model Executables
```bash
# Copy VIC executable
cp /path/to/your/vicNl input/vicNl
chmod +x input/vicNl

# Copy routing model executable  
cp /path/to/your/rout routing_model/rout
chmod +x routing_model/rout
```

#### 3.5 Routing Model Files
```bash
# Copy routing configuration files
cp /path/to/your/station.txt routing_model/
cp /path/to/your/fdr.txt routing_model/
cp /path/to/your/fraction.txt routing_model/
```

### Step 4: Configure the System

#### 4.1 Update Main Configuration
Edit `config/calibration_config.ini`:
```ini
[PATHS]
# Update these paths to match your system
vic_executable = /absolute/path/to/input/vicNl
routing_executable = /absolute/path/to/routing_model/rout
observed_data = /absolute/path/to/input/observed_data/Mahanadi_discharge_14stations.xlsx
# ... update other paths as needed
```

#### 4.2 Create VIC Global Parameters File
```bash
# Copy the template
cp input/global_template.params input/global.params

# Edit input/global.params and update these key lines:
# FORCING1        /absolute/path/to/input/forcing/forcing_
# SOIL            /absolute/path/to/input/soil.param  
# RESULT_DIR      /absolute/path/to/output/fluxes/calibrated/
```

**Important**: Use absolute paths in the global.params file!

### Step 5: Verify Your Setup
```bash
# Run comprehensive verification
python verify_forcing_files.py \
    --forcing-dir input/forcing/ \
    --flux-dir output/fluxes/calibrated/ \
    --config config/calibration_config.ini \
    --create-global-params \
    --verbose

# Check the verification report
cat verification_output/forcing_analysis_report.md
```

Expected output:
```
✓ Found X forcing files
✓ All files have consistent 4 columns  
✓ Configuration file found
✓ VIC executable found
✓ Routing executable found
```

## 🏃‍♂️ Running the Calibration

### Method 1: Simple Execution (Recommended)
```bash
# Basic run with default settings (500 iterations, SCE-UA algorithm)
./run_calibration.sh

# Custom run with specific parameters
./run_calibration.sh [iterations] [algorithm] [processes]

# Examples:
./run_calibration.sh 1000 sceua 4    # 1000 iterations, SCE-UA, 4 CPU cores
./run_calibration.sh 500 mcmc 2      # 500 iterations, MCMC, 2 cores  
./run_calibration.sh 200 dream 1     # 200 iterations, DREAM, 1 core
```

### Method 2: Direct Python Execution
```bash
# Full control with all options
python calibration_enhanced.py \
    --iterations 500 \
    --algorithm sceua \
    --processes 4 \
    --config config/calibration_config.ini \
    --log-level INFO

# Quick test run (10 iterations for testing)
python calibration_enhanced.py --iterations 10 --log-level DEBUG

# Available algorithms: sceua, mcmc, dream
# Available log levels: DEBUG, INFO, WARNING, ERROR
```

### Method 3: Step-by-Step Execution

#### 3.1 Test VIC Model First
```bash
# Test VIC model execution
./input/vicNl -g input/global.params

# Check if flux files are generated
ls output/fluxes/calibrated/
# Should show: fluxes_19.2871_82.7093, fluxes_19.2871_82.7593, etc.
```

#### 3.2 Test Routing Model
```bash
# The calibration script will automatically test routing, but you can test manually:
# (This step is usually handled automatically)
```

#### 3.3 Run Calibration
```bash
# Start calibration after verifying VIC works
./run_calibration.sh 100 sceua 2  # Start with smaller number for testing
```

## 📊 Monitoring Progress

### Real-Time Monitoring
```bash
# Watch calibration progress
tail -f output/calibration/logs/calibration.log

# Monitor objective function evolution
watch -n 30 "tail -10 output/calibration/logs/calibration.log | grep 'Iteration'"

# Check current results
wc -l output/calibration/results/*.csv  # Shows number of completed iterations
```

### Progress Indicators
Look for these messages in the log:
- `✓ Found X existing UH_s files` - Routing setup OK
- `✓ VIC produced X flux files` - VIC model working
- `Successfully routed to X stations` - Routing working
- `Iteration XXX: Valid stations: XX` - Calibration progressing

## 📈 Understanding Results

### Output Files Location
```
output/calibration/
├── 📁 logs/
│   ├── calibration.log              # Detailed execution log
│   └── vic.log                      # VIC model execution log
├── 📁 results/
│   ├── SCEUA_VIC_Mahanadi_enhanced.csv  # Complete calibration history
│   ├── best_parameters.json         # Optimal parameter set
│   └── calibration_summary_report.md    # Summary report
└── 📁 plots/
    ├── parameter_evolution.png      # Parameter progression over time
    ├── objective_function.png       # Optimization progress
    └── parameter_correlation.png    # Parameter relationships
```

### Key Results Files

#### 1. Best Parameters (`best_parameters.json`)
```json
{
  "binfilt": 0.123456,     # Infiltration parameter
  "Ws": 0.856789,          # Baseflow fraction
  "Ds": 0.234567,          # Baseflow nonlinearity  
  "Dsmax": 15.678901,      # Maximum baseflow (mm/day)
  "soil_d1": 0.198765,     # Layer 1 depth (m)
  "soil_d2": 1.234567,     # Layer 2 depth (m)
  "soil_d3": 1.567890,     # Layer 3 depth (m)
  "objective_value": 0.785432,  # NSE value
  "iteration": 342         # Best iteration number
}
```

#### 2. Performance Interpretation
- **NSE > 0.75**: Excellent performance ⭐⭐⭐
- **NSE 0.65-0.75**: Very good performance ⭐⭐
- **NSE 0.50-0.65**: Good performance ⭐
- **NSE < 0.50**: Poor performance - needs investigation

### Generate Analysis Reports
```bash
# Generate comprehensive analysis (automatic plots and summary)
python scripts/analysis/generate_summary.py --config config/calibration_config.ini

# Check generated plots
ls output/calibration/plots/
# Shows: parameter_evolution.png, objective_function.png, etc.
```

## 🔧 Configuration Options

### Key Configuration Sections

#### Parameter Bounds (`config/calibration_config.ini`)
```ini
[PARAMETERS]
# Adjust these based on your basin characteristics
binfilt_min = 0.001      # Infiltration parameter range
binfilt_max = 0.4
ws_min = 0.7             # Baseflow parameters
ws_max = 1.0
# ... other parameters
```

#### Optimization Settings
```ini
[CALIBRATION]
algorithm = sceua        # Options: sceua, mcmc, dream
max_iterations = 500     # Number of iterations
processes = 4            # Parallel processes (adjust based on your CPU)
objective_function = nse # Options: nse, rmse, r
```

#### Simulation Period
```ini
[SIMULATION]
start_date = 2005-01-01      # VIC simulation start
end_date = 2009-12-31        # VIC simulation end  
calibration_start = 2005-03-01  # Calibration period start
calibration_end = 2009-12-31    # Calibration period end
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No forcing files found"
```bash
# Check forcing directory
ls input/forcing/forcing_*

# If empty, copy your files:
cp /path/to/your/forcing_* input/forcing/

# Verify file format (should have 4 columns)
head -5 input/forcing/forcing_19.2871_82.7093
```

#### Issue 2: "VIC execution failed"
```bash
# Check VIC executable
./input/vicNl -h

# Check VIC global parameters file
cat input/global.params

# Common fixes:
chmod +x input/vicNl
# Update absolute paths in global.params
# Check forcing file format
```

#### Issue 3: "UH_s files not found"
```bash
# Check routing executable
./routing_model/rout

# Check station file
cat routing_model/station.txt

# The system will automatically generate UH_s files
# If it fails, check routing model setup
```

#### Issue 4: "No flux files generated"
```bash
# Check VIC log for errors
cat output/calibration/logs/vic.log

# Common issues:
# - Wrong paths in global.params
# - Incorrect forcing file format
# - Missing soil parameter file
```

#### Issue 5: Memory errors
```bash
# Reduce parallel processes
./run_calibration.sh 500 sceua 1  # Use only 1 process

# Or modify config file:
# processes = 1
```

### Debug Mode
```bash
# Run in debug mode for maximum information
python calibration_enhanced.py \
    --iterations 1 \
    --log-level DEBUG \
    --processes 1

# This will show detailed information about each step
```

### Getting Help
1. **Check log files first**: `output/calibration/logs/calibration.log`
2. **Run verification script**: `python verify_forcing_files.py --verbose`
3. **Test components individually**: VIC model, then routing, then calibration
4. **Use debug mode**: Maximum verbosity to identify issues

## 📊 Advanced Usage

### Parallel Processing
```bash
# Use multiple CPU cores for faster calibration
./run_calibration.sh 1000 sceua 8  # 8 processes

# Monitor CPU usage
htop  # or top
```

### Different Algorithms
```bash
# SCE-UA (fastest, good for most cases)
./run_calibration.sh 500 sceua 4

# MCMC (for uncertainty analysis)  
./run_calibration.sh 2000 mcmc 2

# DREAM (advanced MCMC variant)
./run_calibration.sh 1500 dream 2
```

### Custom Objective Functions
Edit the `objectivefunction` method in `calibration_enhanced.py` for:
- Multi-objective calibration
- Different performance metrics
- Station-specific weighting

### Validation
```bash
# Split-period validation (automatic)
# Configure in config/calibration_config.ini:
# validation_start = 2007-01-01
# validation_end = 2009-12-31

# Manual validation scripts
python scripts/validation/split_period_validation.py
```

## 🔬 Performance Optimization

### Speed Up Calibration
1. **Use more CPU cores**: `processes = 4` (or higher)
2. **Tighten parameter bounds**: Reduce search space
3. **Start with fewer iterations**: Test with 100-200 first
4. **Use SCE-UA algorithm**: Generally fastest

### Improve Results  
1. **Increase iterations**: 1000+ for final calibration
2. **Use high-quality data**: Clean observed streamflow data
3. **Appropriate parameter bounds**: Based on literature/experience
4. **Multi-objective calibration**: Consider multiple metrics

### Resource Management
```bash
# Monitor resource usage
free -h        # Memory usage
df -h          # Disk space
nproc          # Available CPU cores
```

## 📚 Files and Data Format Reference

### Forcing Files Format
```
# File: forcing_19.2871_82.7093
# Columns: RAIN(mm) TMAX(°C) TMIN(°C) WIND(m/s)
12.5    25.3    18.7    2.1
0.0     28.1    20.2    1.8
5.2     26.8    19.5    2.3
...
```

### Observed Data Format
Excel file with sheets/columns for each station:
- KANTAMAL, KESINGA, SALEBHATA, etc.
- Daily discharge values in cumecs

### Station File Format
```
# routing_model/station.txt
# Format: NUM_SOURCES STATION_NAME LAT LON AREA_KM2
1    KANTAMAL     20.0    85.0    1000
1    KESINGA      20.1    85.1    1200
...
```

## 🎯 Success Checklist

Before starting calibration, ensure:
- [ ] All forcing files present and properly formatted
- [ ] VIC model runs successfully and generates flux files
- [ ] Routing model executables have proper permissions
- [ ] Observed data file exists and contains all stations
- [ ] Configuration file paths are correct (use absolute paths)
- [ ] Sufficient disk space (>10GB recommended)
- [ ] System has adequate memory (>4GB recommended)

## 📞 Support and Contributing

### Getting Support
1. Check this README and troubleshooting section
2. Review log files in `output/calibration/logs/`
3. Run verification script with `--verbose` flag
4. Check the documentation in `docs/` folder

### Contributing
- Follow Python PEP 8 style guidelines
- Add comprehensive logging to new functions
- Include error handling and validation
- Update configuration files as needed
- Add unit tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Citation

If you use this calibration system in your research, please cite:
```
Enhanced VIC Multi-Station Calibration System for Mahanadi Basin
Version 1.0.0 (2024)
```

## 🙏 Acknowledgments

- VIC Development Team
- SPOTPY Development Team  
- Mahanadi Basin Data Providers

---

**Ready to start? Run the setup script and begin calibrating! 🚀**

```bash
bash quick_setup.sh
./run_calibration.sh
```