#!/usr/bin/env python3
"""
Forcing Files Verification and Setup Script
This script helps verify your forcing files and sets up the VIC calibration system
"""

import os
import glob
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
import argparse

def setup_logging():
    """Setup logging for the verification script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_forcing_files(forcing_dir):
    """Analyze forcing files in the directory"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing forcing files in: {forcing_dir}")
    
    # Find all forcing files
    forcing_pattern = os.path.join(forcing_dir, "forcing_*")
    forcing_files = glob.glob(forcing_pattern)
    
    if not forcing_files:
        logger.error(f"No forcing files found with pattern: {forcing_pattern}")
        return None
    
    logger.info(f"Found {len(forcing_files)} forcing files")
    
    # Analyze file naming pattern
    coordinates = []
    file_info = {}
    
    for file_path in forcing_files:
        filename = os.path.basename(file_path)
        logger.debug(f"Analyzing file: {filename}")
        
        # Extract coordinates from filename
        # Pattern: forcing_19.2871_82.7093
        coord_match = re.search(r'forcing_(\d+\.\d+)_(\d+\.\d+)', filename)
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            coordinates.append((lat, lon))
            
            # Analyze file content
            try:
                # Read first few lines to understand format
                with open(file_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                
                # Try to read as data
                sample_data = pd.read_csv(file_path, sep=r'\s+', nrows=10, header=None)
                
                file_info[filename] = {
                    'coordinates': (lat, lon),
                    'file_size': os.path.getsize(file_path),
                    'num_columns': len(sample_data.columns),
                    'sample_data': sample_data.head(3),
                    'first_lines': first_lines
                }
                
            except Exception as e:
                logger.warning(f"Error reading {filename}: {e}")
                file_info[filename] = {
                    'coordinates': (lat, lon),
                    'file_size': os.path.getsize(file_path),
                    'error': str(e)
                }
    
    # Summary statistics
    logger.info("=" * 50)
    logger.info("FORCING FILES ANALYSIS SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"Total forcing files: {len(forcing_files)}")
    
    if coordinates:
        lats, lons = zip(*coordinates)
        logger.info(f"Latitude range: {min(lats):.4f} to {max(lats):.4f}")
        logger.info(f"Longitude range: {min(lons):.4f} to {max(lons):.4f}")
    
    # Check file format consistency
    column_counts = []
    file_sizes = []
    
    for filename, info in file_info.items():
        if 'error' not in info:
            column_counts.append(info['num_columns'])
            file_sizes.append(info['file_size'])
            
            logger.info(f"\n{filename}:")
            logger.info(f"  Coordinates: {info['coordinates']}")
            logger.info(f"  Columns: {info['num_columns']}")
            logger.info(f"  File size: {info['file_size']} bytes")
            
            if 'sample_data' in info:
                logger.info("  Sample data (first 3 rows):")
                logger.info(f"    {info['sample_data'].to_string(index=False)}")
    
    # Check consistency
    if column_counts:
        unique_columns = set(column_counts)
        if len(unique_columns) == 1:
            logger.info(f"\n✓ All files have consistent {list(unique_columns)[0]} columns")
        else:
            logger.warning(f"\n⚠ Inconsistent column counts: {unique_columns}")
    
    # File size analysis
    if file_sizes:
        avg_size = np.mean(file_sizes)
        std_size = np.std(file_sizes)
        logger.info(f"\nFile size statistics:")
        logger.info(f"  Average: {avg_size:.0f} bytes")
        logger.info(f"  Std dev: {std_size:.0f} bytes")
        logger.info(f"  Range: {min(file_sizes)} - {max(file_sizes)} bytes")
        
        # Check for suspiciously small/large files
        threshold = 2 * std_size
        for filename, info in file_info.items():
            if 'file_size' in info:
                size_diff = abs(info['file_size'] - avg_size)
                if size_diff > threshold:
                    logger.warning(f"  ⚠ {filename}: Unusual size ({info['file_size']} bytes)")
    
    return file_info

def verify_expected_flux_files(flux_dir, forcing_info):
    """Verify that expected flux files exist based on forcing files"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\nChecking for corresponding flux files in: {flux_dir}")
    
    if not os.path.exists(flux_dir):
        logger.warning(f"Flux directory does not exist: {flux_dir}")
        return False
    
    # Check for expected flux files
    expected_flux_files = []
    found_flux_files = []
    
    for forcing_filename in forcing_info.keys():
        # Convert forcing filename to expected flux filename
        # forcing_19.2871_82.7093 -> fluxes_19.2871_82.7093
        expected_flux = forcing_filename.replace('forcing_', 'fluxes_')
        expected_flux_path = os.path.join(flux_dir, expected_flux)
        
        expected_flux_files.append(expected_flux)
        
        if os.path.exists(expected_flux_path):
            found_flux_files.append(expected_flux)
            logger.debug(f"✓ Found: {expected_flux}")
        else:
            logger.debug(f"✗ Missing: {expected_flux}")
    
    logger.info(f"Expected flux files: {len(expected_flux_files)}")
    logger.info(f"Found flux files: {len(found_flux_files)}")
    
    if len(found_flux_files) == 0:
        logger.warning("No flux files found. Run VIC model first to generate flux files.")
    elif len(found_flux_files) < len(expected_flux_files):
        logger.warning(f"Missing {len(expected_flux_files) - len(found_flux_files)} flux files")
        missing_files = set(expected_flux_files) - set(found_flux_files)
        for missing in list(missing_files)[:5]:  # Show first 5 missing files
            logger.warning(f"  Missing: {missing}")
        if len(missing_files) > 5:
            logger.warning(f"  ... and {len(missing_files) - 5} more")
    else:
        logger.info("✓ All expected flux files found!")
    
    return len(found_flux_files) == len(expected_flux_files)

def create_sample_global_params(output_file, forcing_dir, flux_dir):
    """Create a sample VIC global parameters file"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating sample global parameters file: {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sample_content = f"""#######################################################################
# VIC Model Global Parameters File
# Generated by forcing verification script
#######################################################################

#######################################################################
# SIMULATION CONTROL
#######################################################################
NLAYER          3      # number of soil layers
NODES           3      # number of soil thermal nodes
TIME_STEP       24     # model time step in hours (24 = daily)
SNOW_STEP       3      # time step for snow model
STARTDATE       2005 01 01  # start date of simulation
ENDDATE         2009 12 31  # end date of simulation
CALENDAR        PROLEPTIC_GREGORIAN

#######################################################################
# ENERGY BALANCE
#######################################################################
FULL_ENERGY     FALSE  # calculate full energy balance
FROZEN_SOIL     FALSE  # calculate frozen soil

#######################################################################
# INPUT/OUTPUT FILE LOCATIONS
#######################################################################
FORCING1        {os.path.abspath(forcing_dir)}/forcing_  # forcing file prefix
FORCE_FORMAT    ASCII       # forcing file format
FORCE_ENDIAN    LITTLE      # forcing file endianness
N_TYPES         4           # number of forcing types
FORCE_TYPE      PREC        # precipitation
FORCE_TYPE      TMAX        # maximum temperature
FORCE_TYPE      TMIN        # minimum temperature  
FORCE_TYPE      WIND        # wind speed

SOIL            {os.path.abspath('../input/soil.param')}
RESULT_DIR      {os.path.abspath(flux_dir)}

#######################################################################
# OUTPUT OPTIONS
#######################################################################
COMPRESS        FALSE
OUT_STEP        24         # output time step (hours)
SKIPYEAR        0          # number of years to skip
STARTMONTH      1          # start month for output
STARTDAY        1          # start day for output

# Output variables
OUTVAR          OUT_RUNOFF      # surface runoff
OUTVAR          OUT_BASEFLOW    # baseflow
OUTVAR          OUT_EVAP        # evapotranspiration
OUTVAR          OUT_SWE         # snow water equivalent
OUTVAR          OUT_SOIL_MOIST  # soil moisture

#######################################################################
# METEOROLOGICAL FORCING DISAGGREGATION
#######################################################################
PLAPSE          TRUE       # use precipitation lapse
TDIFF           FALSE      # calculate temperature difference
"""

    with open(output_file, 'w') as f:
        f.write(sample_content)
    
    logger.info(f"Sample global parameters file created: {output_file}")
    logger.info("Please review and modify the file according to your specific setup.")

def create_forcing_summary_report(forcing_info, output_file):
    """Create a detailed summary report of forcing files"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating forcing summary report: {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Forcing Files Analysis Report\n\n")
        f.write(f"**Total Files:** {len(forcing_info)}\n\n")
        
        f.write("## File Information\n\n")
        f.write("| Filename | Coordinates (Lat, Lon) | Columns | Size (bytes) | Status |\n")
        f.write("|----------|------------------------|---------|--------------|--------|\n")
        
        for filename, info in sorted(forcing_info.items()):
            if 'error' in info:
                status = f"ERROR: {info['error']}"
                columns = "N/A"
            else:
                status = "OK"
                columns = info.get('num_columns', 'N/A')
            
            coords = info.get('coordinates', ('N/A', 'N/A'))
            size = info.get('file_size', 'N/A')
            
            f.write(f"| {filename} | ({coords[0]}, {coords[1]}) | {columns} | {size} | {status} |\n")
        
        f.write("\n## Expected VIC Output Files\n\n")
        f.write("Based on your forcing files, VIC should generate the following flux files:\n\n")
        
        for filename in sorted(forcing_info.keys()):
            flux_filename = filename.replace('forcing_', 'fluxes_')
            f.write(f"- `{flux_filename}`\n")
        
        f.write("\n## Routing Model Configuration\n\n")
        f.write("For the routing model, ensure your `routing_input.txt` includes:\n\n")
        f.write("```\n")
        f.write("IN_RUNOFF               /path/to/flux/directory/fluxes_\n")
        f.write("```\n\n")
        f.write("The routing model will automatically find files matching the pattern `fluxes_*`.\n\n")
        
        f.write("## Calibration Setup Checklist\n\n")
        f.write("- [ ] Verify all forcing files have 4 columns (rain, tmax, tmin, wind)\n")
        f.write("- [ ] Check that forcing files cover the calibration period\n")
        f.write("- [ ] Ensure VIC global parameters file points to correct forcing directory\n")
        f.write("- [ ] Run VIC model to generate flux files\n")
        f.write("- [ ] Verify flux files are created in the expected location\n")
        f.write("- [ ] Update calibration config file with correct paths\n")
        f.write("- [ ] Prepare station files for routing model\n")
        f.write("- [ ] Test routing model with generated flux files\n")
    
    logger.info(f"Forcing summary report created: {output_file}")

def check_calibration_prerequisites(config_file="config/calibration_config.ini"):
    """Check if all prerequisites for calibration are met"""
    logger = logging.getLogger(__name__)
    
    logger.info("Checking calibration prerequisites...")
    
    prerequisites = {
        'config_file': config_file,
        'vic_executable': None,
        'routing_executable': None,
        'forcing_files': None,
        'flux_files': None,
        'observed_data': None,
        'station_files': None
    }
    
    # Check config file
    if os.path.exists(config_file):
        logger.info("✓ Configuration file found")
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Extract paths from config
            vic_exec = config.get('PATHS', 'vic_executable', fallback=None)
            routing_exec = config.get('PATHS', 'routing_executable', fallback=None)
            forcing_dir = config.get('PATHS', 'forcing_dir', fallback=None)
            flux_dir = config.get('PATHS', 'flux_output', fallback=None)
            obs_data = config.get('PATHS', 'observed_data', fallback=None)
            station_file = config.get('STATIONS', 'station_file', fallback=None)
            
            prerequisites.update({
                'vic_executable': vic_exec,
                'routing_executable': routing_exec,
                'forcing_dir': forcing_dir,
                'flux_dir': flux_dir,
                'observed_data': obs_data,
                'station_file': station_file
            })
            
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
    else:
        logger.error("✗ Configuration file not found")
    
    # Check each prerequisite
    all_good = True
    
    for item, path in prerequisites.items():
        if path and os.path.exists(path):
            logger.info(f"✓ {item}: {path}")
        else:
            logger.error(f"✗ {item}: {path} (not found)")
            all_good = False
    
    return all_good

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verify forcing files and setup VIC calibration')
    parser.add_argument('--forcing-dir', '-f', default='../input/forcing/', 
                       help='Directory containing forcing files')
    parser.add_argument('--flux-dir', '-x', default='../output/fluxes/calibrated/', 
                       help='Directory for VIC flux outputs')
    parser.add_argument('--output-dir', '-o', default='./verification_output/', 
                       help='Directory for verification outputs')
    parser.add_argument('--config', '-c', default='config/calibration_config.ini',
                       help='Calibration configuration file')
    parser.add_argument('--create-global-params', action='store_true',
                       help='Create sample VIC global parameters file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("VIC CALIBRATION FORCING FILES VERIFICATION")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Analyze forcing files
    forcing_info = analyze_forcing_files(args.forcing_dir)
    
    if not forcing_info:
        logger.error("No forcing files found. Cannot proceed.")
        return 1
    
    # Step 2: Check for flux files
    verify_expected_flux_files(args.flux_dir, forcing_info)
    
    # Step 3: Create summary report
    report_file = os.path.join(args.output_dir, 'forcing_analysis_report.md')
    create_forcing_summary_report(forcing_info, report_file)
    
    # Step 4: Create sample global parameters file if requested
    if args.create_global_params:
        global_params_file = os.path.join(args.output_dir, 'sample_global.params')
        create_sample_global_params(global_params_file, args.forcing_dir, args.flux_dir)
    
    # Step 5: Check calibration prerequisites
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION PREREQUISITES CHECK")
    logger.info("=" * 60)
    
    all_ready = check_calibration_prerequisites(args.config)
    
    # Step 6: Provide recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    if all_ready:
        logger.info("✓ All prerequisites met! You can proceed with calibration.")
        logger.info("\nNext steps:")
        logger.info("1. Review the configuration file")
        logger.info("2. Run a test VIC simulation")
        logger.info("3. Start calibration with: ./run_calibration.sh")
    else:
        logger.warning("⚠ Some prerequisites are missing. Please address the issues above.")
        logger.info("\nRecommended actions:")
        logger.info("1. Ensure all forcing files are properly formatted")
        logger.info("2. Run VIC model to generate flux files")
        logger.info("3. Prepare observed streamflow data")
        logger.info("4. Set up routing model station files")
        logger.info("5. Update configuration file paths")
    
    # Summary
    logger.info(f"\nVerification complete. Check detailed report: {report_file}")
    
    return 0 if all_ready else 1

if __name__ == "__main__":
    exit(main())