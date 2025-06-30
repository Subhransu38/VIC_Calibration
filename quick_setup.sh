#!/bin/bash

# Quick Setup Script for VIC Calibration with Your Forcing Files
# This script sets up the enhanced VIC calibration system for your specific setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}    VIC Calibration Enhanced Setup Script${NC}"
echo -e "${BLUE}    Customized for Your Forcing File Structure${NC}"
echo -e "${BLUE}================================================================${NC}"
echo

# Function to create directory and report
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "${GREEN}Created:${NC} $1"
    else
        echo -e "${YELLOW}Exists:${NC} $1"
    fi
}

# Function to copy file if it doesn't exist
copy_file() {
    if [ ! -f "$2" ]; then
        cp "$1" "$2"
        echo -e "${GREEN}Copied:${NC} $1 → $2"
    else
        echo -e "${YELLOW}Exists:${NC} $2"
    fi
}

echo -e "${YELLOW}Step 1: Creating directory structure...${NC}"

# Create all necessary directories
create_dir "config"
create_dir "src/utilities"
create_dir "input/forcing"
create_dir "input/observed_data"
create_dir "input/gis"
create_dir "output/fluxes/calibrated"
create_dir "output/fluxes/validation"
create_dir "output/calibration/logs"
create_dir "output/calibration/results"
create_dir "output/calibration/plots"
create_dir "output/routing"
create_dir "routing_model"
create_dir "scripts/data_preparation"
create_dir "scripts/analysis"
create_dir "scripts/validation"
create_dir "verification_output"
create_dir "tests"
create_dir "docs"
create_dir "backup/configurations"
create_dir "backup/results"

echo
echo -e "${YELLOW}Step 2: Setting up configuration files...${NC}"

# Create the main configuration file
cat > config/calibration_config.ini << 'EOF'
[PATHS]
# Base directories
base_dir = .
input_dir = ../input
output_dir = ../output
calibration_output = ../output/calibration

# Executables
vic_executable = ../input/vicNl
routing_executable = ./routing_model/rout

# Data files
observed_data = ../input/observed_data/Mahanadi_discharge_14stations.xlsx
global_params = ../input/global.params
soil_params = ../input/soil.param

# GIS data
grid_raster = ../input/gis/Mahanadi_grid.tif
elevation_raster = ../input/gis/Mahanadi_basin_ElvAvg.tif
slope_raster = ../input/gis/Mahanadi_basin_SlopeAvg.tif
precip_raster = ../input/gis/Mahanadi_basin_Precip.tif
soil_raster = ../input/gis/Mahanadi_basin_SoilsAgg.tif

# Forcing and flux paths (updated for your file structure)
forcing_dir = ../input/forcing/
flux_output = ../output/fluxes/calibrated/
routing_output = ./output/

[SIMULATION]
start_date = 2005-01-01
end_date = 2009-12-31
calibration