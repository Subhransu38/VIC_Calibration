from setuptools import setup, find_packages

setup(
    name="vic-calibration-enhanced",
    version="1.0.0",
    description="Enhanced VIC hydrological model calibration with multi-station support",
    author="VIC Calibration Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "spotpy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "rasterio>=1.2.0",
        "geopandas>=0.9.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.0",
        "PyYAML>=5.4.0",
        "configparser>=5.0.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)