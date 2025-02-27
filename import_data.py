# import_data.py
"""
Satellite Data Import and Validation Module
=========================================

This module handles the loading and validation of satellite communication system data
from structured Excel files. It supports both military (MILSATCOM) and commercial (COMSATCOM)
satellite data processing.


Key Data Structures
-----------------
1. Beam Data DataFrame:
   - Generalized Location: Categorical (e.g., 'Africa', 'APAC', 'CONUS')
   - Specific Location: String
   - Latitude: Float [-90.0 to 90.0 degrees]
   - Longitude: Float [-180.0 to 180.0 degrees]

2. Satellite Data DataFrame (COMSATCOM/MILSATCOM):
   - Satellite Name: String
   - Orbital Location: Float [degrees East/West]
   - Beam Type: Categorical ['Receive', 'Transmit']
   - Frequency Range: String [MHz]
   - EIRP/G/T: Float [dBW or dB/K]
   - Band: Categorical ['Ku-Band', 'C-Band', 'Ka-Band', 'L-Band', 'X-Band']
   - Bandwidth: Float [MHz]
   - Modulation: Categorical ['BPSK', 'QPSK', '8PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
   - FEC: Float [0.0 to 1.0]

Assumptions and Limitations
-------------------------
1. Excel file structure is fixed with predefined sheet names
2. Historical spending data
3. All satellite parameters are within standard industry ranges
4. Coordinate system uses standard GPS coordinates (WGS84)
"""

from typing import Tuple, Set, Dict, Any
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Constants
REQUIRED_SHEETS: Set[str] = {
    'Beam_Datav2',
    'COMSATCOM_Sat_Datav2',
    'MILSATCOM_Sat_Datav2',
    'COMSATCOM_Historical Spendingv2'
}

BEAM_DATA_DTYPES: Dict[str, Any] = {
    'Generalized Location': 'category',  # Memory optimization for categorical data
    'Specific Location': str,
    'Latitude': 'float32',  # Reduced precision float sufficient for coordinates
    'Longitude': 'float32'
}

SATELLITE_DATA_DTYPES = {
    'Satellite Name': str,
    'Orbital Location': 'float32',
    'Beam Name': str,
    'Type': 'category',
    'Band': 'category',
    'Bandwidth Capacity (MHz)': 'float32',
    'Modulation': 'category',
    'FEC': 'float32'
}

def load_data(excel_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the Excel file containing satellite information.
    
    Args:
        excel_file (str): Path to the Excel file
        
    Returns:
        Tuple containing beam_data, comsatcom_data, milsatcom_data, and historical_spending
    """
    try:
        xls = pd.ExcelFile(excel_file)
        required_sheets = {
            'Beam_Datav2', 
            'COMSATCOM_Sat_Datav2',
            'MILSATCOM_Sat_Datav2', 
            'COMSATCOM_Historical Spendingv2'
        }
        
        if not required_sheets.issubset(set(xls.sheet_names)):
            missing_sheets = required_sheets - set(xls.sheet_names)
            raise ValueError(f"Missing required sheets: {missing_sheets}")

        # Load beam data
        beam_data = pd.read_excel(xls, 'Beam_Datav2', dtype={
            'Specific Location': str,
            'Generalized Location': 'category',
            'Latitude': float,
            'Longitude': float
        })
        
        # Load COMSATCOM data with NA handling for Integrated column
        comsatcom_data = pd.read_excel(
            xls,
            'COMSATCOM_Sat_Datav2',
            dtype=SATELLITE_DATA_DTYPES
        )
        # Handle Integrated column separately with NA filling
        comsatcom_data['Integrated'] = comsatcom_data.get('Integrated', pd.Series()).fillna(False).astype(bool)
        
        # Load MILSATCOM data
        milsatcom_data = pd.read_excel(
            xls,
            'MILSATCOM_Sat_Datav2',
            dtype=SATELLITE_DATA_DTYPES
        )
        # Set Integrated to True for all MILSATCOM
        milsatcom_data['Integrated'] = True
        
        # Load historical spending
        historical_spending = pd.read_excel(
            xls, 
            'COMSATCOM_Historical Spendingv2',
            nrows=27
        )

        return beam_data, comsatcom_data, milsatcom_data, historical_spending

    except Exception as e:
        logger.error(f"Error loading data from {excel_file}: {str(e)}")
        raise

def validate_satellite_data(
    beam_data: pd.DataFrame,
    comsatcom_data: pd.DataFrame,
    milsatcom_data: pd.DataFrame,
    historical_spending: pd.DataFrame
) -> None:
    """
    Validate satellite data integrity with enhanced checks.

    Args:
        beam_data: DataFrame with beam location mappings
        comsatcom_data: DataFrame with commercial satellite data
        milsatcom_data: DataFrame with military satellite data
        historical_spending: DataFrame with historical spending data

    Raises:
        ValueError: If data validation fails
    """
    # Validate beam data
    _validate_beam_data(beam_data)
    
    # Validate satellite data
    for df, name in [(comsatcom_data, 'COMSATCOM'), (milsatcom_data, 'MILSATCOM')]:
        _validate_satellite_data(df, name)

    # Validate historical spending
    _validate_historical_spending(historical_spending)

def _validate_beam_data(beam_data: pd.DataFrame) -> None:
    """Helper function to validate beam data."""
    required_columns = {'Generalized Location', 'Specific Location', 'Latitude', 'Longitude'}
    if not required_columns.issubset(beam_data.columns):
        raise ValueError(f"Beam data missing required columns: {required_columns - set(beam_data.columns)}")
    
    # Validate coordinate ranges
    if not beam_data['Latitude'].between(-90, 90).all():
        raise ValueError("Invalid latitude values found (must be between -90 and 90)")
    if not beam_data['Longitude'].between(-180, 180).all():
        raise ValueError("Invalid longitude values found (must be between -180 and 180)")

def _validate_satellite_data(df: pd.DataFrame, name: str) -> None:
    """Helper function to validate satellite data."""
    required_columns = {
        'Satellite Name', 'Orbital Location', 'Beam Name',
        'Type', 'Frequency Range (MHz)', 'EIRP (dBW) / G/T (dB/K)',
        'Band', 'Bandwidth Capacity (MHz)', 'Modulation', 'FEC'
    }
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} data missing required columns: {required_columns - set(df.columns)}")
    
    # Validate FEC range
    if not df['FEC'].between(0, 1).all():
        raise ValueError(f"{name} contains invalid FEC values (must be between 0 and 1)")

def _validate_historical_spending(historical_spending: pd.DataFrame) -> None:
    """Helper function to validate historical spending data."""
    if historical_spending.shape[0] != 26:
        raise ValueError(f"Historical spending data should have 26 rows, found {historical_spending.shape[0]}")