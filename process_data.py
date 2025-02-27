# process_data.py
"""
Satellite Data Processing Module
==============================

This module handles the processing and transformation of satellite communication data,
focusing on bandwidth calculations and coordinate processing.

"""

from typing import List, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Constants for modulation efficiency (bits/symbol)
MODULATION_EFFICIENCY = {
    'BPSK': 1,
    'QPSK': 2,
    '8PSK': 3,
    '16QAM': 4,
    '32QAM': 5,
    '64QAM': 6,
    '128QAM': 7,
    '256QAM': 8
}

class SatelliteDataProcessor:
    """Class for processing satellite data with efficient methods."""
    
    def preprocess_satellite_data(self, sat_data: pd.DataFrame) -> pd.DataFrame:
        """Process satellite data with band handling and calculations."""
        try:
            # Create single copy for all modifications
            processed_data = sat_data.copy()
            
            # Basic type conversions
            processed_data['Modulation'] = processed_data['Modulation'].astype(str)
            processed_data['Band'] = processed_data['Band'].astype(str)
            
            # Process bands and expand multi-band entries
            processed_data = self._expand_multi_band_entries(processed_data)
            
            # Create Band_List after expansion
            processed_data['Band_List'] = processed_data.apply(
                lambda row: [row['Band']] if pd.notna(row['Band']) else [],
                axis=1
            )
            
            # Calculate data rates
            processed_data = self._calculate_data_rates(processed_data)
            
            # Process coordinates
            processed_data['Orbital Longitude'] = processed_data['Orbital Location'].apply(
                self._orbital_to_lon
            )
            
            # Ensure boolean type for Integrated
            processed_data['Integrated'] = processed_data.get('Integrated', False).astype(bool)
            
            return processed_data

        except Exception as e:
            logger.error(f"Error preprocessing satellite data: {str(e)}")
            logger.debug(f"Data sample: {sat_data.head()}")
            logger.debug(f"Columns: {sat_data.columns}")
            raise

    def _calculate_data_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate maximum data rates based on modulation and bandwidth."""
        # Map modulation efficiency
        modulation_values = data['Modulation'].map(MODULATION_EFFICIENCY).fillna(1)
        
        # Convert bandwidth and FEC to float
        bandwidth = data['Bandwidth Capacity (MHz)'].astype(float)
        fec = data['FEC'].fillna(1).astype(float)
        
        # Calculate max data rate
        data['Max Data Rate (Mbps)'] = bandwidth * modulation_values * fec
        
        return data

    def _expand_multi_band_entries(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle multi-band entries by splitting into separate rows."""
        # Count bands per entry
        data['band_count'] = data['Band'].str.count('[/,]') + 1
        
        # Separate single and multi-band entries
        single_band = data[data['band_count'] == 1].copy()
        multi_band = data[data['band_count'] > 1].copy()

        if multi_band.empty:
            single_band['Original Bandwidth'] = single_band['Bandwidth Capacity (MHz)']
            single_band['Band Count'] = 1
            return single_band

        # Process multi-band entries
        expanded_rows = []
        for _, row in multi_band.iterrows():
            bands = self._split_bands(row['Band'])
            bandwidth_per_band = row['Bandwidth Capacity (MHz)'] / len(bands)
            
            for band in bands:
                new_row = row.copy()
                new_row['Band'] = band
                new_row['Bandwidth Capacity (MHz)'] = bandwidth_per_band
                new_row['Original Bandwidth'] = row['Bandwidth Capacity (MHz)']
                new_row['Band Count'] = len(bands)
                expanded_rows.append(new_row)

        # Combine results
        expanded_df = pd.DataFrame(expanded_rows) if expanded_rows else pd.DataFrame()
        single_band['Original Bandwidth'] = single_band['Bandwidth Capacity (MHz)']
        single_band['Band Count'] = 1
        
        if expanded_df.empty:
            return single_band
        
        return pd.concat([expanded_df, single_band], ignore_index=True)

    @staticmethod
    def _orbital_to_lon(location: str) -> float:
        """Convert orbital location to longitude in degrees."""
        if isinstance(location, str):
            location = location.strip()
            if location.endswith('E'):
                return float(location[:-1])
            elif location.endswith('W'):
                return -float(location[:-1])
        return float(location)

    @staticmethod
    def _split_bands(band_str: str) -> List[str]:
        """Split band string into list of individual bands."""
        if pd.isna(band_str) or band_str == 'nan':
            return []
        
        # Handle multiple bands separated by either '/' or ','
        bands = str(band_str).replace('/', ',').split(',')
        return [band.strip() for band in bands if band.strip()]

def process_satellite_data(sat_data: pd.DataFrame) -> pd.DataFrame:
    """Process satellite data using SatelliteDataProcessor."""
    processor = SatelliteDataProcessor()
    return processor.preprocess_satellite_data(sat_data)

def process_beam_data(beam_data: pd.DataFrame) -> pd.DataFrame:
    """Process beam data."""
    processed = beam_data.copy()
    processed['Location'] = processed['Location'].astype(str)
    return processed