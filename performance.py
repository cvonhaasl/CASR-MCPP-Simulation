# performance.py
"""
Satellite Performance Analysis Module
==================================

This module calculates satellite coverage areas, signal strength, and redundancy
using spherical geometry and signal propagation models.

Mathematical Models
-----------------
1. Haversine Distance Formula:
   d = 2R * arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
   where:
   - d: Great-circle distance
   - R: Earth radius (6371 km)
   - φ: Latitude in radians
   - λ: Longitude in radians

2. Coverage Radius Adjustment:
   R_adjusted = R_base * √(max(EIRP, 1) / 50)  [for EIRP-based]
   R_adjusted = R_base * √(max(G/T + 30, 1) / 50)  [for G/T-based]
   where:
   - R_base: Base radius for frequency band
   - EIRP: Effective Isotropic Radiated Power (dBW)
   - G/T: Figure of Merit (dB/K)

3. Area Calculation:
   Area = N * (resolution)² * 12321
   where:
   - N: Number of covered grid points
   - resolution: Grid resolution in degrees
   - 12321: Approximate km² per square degree at equator

Base Coverage Radii (km)
----------------------
- C-Band:  1750.0
- Ku-Band: 1000.0
- Ka-Band: 350.0
- X-Band:  1375.0
- L-Band:  2000.0

Key Assumptions
-------------
1. Earth modeled as perfect sphere (R = 6371 km)
2. Coverage area approximated using grid-based approach
3. Signal strength decreases with square root of distance
4. Uniform Earth surface curvature for area calculations
5. Perfect atmospheric conditions
"""

from typing import Dict, List, Tuple, Optional, NamedTuple, Set
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Earth geometry constants
EARTH_RADIUS_KM: float = 6371.0
KM_PER_DEGREE: float = 111.32  # at equator
AREA_CONVERSION: float = 12321  # km² per square degree

class FrequencyBand(Enum):
    """Frequency band enumeration with standard names."""
    C = "C-Band"
    KU = "Ku-Band"
    KA = "Ka-Band"
    X = "X-Band"
    L = "L-Band"

@dataclass
class CoverageRadius:
    """Base coverage radii by frequency band (km)."""
    c_band: float = 1750.0
    ku_band: float = 1000.0
    ka_band: float = 350.0
    x_band: float = 1375.0
    l_band: float = 2000.0

class SignalParameters(NamedTuple):
    """Signal strength parameters."""
    eirp: Optional[float]  # Effective Isotropic Radiated Power (dBW)
    gt: Optional[float]    # Figure of Merit (dB/K)

class GridParameters(NamedTuple):
    """Global coverage grid parameters."""
    resolution: float      # Grid resolution in degrees
    lon_range: np.ndarray # Longitude points (-180 to 180)
    lat_range: np.ndarray # Latitude points (-90 to 90)

class SatellitePerformance:
    """Calculates satellite coverage and performance metrics."""
    
    def __init__(
        self,
        coverage_radius: CoverageRadius = CoverageRadius(),
        grid_resolution: float = 30.0
    ):
        """
        Initialize calculator with coverage parameters.
        
        Args:
            coverage_radius: Base coverage radii for each band
            grid_resolution: Grid cell size in degrees
        """
        self.coverage_radius = coverage_radius
        self.grid_params = self._initialize_grid(grid_resolution)
        
    @staticmethod
    def _initialize_grid(resolution: float) -> GridParameters:
        """Initialize global coverage grid."""
        return GridParameters(
            resolution=resolution,
            lon_range=np.arange(-180, 180 + resolution, resolution),
            lat_range=np.arange(-90, 90 + resolution, resolution)
        )

    def calculate_coverage_area(
        self,
        satellites: pd.DataFrame,
        beam_data: pd.DataFrame
    ) -> float:
        """
        Calculate total satellite coverage area.
        
        Args:
            satellites: Satellite configuration data
            beam_data: Beam pointing locations
            
        Returns:
            Total coverage area in square kilometers
        """
        try:
            if satellites.empty:
                return 0.0
                
            required_cols = {'Satellite Name', 'Band_List', 'Orbital Longitude'}
            if not required_cols.issubset(satellites.columns):
                logger.error(f"Missing columns: {required_cols - set(satellites.columns)}")
                return 0.0
            
            # Calculate coverage for each band
            sat_groups = satellites.groupby('Satellite Name')
            bands = self._get_unique_bands(sat_groups)
            coverage_stats = self._calculate_band_coverage(sat_groups, bands, beam_data)
            
            return sum(
                stats['Total Coverage Area (km²)']
                for stats in coverage_stats.values()
            )

        except Exception as e:
            logger.error(f"Coverage calculation error: {str(e)}")
            return 0.0

    def calculate_redundancy(
        self,
        satellites: pd.DataFrame,
        beam_data: pd.DataFrame
    ) -> float:
        """
        Calculate average coverage redundancy.
        
        Args:
            satellites: Satellite configuration data
            beam_data: Beam pointing locations
            
        Returns:
            Average number of overlapping coverages
        """
        try:
            sat_groups = satellites.groupby('Satellite Name')
            bands = self._get_unique_bands(sat_groups)
            coverage_stats = self._calculate_band_coverage(sat_groups, bands, beam_data)
            
            total_redundancy = sum(
                stats['Average Redundancy']
                for stats in coverage_stats.values()
            )
            return total_redundancy / len(coverage_stats) if coverage_stats else 0.0

        except Exception as e:
            logger.error(f"Redundancy calculation error: {str(e)}")
            raise

    @staticmethod
    def _get_unique_bands(sat_groups: pd.core.groupby.DataFrameGroupBy) -> Set[str]:
        """Extract unique frequency bands from the group data."""
        return {
            band
            for _, group in sat_groups
            for bands in group['Band_List']
            for band in bands
        }

    def _calculate_band_coverage(
        self,
        sat_groups: pd.core.groupby.DataFrameGroupBy,
        bands: Set[str],
        beam_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate coverage statistics for each frequency band."""
        # Initialize grid
        lon_grid, lat_grid = np.meshgrid(
            self.grid_params.lon_range,
            self.grid_params.lat_range
        )
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        
        # Calculate coverage for each band
        return {
            band: self._calculate_coverage_stats(
                self._calculate_redundancy_map(
                    sat_groups, band, beam_data,
                    lon_grid, lat_grid, lon_flat, lat_flat
                ),
                self.grid_params.resolution
            )
            for band in bands
        }

    def _calculate_redundancy_map(
        self,
        sat_groups: pd.core.groupby.DataFrameGroupBy,
        band: str,
        beam_data: pd.DataFrame,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        lon_flat: np.ndarray,
        lat_flat: np.ndarray
    ) -> np.ndarray:
        """Generate coverage redundancy map for a frequency band."""
        redundancy_map = np.zeros_like(lon_grid, dtype=int)
        
        for _, group in sat_groups:
            orbital_lon = self._orbital_to_lon(group['Orbital Longitude'].iloc[0])
            
            for _, transponder in group.iterrows():
                if band in transponder['Band_List']:
                    coverage_mask = self._calculate_coverage_mask(
                        transponder, band, beam_data,
                        orbital_lon, lon_flat, lat_flat,
                        lon_grid.shape
                    )
                    redundancy_map += coverage_mask
    
        return redundancy_map

    def _calculate_coverage_mask(
        self,
        transponder: pd.Series,
        band: str,
        beam_data: pd.DataFrame,
        orbital_lon: float,
        lon_flat: np.ndarray,
        lat_flat: np.ndarray,
        grid_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Calculate coverage mask for a single transponder."""
        # Get signal parameters and coverage radius
        signal_params = self._extract_signal_parameters(
            transponder['EIRP (dBW) / G/T (dB/K)']
        )
        radius = self._estimate_coverage_radius(band, signal_params)
        
        # Get coverage centers
        centers = self._get_coverage_centers(transponder, beam_data, orbital_lon)
        
        # Calculate coverage using vectorized operations
        coverage_mask = np.zeros(len(lon_flat), dtype=bool)
        for center_lon, center_lat in centers:
            distances = self._haversine_distance(
                center_lon, center_lat,
                lon_flat, lat_flat
            )
            coverage_mask |= distances <= radius
    
        return coverage_mask.reshape(grid_shape)

    @staticmethod
    def _haversine_distance(
        lon1: float,
        lat1: float,
        lon2: np.ndarray,
        lat2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate great-circle distances using vectorized Haversine formula.
        
        Args:
            lon1, lat1: Reference point coordinates
            lon2, lat2: Arrays of target coordinates
            
        Returns:
            Array of distances in kilometers
        """
        # Convert to radians
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)
        
        # Coordinate differences
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        a = np.clip(a, 0, 1)  # Handle numerical instabilities
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return EARTH_RADIUS_KM * c

    @staticmethod
    def _extract_signal_parameters(eirp_gt_str: str) -> SignalParameters:
        """Parse EIRP and G/T values from string representation."""
        if pd.isna(eirp_gt_str):
            return SignalParameters(None, None)
        
        parts = str(eirp_gt_str).split()
        eirp = gt = None
        
        for part in parts:
            try:
                value = float(part.replace('(EIRP)', '').replace('(G/T)', ''))
                if '(EIRP)' in part:
                    eirp = value
                elif '(G/T)' in part:
                    gt = value
            except ValueError:
                continue
                
        return SignalParameters(eirp, gt)

    def _estimate_coverage_radius(
        self,
        band: str,
        signal_params: SignalParameters
    ) -> float:
        """
        Estimate coverage radius based on signal parameters.
        
        Uses signal strength to adjust base coverage radius:
        - EIRP-based: R = R_base * √(max(EIRP, 1) / 50)
        - G/T-based:  R = R_base * √(max(G/T + 30, 1) / 50)
        """
        base_radius = getattr(
            self.coverage_radius,
            band.lower().replace('-', '_').replace('band', ''),
            self.coverage_radius.c_band  # Default to C-band if unknown
        )
        
        if signal_params.eirp is not None:
            factor = np.sqrt(max(signal_params.eirp, 1) / 50)
        elif signal_params.gt is not None:
            factor = np.sqrt(max(signal_params.gt + 30, 1) / 50)
        else:
            factor = 1.0
            
        return base_radius * factor

    @staticmethod
    def _orbital_to_lon(location: str) -> float:
        """Convert orbital location string to longitude in degrees."""
        if isinstance(location, str):
            location = location.strip()
            if location.endswith('E'):
                return float(location[:-1])
            elif location.endswith('W'):
                return -float(location[:-1])
        return float(location)

    def _get_coverage_centers(
        self,
        transponder: pd.Series,
        beam_data: pd.DataFrame,
        orbital_lon: float
    ) -> List[Tuple[float, float]]:
        """Determine coverage center points for a transponder."""
        centers = []
        
        # Check service areas
        service_areas = [
            str(transponder.get(f'Service Area {i}', '')).lower()
            for i in range(1, 7)
        ]
        
        # Handle global coverage
        if any('global' in area for area in service_areas):
            centers.append((orbital_lon, 0))
            return centers
        
        # Handle specific service areas
        for area in service_areas:
            if area and area != 'nan':
                beam_match = beam_data[
                    beam_data['Location'].str.lower() == area
                ]
                if not beam_match.empty:
                    loc = beam_match.iloc[0]
                    centers.append((loc['Longitude'], loc['Latitude']))
        
        # Default to orbital position if no centers found
        if not centers:
            centers.append((orbital_lon, 0))
            
        return centers

    @staticmethod
    def _calculate_coverage_stats(
        redundancy_map: np.ndarray,
        resolution: float
    ) -> Dict[str, float]:
        """Calculate coverage statistics from redundancy map."""
        covered_points = redundancy_map > 0
        covered_area_sq_deg = np.sum(covered_points) * (resolution ** 2)
        coverage_area_km2 = covered_area_sq_deg * AREA_CONVERSION
        
        max_redundancy = np.max(redundancy_map) if np.any(covered_points) else 0
        avg_redundancy = (
            np.mean(redundancy_map[covered_points])
            if np.any(covered_points) else 0
        )
        
        return {
            'Total Coverage Area (sq degrees)': covered_area_sq_deg,
            'Total Coverage Area (km²)': coverage_area_km2,
            'Max Redundancy': max_redundancy,
            'Average Redundancy': avg_redundancy,
            'Redundancy Map': redundancy_map
        }


def calculate_coverage_area(
    satellites: pd.DataFrame,
    beam_data: pd.DataFrame
) -> float:
    """Calculate total coverage area using default parameters."""
    return SatellitePerformance().calculate_coverage_area(satellites, beam_data)

def calculate_redundancy(
    satellites: pd.DataFrame,
    beam_data: pd.DataFrame
) -> float:
    """Calculate average coverage redundancy using default parameters."""
    return SatellitePerformance().calculate_redundancy(satellites, beam_data)