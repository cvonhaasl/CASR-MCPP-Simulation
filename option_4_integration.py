# option_4_integration.py
"""
Satellite Full Integration Strategy Module
=======================================

This module implements a full integration strategy for MILSATCOM and COMSATCOM
systems with a defined integration period.

Integration Timeline
------------------
- Integration Period: 2024-2026 (first 3 years)
- Pre-2024: Historical projection period
- Post-2026: All satellites fully integrated

Integration Cost Model
-------------------
Total Integration Cost = Base Cost * Number of Satellites * Urgency Multiplier
Yearly Cost = Total Cost / 3 years

Where:
- Base Cost: Calculated per satellite
- Urgency Multipliers:
  * Standard: 1.0x
  * Accelerated: 1.25x
  * Critical: 1.5x

Key Assumptions
-------------
1. Integration costs are evenly distributed over 3 years
2. All COMSATCOM satellites must be integrated by end of 2026
3. Different availability probabilities for MILSATCOM vs COMSATCOM
4. Integration status affects satellite performance and cost
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from .helpers import calculate_integration_cost_time
import logging

logger = logging.getLogger(__name__)

# Constants
INTEGRATION_START_YEAR = 2024
INTEGRATION_YEARS = 3
URGENCY_MULTIPLIERS = {
    'Standard': 1.0,
    'Accelerated': 1.25,
    'Critical': 1.5
}

def option_4_integration(
    milsatcom_data: pd.DataFrame,
    comsatcom_data: pd.DataFrame,
    demand_mbps: float,
    urgency_level: str,
    percentage_use: float,
    base_config: Dict[str, Any],
    year: int,
    initial_year: int
) -> Dict[str, Any]:
    """
    Implement full integration strategy with 3-year integration period.
    
    Timeline:
    - Pre-2024: Historical projection period
    - 2024-2026: Active integration period
    - Post-2026: All satellites integrated
    
    Args:
        milsatcom_data: MILSATCOM configurations
        comsatcom_data: COMSATCOM configurations
        demand_mbps: Bandwidth demand
        urgency_level: Integration urgency
        percentage_use: Capacity utilization
        base_config: Configuration parameters
        year: Current simulation year
        initial_year: Simulation start year
    """
    try:
        # Prepare satellite data
        milsatcom_data, comsatcom_data = _prepare_satellite_data(
            milsatcom_data.copy(),
            comsatcom_data.copy(),
            percentage_use
        )

        # Determine integration period status
        integration_status = _check_integration_period(year)
        
        # Calculate integration costs if in integration period
        integration_cost = 0
        if integration_status == 'active':
            integration_cost = _calculate_integration_cost(
                comsatcom_data,
                urgency_level,
                base_config
            )
            logger.debug(
                f"Integration cost for {year}: ${integration_cost:,.2f}"
            )

        # Update integration status based on period
        if integration_status == 'post':
            comsatcom_data['Integrated'] = True

        # Select available satellites
        milsatcom_sats, comsatcom_sats = select_available_satellites_with_probabilities(
            milsatcom_data,
            comsatcom_data,
            base_config
        )

        # Combine and process satellites
        return _process_selected_satellites(
            milsatcom_sats,
            comsatcom_sats,
            integration_status,
            urgency_level,
            base_config,
            integration_cost
        )

    except Exception as e:
        logger.error(f"Option 4 integration error: {str(e)}")
        raise

def _prepare_satellite_data(
    milsatcom_data: pd.DataFrame,
    comsatcom_data: pd.DataFrame,
    percentage_use: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare satellite data with usable capacity calculations."""
    # Calculate usable capacity
    for df in [milsatcom_data, comsatcom_data]:
        df['Usable Capacity (Mbps)'] = (
            df['Max Data Rate (Mbps)'] * (percentage_use / 100)
        )
    
    # Ensure integration status column
    comsatcom_data['Integrated'] = comsatcom_data.get('Integrated', False)
    
    return milsatcom_data, comsatcom_data

def _check_integration_period(year: int) -> str:
    """
    Determine integration period status.
    
    Returns:
        'pre': Before integration period
        'active': During integration period
        'post': After integration period
    """
    if year < INTEGRATION_START_YEAR:
        return 'pre'
    elif year < INTEGRATION_START_YEAR + INTEGRATION_YEARS:
        return 'active'
    else:
        return 'post'

def _calculate_integration_cost(
    comsatcom_data: pd.DataFrame,
    urgency_level: str,
    base_config: Dict[str, Any]
) -> float:
    """Calculate yearly integration cost during integration period."""
    non_integrated = comsatcom_data[~comsatcom_data['Integrated']]
    if non_integrated.empty:
        return 0.0

    # Calculate base cost using one satellite
    base_cost, _ = calculate_integration_cost_time(
        non_integrated.iloc[[0]],
        urgency_level,
        base_config,
        'Non-Integrated'
    )

    # Calculate total cost
    total_satellites = len(non_integrated['Satellite Name'].unique())
    total_cost = base_cost * total_satellites
    yearly_cost = total_cost / INTEGRATION_YEARS
    
    # Apply urgency multiplier
    return yearly_cost * URGENCY_MULTIPLIERS.get(urgency_level, 1.0)

def select_available_satellites_with_probabilities(
    milsatcom_data: pd.DataFrame,
    comsatcom_data: pd.DataFrame,
    base_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select available satellites based on availability probabilities."""
    # MILSATCOM selection
    selected_mil = _select_satellites(
        milsatcom_data,
        base_config.get('milsatcom_availability_prob', 0.2),
        'MILSATCOM'
    )

    # COMSATCOM selection
    selected_com = _select_satellites(
        comsatcom_data,
        base_config.get('comsatcom_availability_prob', 0.5),
        'COMSATCOM'
    )

    return selected_mil, selected_com

def _select_satellites(
    data: pd.DataFrame,
    probability: float,
    sat_type: str
) -> pd.DataFrame:
    """Select satellites based on availability probability."""
    if data.empty:
        return pd.DataFrame()
        
    satellites = data['Satellite Name'].unique()
    available = satellites[np.random.rand(len(satellites)) < probability]
    
    selected = data[data['Satellite Name'].isin(available)].copy()
    if not selected.empty:
        selected['Satellite Type'] = sat_type
        
    return selected

def _process_selected_satellites(
    milsatcom_sats: pd.DataFrame,
    comsatcom_sats: pd.DataFrame,
    integration_status: str,
    urgency_level: str,
    base_config: Dict[str, Any],
    integration_cost: float
) -> Dict[str, Any]:
    """Process selected satellites and calculate final metrics."""
    # Combine satellites
    all_satellites = pd.concat(
        [milsatcom_sats, comsatcom_sats],
        ignore_index=True
    ) if not (milsatcom_sats.empty and comsatcom_sats.empty) else pd.DataFrame()

    if all_satellites.empty:
        return create_empty_result()

    # Calculate integration time based on period
    integration_type = (
        'Mixed' if integration_status == 'active'
        else 'Integrated'
    )
    
    _, times = calculate_integration_cost_time(
        all_satellites,
        urgency_level,
        base_config,
        integration_type
    )
    
    time_to_meet = times['Integration Time'].mean()
    total_capacity = all_satellites['Usable Capacity (Mbps)'].sum()

    return {
        'Total Capacity (Mbps)': total_capacity,
        'Time to Meet Demand (days)': time_to_meet,
        'Integration Cost ($)': integration_cost,
        'Satellite Data': all_satellites,
        'Satellites Used': len(all_satellites['Satellite Name'].unique()),
        'Satellites Available': len(all_satellites['Satellite Name'].unique())
    }

def create_empty_result() -> Dict[str, Any]:
    """Create empty result when no satellites are available."""
    return {
        'Total Capacity (Mbps)': 0,
        'Time to Meet Demand (days)': float('inf'),
        'Integration Cost ($)': 0,
        'Satellite Data': pd.DataFrame(),
        'Satellites Used': 0,
        'Satellites Available': 0
    }