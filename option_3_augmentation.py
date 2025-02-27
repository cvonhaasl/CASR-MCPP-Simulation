# option_3_augmentation.py
"""
Satellite Augmentation Strategy Module
===================================

This module implements a hybrid satellite selection strategy combining MILSATCOM
and COMSATCOM systems to meet bandwidth demands.

Strategy Overview
---------------
- MILSATCOM: 30% of total demand
- COMSATCOM: 70% of total demand

Selection Priority
----------------
1. Available MILSATCOM satellites up to 30% capacity
2. Integrated COMSATCOM satellites
3. Non-integrated COMSATCOM satellites (if needed)

Key Parameters
------------
- Availability probability: Default 0.2 for MILSATCOM, 0.5 for COMSATCOM
- Integration status: Prioritizes already integrated satellites
- Capacity utilization: Adjustable percentage of max data rate
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from .helpers import calculate_integration_cost_time
import logging

logger = logging.getLogger(__name__)

def select_milsatcom_satellites(
    milsatcom_data: pd.DataFrame,
    required_capacity: float,
    base_config: Dict[str, Any]
) -> Tuple[float, pd.DataFrame]:
    """
    Select available MILSATCOM satellites to meet capacity requirement.
    
    Args:
        milsatcom_data: MILSATCOM satellite configurations
        required_capacity: Required bandwidth in Mbps (30% of total)
        base_config: Configuration parameters
        
    Returns:
        Tuple of (achieved capacity, selected satellites DataFrame)
    """
    if milsatcom_data.empty:
        return 0.0, pd.DataFrame()

    # Simulate satellite availability
    avail_prob = base_config.get('satellite_availability_prob', 0.2)
    available_sats = milsatcom_data['Satellite Name'].unique()[
        np.random.rand(len(milsatcom_data['Satellite Name'].unique())) < avail_prob
    ]
    
    selected = milsatcom_data[
        milsatcom_data['Satellite Name'].isin(available_sats)
    ].copy()
    
    if selected.empty:
        return 0.0, pd.DataFrame()

    selected['Satellite Type'] = 'MILSATCOM'
    total_capacity = selected['Usable Capacity (Mbps)'].sum()
    
    return total_capacity, selected

def select_comsatcom_satellites(
    comsatcom_data: pd.DataFrame,
    required_capacity: float,
    urgency_level: str,
    base_config: Dict[str, Any]
) -> Tuple[float, pd.DataFrame, float, float]:
    """
    Select COMSATCOM satellites with integration priority.
    
    Selection Strategy:
    1. Use integrated satellites first
    2. Add non-integrated satellites if needed
    3. Sort by capacity for optimal selection
    
    Args:
        comsatcom_data: COMSATCOM satellite configurations
        required_capacity: Required bandwidth in Mbps (70% of total)
        urgency_level: Integration urgency indicator
        base_config: Configuration parameters
        
    Returns:
        Tuple of (achieved capacity, selected satellites DataFrame,
                 integration time, integration cost)
    """
    if comsatcom_data.empty:
        return 0.0, pd.DataFrame(), 0.0, 0.0

    # Prepare data
    comsatcom_data = comsatcom_data.copy()
    comsatcom_data['Integrated'] = comsatcom_data.get('Integrated', False)

    # Simulate availability
    avail_prob = base_config.get('satellite_availability_prob', 0.5)
    available_sats = comsatcom_data['Satellite Name'].unique()[
        np.random.rand(len(comsatcom_data['Satellite Name'].unique())) < avail_prob
    ]
    
    selected = comsatcom_data[
        comsatcom_data['Satellite Name'].isin(available_sats)
    ].copy()

    if selected.empty:
        return 0.0, pd.DataFrame(), 0.0, 0.0

    # Process integrated satellites first
    integrated_sats = selected[selected['Integrated']].copy()
    integrated_capacity = integrated_sats['Usable Capacity (Mbps)'].sum()
    
    if integrated_capacity >= required_capacity:
        integrated_sats['Satellite Type'] = 'COMSATCOM'
        _, int_times = calculate_integration_cost_time(
            integrated_sats,
            urgency_level,
            base_config,
            'Integrated'
        )
        return required_capacity, integrated_sats, int_times['Integration Time'].mean(), 0.0

    # Process non-integrated satellites if needed
    non_integrated_sats = selected[~selected['Integrated']].copy()
    final_satellites = [integrated_sats] if not integrated_sats.empty else []
    total_capacity = integrated_capacity

    if not non_integrated_sats.empty:
        needed_capacity = required_capacity - integrated_capacity
        selected_non_integrated = _select_best_satellites(
            non_integrated_sats,
            needed_capacity
        )
        
        if not selected_non_integrated.empty:
            int_cost, int_times = calculate_integration_cost_time(
                selected_non_integrated,
                urgency_level,
                base_config,
                'Non-Integrated'
            )
            final_satellites.append(selected_non_integrated)
            total_capacity += selected_non_integrated['Usable Capacity (Mbps)'].sum()
            
            combined_sats = pd.concat(final_satellites, ignore_index=True)
            combined_sats['Satellite Type'] = 'COMSATCOM'
            
            return (
                total_capacity,
                combined_sats,
                int_times['Integration Time'].mean(),
                int_cost
            )
    
    # Handle integrated-only case
    if final_satellites:
        combined_sats = pd.concat(final_satellites, ignore_index=True)
        combined_sats['Satellite Type'] = 'COMSATCOM'
        _, int_times = calculate_integration_cost_time(
            combined_sats,
            urgency_level,
            base_config,
            'Integrated'
        )
        return (
            total_capacity,
            combined_sats,
            int_times['Integration Time'].mean(),
            0.0
        )
    
    return 0.0, pd.DataFrame(), 0.0, 0.0

def _select_best_satellites(
    satellites: pd.DataFrame,
    needed_capacity: float
) -> pd.DataFrame:
    """
    Select satellites optimally to meet capacity requirement.
    Uses greedy approach sorting by capacity.
    """
    selected = pd.DataFrame()
    current_capacity = 0.0
    
    for _, sat in satellites.sort_values('Usable Capacity (Mbps)', ascending=False).iterrows():
        if current_capacity >= needed_capacity:
            break
        selected = pd.concat([selected, pd.DataFrame([sat])])
        current_capacity += sat['Usable Capacity (Mbps)']
    
    return selected

def option_3_augmentation(
    milsatcom_data: pd.DataFrame,
    comsatcom_data: pd.DataFrame,
    demand_mbps: float,
    urgency_level: str,
    percentage_use: float,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Implement augmentation strategy with 30% MILSATCOM and 70% COMSATCOM.
    
    Strategy:
    1. Allocate 30% of demand to MILSATCOM
    2. Allocate 70% of demand to COMSATCOM
    3. Prioritize integrated satellites
    4. Calculate total capacity and integration requirements
    
    Args:
        milsatcom_data: MILSATCOM satellite configurations
        comsatcom_data: COMSATCOM satellite configurations
        demand_mbps: Total bandwidth demand in Mbps
        urgency_level: Integration urgency indicator
        percentage_use: Capacity utilization percentage
        base_config: Configuration parameters
        
    Returns:
        Dictionary containing:
        - Total Capacity (Mbps)
        - Time to Meet Demand (days)
        - Integration Cost ($)
        - Satellite Data
        - Satellites Used/Available counts
    """
    try:
        # Prepare satellite data with usable capacity
        milsatcom_data = milsatcom_data.copy()
        comsatcom_data = comsatcom_data.copy()
        
        usable_capacity = lambda df: df['Max Data Rate (Mbps)'] * (percentage_use / 100)
        milsatcom_data['Usable Capacity (Mbps)'] = usable_capacity(milsatcom_data)
        comsatcom_data['Usable Capacity (Mbps)'] = usable_capacity(comsatcom_data)

        # Calculate required capacities (30/70 split)
        milsatcom_required = demand_mbps * 0.30
        comsatcom_required = demand_mbps * 0.70

        # Select satellites
        mil_capacity, mil_satellites = select_milsatcom_satellites(
            milsatcom_data,
            milsatcom_required,
            base_config
        )

        com_capacity, com_satellites, com_time, com_cost = select_comsatcom_satellites(
            comsatcom_data,
            comsatcom_required,
            urgency_level,
            base_config
        )

        # Combine results
        all_satellites = pd.concat(
            [mil_satellites, com_satellites],
            ignore_index=True
        ) if not (mil_satellites.empty and com_satellites.empty) else pd.DataFrame()

        if all_satellites.empty:
            return {
                'Total Capacity (Mbps)': 0,
                'Time to Meet Demand (days)': float('inf'),
                'Integration Cost ($)': 0,
                'Satellite Data': pd.DataFrame(),
                'Satellites Used': 0,
                'Satellites Available': 0
            }

        # Calculate final integration time
        time_to_meet = (
            com_time if not com_satellites.empty and any(~com_satellites['Integrated'])
            else calculate_integration_cost_time(
                all_satellites, urgency_level, base_config, 'Integrated'
            )[1]['Integration Time'].mean()
        )

        return {
            'Total Capacity (Mbps)': mil_capacity + com_capacity,
            'Time to Meet Demand (days)': time_to_meet,
            'Integration Cost ($)': com_cost,
            'Satellite Data': all_satellites,
            'Satellites Used': len(all_satellites['Satellite Name'].unique()),
            'Satellites Available': len(all_satellites['Satellite Name'].unique())
        }

    except Exception as e:
        logger.error(f"Option 3 augmentation error: {str(e)}")
        raise