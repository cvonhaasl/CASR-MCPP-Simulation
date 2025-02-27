# option_1_milsatcom.py
"""
MILSATCOM Primary with COMSATCOM Reserve Strategy
==============================================

This module implements a reserve strategy where MILSATCOM serves as the primary
capacity source, with integrated COMSATCOM satellites available as backup only
when needed.

Strategy Rules
------------
1. MILSATCOM is always the primary capacity source
2. Only use integrated COMSATCOM when:
   - MILSATCOM capacity is insufficient
   - COMSATCOM satellites are already integrated
3. MILSATCOM must maintain primary role even when COMSATCOM is used

Selection Priority
----------------
1. All available MILSATCOM satellites
2. Integrated COMSATCOM satellites (only if needed)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from options.helpers import calculate_integration_cost_time

logger = logging.getLogger(__name__)

@dataclass
class ReserveConfig:
    """Configuration for reserve strategy."""
    milsatcom_availability_prob: float = 0.2
    comsatcom_availability_prob: float = 0.5

class MILSATCOMPrimaryStrategy:
    """Implements MILSATCOM-primary with COMSATCOM reserve strategy."""
    
    def __init__(self, config: ReserveConfig = ReserveConfig()):
        """Initialize with configuration parameters."""
        self.config = config

    def calculate_option(
        self,
        milsatcom_data: pd.DataFrame,
        comsatcom_data: pd.DataFrame,
        demand_mbps: float,
        urgency_level: str,
        percentage_use: float,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate reserve strategy configuration.
        
        Strategy:
        1. First attempt to meet demand with MILSATCOM
        2. If MILSATCOM insufficient, add integrated COMSATCOM
        3. Maintain MILSATCOM as primary source
        """
        try:
            # Process and prepare satellite data
            milsatcom_sats, comsatcom_sats = self._prepare_satellite_data(
                milsatcom_data.copy(),
                comsatcom_data.copy(),
                percentage_use
            )

            # First select MILSATCOM satellites
            selected_mil, mil_capacity = self._select_milsatcom_satellites(
                milsatcom_sats,
                self.config.milsatcom_availability_prob
            )

            # Determine if we need COMSATCOM reserve
            remaining_demand = max(0, demand_mbps - mil_capacity)
            selected_com = pd.DataFrame()
            com_capacity = 0.0
            
            if remaining_demand > 0:
                # Only select integrated COMSATCOM satellites as reserve
                selected_com, com_capacity = self._select_comsatcom_reserve(
                    comsatcom_sats,
                    remaining_demand,
                    self.config.comsatcom_availability_prob
                )

            # Combine results
            return self._create_result(
                selected_mil,
                selected_com,
                mil_capacity,
                com_capacity,
                urgency_level,
                base_config
            )

        except Exception as e:
            logger.error(f"Error in reserve strategy calculation: {str(e)}")
            raise

    def _prepare_satellite_data(
        self,
        milsatcom_data: pd.DataFrame,
        comsatcom_data: pd.DataFrame,
        percentage_use: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare satellite data with usable capacity calculations."""
        # Calculate usable capacity for both types
        for df in [milsatcom_data, comsatcom_data]:
            df['Usable Capacity (Mbps)'] = (
                df['Max Data Rate (Mbps)'] * (percentage_use / 100)
            )

        # Set satellite types
        milsatcom_data['Satellite Type'] = 'MILSATCOM'
        comsatcom_data['Satellite Type'] = 'COMSATCOM'
        
        # Ensure integrated status for COMSATCOM
        comsatcom_data['Integrated'] = comsatcom_data.get('Integrated', False)
        
        return milsatcom_data, comsatcom_data

    def _select_milsatcom_satellites(
        self,
        milsatcom_data: pd.DataFrame,
        availability_prob: float
    ) -> Tuple[pd.DataFrame, float]:
        """Select available MILSATCOM satellites."""
        if milsatcom_data.empty:
            return pd.DataFrame(), 0.0

        # Simulate availability
        available_sats = milsatcom_data['Satellite Name'].unique()[
            np.random.rand(len(milsatcom_data['Satellite Name'].unique())) < availability_prob
        ]
        
        selected = milsatcom_data[
            milsatcom_data['Satellite Name'].isin(available_sats)
        ].copy()
        
        total_capacity = selected['Usable Capacity (Mbps)'].sum()
        
        return selected, total_capacity

    def _select_comsatcom_reserve(
        self,
        comsatcom_data: pd.DataFrame,
        remaining_demand: float,
        availability_prob: float
    ) -> Tuple[pd.DataFrame, float]:
        """Select integrated COMSATCOM satellites as reserve capacity."""
        if comsatcom_data.empty:
            return pd.DataFrame(), 0.0

        # Filter for integrated satellites only
        integrated_sats = comsatcom_data[comsatcom_data['Integrated']].copy()
        if integrated_sats.empty:
            return pd.DataFrame(), 0.0

        # Simulate availability
        available_sats = integrated_sats['Satellite Name'].unique()[
            np.random.rand(len(integrated_sats['Satellite Name'].unique())) < availability_prob
        ]
        
        selected = integrated_sats[
            integrated_sats['Satellite Name'].isin(available_sats)
        ].copy()
        
        if selected.empty:
            return pd.DataFrame(), 0.0

        # Sort by capacity and select only what's needed
        selected = selected.sort_values('Usable Capacity (Mbps)', ascending=False)
        cumulative_capacity = selected['Usable Capacity (Mbps)'].cumsum()
        needed_sats = selected[cumulative_capacity <= remaining_demand]
        
        if needed_sats.empty:
            needed_sats = selected.iloc[[0]]  # Take at least one satellite if needed
            
        total_capacity = needed_sats['Usable Capacity (Mbps)'].sum()
        
        return needed_sats, total_capacity

    def _create_result(
        self,
        milsatcom_sats: pd.DataFrame,
        comsatcom_sats: pd.DataFrame,
        mil_capacity: float,
        com_capacity: float,
        urgency_level: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create final result dictionary."""
        # Combine satellites if any were selected
        all_satellites = pd.concat(
            [milsatcom_sats, comsatcom_sats],
            ignore_index=True
        ) if not (milsatcom_sats.empty and comsatcom_sats.empty) else pd.DataFrame()

        if all_satellites.empty:
            return {
                'Total Capacity (Mbps)': 0,
                'Time to Meet Demand (days)': float('inf'),
                'Integration Cost ($)': 0,
                'Satellite Data': pd.DataFrame(),
                'Satellites Used': 0,
                'Satellites Available': 0,
                'MILSATCOM Capacity (Mbps)': 0,
                'COMSATCOM Capacity (Mbps)': 0
            }

        # Calculate integration time based on satellite mix
        _, times = calculate_integration_cost_time(
            all_satellites,
            urgency_level,
            base_config,
            'Mixed'
        )

        return {
            'Total Capacity (Mbps)': mil_capacity + com_capacity,
            'Time to Meet Demand (days)': times['Integration Time'].mean(),
            'Integration Cost ($)': 0,  # No additional integration cost as we only use integrated COMSATCOM
            'Satellite Data': all_satellites,
            'Satellites Used': len(all_satellites['Satellite Name'].unique()),
            'Satellites Available': len(all_satellites['Satellite Name'].unique()),
            'MILSATCOM Capacity (Mbps)': mil_capacity,
            'COMSATCOM Capacity (Mbps)': com_capacity
        }
