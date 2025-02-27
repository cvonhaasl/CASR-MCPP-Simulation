# option_2_comsatcom.py
"""
COMSATCOM Primary with MILSATCOM Reserve Strategy
=============================================

This module implements a reserve strategy where COMSATCOM serves as the primary
capacity source, with MILSATCOM satellites available as backup only when needed.

Strategy Rules
------------
1. COMSATCOM is always the primary capacity source
2. Prioritize integrated COMSATCOM first
3. Use non-integrated COMSATCOM next
4. Only use MILSATCOM as reserve when:
   - COMSATCOM capacity is insufficient
   - All available COMSATCOM has been utilized

Selection Priority
----------------
1. Integrated COMSATCOM satellites
2. Non-integrated COMSATCOM satellites
3. MILSATCOM satellites (only if needed)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from .helpers import calculate_integration_cost_time

logger = logging.getLogger(__name__)

@dataclass
class ReserveConfig:
    """Configuration for reserve strategy."""
    comsatcom_availability_prob: float = 0.5
    milsatcom_availability_prob: float = 0.2

class COMSATCOMPrimaryStrategy:
    """Implements COMSATCOM-primary with MILSATCOM reserve strategy."""
    
    def __init__(self, config: ReserveConfig = ReserveConfig()):
        """Initialize with configuration parameters."""
        self.config = config

    def calculate_option(
        self,
        comsatcom_data: pd.DataFrame,
        milsatcom_data: pd.DataFrame,
        demand_mbps: float,
        urgency_level: str,
        percentage_use: float,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate reserve strategy configuration.
        
        Strategy:
        1. First attempt to meet demand with COMSATCOM (integrated, then non-integrated)
        2. If COMSATCOM insufficient, add MILSATCOM as reserve
        3. Maintain COMSATCOM as primary source
        """
        try:
            # Process satellite data
            comsatcom_sats, milsatcom_sats = self._prepare_satellite_data(
                comsatcom_data.copy(),
                milsatcom_data.copy(),
                percentage_use
            )

            # Select COMSATCOM satellites first (both integrated and non-integrated)
            selected_com, com_capacity, integration_cost = self._select_comsatcom_satellites(
                comsatcom_sats,
                demand_mbps,
                urgency_level,
                base_config
            )

            # Determine if we need MILSATCOM reserve
            remaining_demand = max(0, demand_mbps - com_capacity)
            selected_mil = pd.DataFrame()
            mil_capacity = 0.0
            
            if remaining_demand > 0:
                selected_mil, mil_capacity = self._select_milsatcom_reserve(
                    milsatcom_sats,
                    remaining_demand,
                    self.config.milsatcom_availability_prob
                )

            # Combine results
            return self._create_result(
                selected_com,
                selected_mil,
                com_capacity,
                mil_capacity,
                integration_cost,
                urgency_level,
                base_config
            )

        except Exception as e:
            logger.error(f"Error in reserve strategy calculation: {str(e)}")
            raise

    def _prepare_satellite_data(
        self,
        comsatcom_data: pd.DataFrame,
        milsatcom_data: pd.DataFrame,
        percentage_use: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare satellite data with usable capacity calculations."""
        # Calculate usable capacity for both types
        for df in [comsatcom_data, milsatcom_data]:
            df['Usable Capacity (Mbps)'] = (
                df['Max Data Rate (Mbps)'] * (percentage_use / 100)
            )

        # Set satellite types
        comsatcom_data['Satellite Type'] = 'COMSATCOM'
        milsatcom_data['Satellite Type'] = 'MILSATCOM'
        
        # Ensure integrated status for COMSATCOM
        comsatcom_data['Integrated'] = comsatcom_data.get('Integrated', False)
        
        return comsatcom_data, milsatcom_data

    def _select_comsatcom_satellites(
        self,
        comsatcom_data: pd.DataFrame,
        demand_mbps: float,
        urgency_level: str,
        base_config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, float, float]:
        """Select COMSATCOM satellites, prioritizing integrated ones."""
        if comsatcom_data.empty:
            return pd.DataFrame(), 0.0, 0.0

        # Simulate availability
        available_sats = comsatcom_data['Satellite Name'].unique()[
            np.random.rand(len(comsatcom_data['Satellite Name'].unique())) < 
            self.config.comsatcom_availability_prob
        ]
        
        available = comsatcom_data[
            comsatcom_data['Satellite Name'].isin(available_sats)
        ].copy()

        if available.empty:
            return pd.DataFrame(), 0.0, 0.0

        # First try integrated satellites
        integrated = available[available['Integrated']].copy()
        integrated_capacity = integrated['Usable Capacity (Mbps)'].sum()

        # If integrated satellites are sufficient
        if integrated_capacity >= demand_mbps:
            _, times = calculate_integration_cost_time(
                integrated,
                urgency_level,
                base_config,
                'Integrated'
            )
            return integrated, integrated_capacity, 0.0

        # Need non-integrated satellites
        non_integrated = available[~available['Integrated']].copy()
        selected = integrated if not integrated.empty else pd.DataFrame()
        total_capacity = integrated_capacity

        if not non_integrated.empty:
            # Select non-integrated satellites by capacity
            needed = demand_mbps - integrated_capacity
            selected_non_int = self._select_by_capacity(non_integrated, needed)
            
            if not selected_non_int.empty:
                int_cost, _ = calculate_integration_cost_time(
                    selected_non_int,
                    urgency_level,
                    base_config,
                    'Non-Integrated'
                )
                selected = pd.concat([selected, selected_non_int], ignore_index=True)
                total_capacity += selected_non_int['Usable Capacity (Mbps)'].sum()
                return selected, total_capacity, int_cost

        return selected, total_capacity, 0.0

    def _select_milsatcom_reserve(
        self,
        milsatcom_data: pd.DataFrame,
        remaining_demand: float,
        availability_prob: float
    ) -> Tuple[pd.DataFrame, float]:
        """Select MILSATCOM satellites as reserve capacity."""
        if milsatcom_data.empty:
            return pd.DataFrame(), 0.0

        # Simulate availability
        available_sats = milsatcom_data['Satellite Name'].unique()[
            np.random.rand(len(milsatcom_data['Satellite Name'].unique())) < 
            availability_prob
        ]
        
        selected = milsatcom_data[
            milsatcom_data['Satellite Name'].isin(available_sats)
        ].copy()

        if selected.empty:
            return pd.DataFrame(), 0.0

        # Select only what's needed
        selected = self._select_by_capacity(selected, remaining_demand)
        total_capacity = selected['Usable Capacity (Mbps)'].sum()
        
        return selected, total_capacity

    @staticmethod
    def _select_by_capacity(
        satellites: pd.DataFrame,
        needed_capacity: float
    ) -> pd.DataFrame:
        """Select satellites optimally to meet capacity requirement."""
        selected = pd.DataFrame()
        current_capacity = 0.0
        
        for _, sat in satellites.sort_values('Usable Capacity (Mbps)', ascending=False).iterrows():
            if current_capacity >= needed_capacity:
                break
            selected = pd.concat([selected, pd.DataFrame([sat])])
            current_capacity += sat['Usable Capacity (Mbps)']
        
        return selected

    def _create_result(
        self,
        comsatcom_sats: pd.DataFrame,
        milsatcom_sats: pd.DataFrame,
        com_capacity: float,
        mil_capacity: float,
        integration_cost: float,
        urgency_level: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create final result dictionary."""
        all_satellites = pd.concat(
            [comsatcom_sats, milsatcom_sats],
            ignore_index=True
        ) if not (comsatcom_sats.empty and milsatcom_sats.empty) else pd.DataFrame()

        if all_satellites.empty:
            return {
                'Total Capacity (Mbps)': 0,
                'Time to Meet Demand (days)': float('inf'),
                'Integration Cost ($)': 0,
                'Satellite Data': pd.DataFrame(),
                'Satellites Used': 0,
                'Satellites Available': 0,
                'COMSATCOM Capacity (Mbps)': 0,
                'MILSATCOM Capacity (Mbps)': 0
            }

        # Calculate integration time based on satellite mix
        _, times = calculate_integration_cost_time(
            all_satellites,
            urgency_level,
            base_config,
            'Mixed'
        )

        return {
            'Total Capacity (Mbps)': com_capacity + mil_capacity,
            'Time to Meet Demand (days)': times['Integration Time'].mean(),
            'Integration Cost ($)': integration_cost,
            'Satellite Data': all_satellites,
            'Satellites Used': len(all_satellites['Satellite Name'].unique()),
            'Satellites Available': len(all_satellites['Satellite Name'].unique()),
            'COMSATCOM Capacity (Mbps)': com_capacity,
            'MILSATCOM Capacity (Mbps)': mil_capacity
        }
