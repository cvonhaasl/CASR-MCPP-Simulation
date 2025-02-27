# generate_scenarios.py
"""
Satellite Demand Scenario Generation Module
=========================================

This module generates demand scenarios for satellite capacity planning using
historical spending data and regression-based forecasting.

Mathematical Model
----------------
1. Bandwidth Prediction Equation:
   BW = β₀ + β₁⋅year + β₂⋅C_spend + β₃⋅Ka_spend + β₄⋅Ku_spend + β₅⋅L_spend + β₆⋅X_spend
   where:
   - BW: Predicted bandwidth in Mbps
   - β₀: Base intercept (-2,027,811.469)
   - β₁: Year coefficient (1,011.767887)
   - β₂-β₆: Band-specific spending coefficients

2. Total Bandwidth Need:
   Total_BW = Purchased_BW / Utilization_Factor

3. Regional Distribution:
   Regional_BW = Band_BW * Regional_Weight

Key Assumptions
-------------
- Linear relationship between spending and bandwidth demand
- Independent band utilization
- Regional demand follows preset distribution weights
- Volatility follows normal distribution
"""

from typing import List, Dict, Union, Any
import numpy as np
import pandas as pd
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Constants
SCENARIO_TYPES = {
    'BASELINE': "BASELINE",
    'HIGH_DEMAND': "HIGH_DEMAND",
    'MID_LEVEL': "MID_DEMAND",
    'SURGE': "SURGE"
}

# Regression coefficients (stored as dict for better performance)
COEFFICIENTS = {
    'intercept': -2027811.469,
    'year': 1011.767887,
    'c_band': 5.8535e-07,
    'ka_band': -5.23163e-05,
    'ku_band': -2.01259e-06,
    'l_band': 4.89206e-06,
    'x_band': 6.05904e-07
}

# Regional demand distribution weights
REGIONAL_WEIGHTS = {
    'africa': 0.1,
    'apac': 0.15,
    'conus': 0.2,
    'europe': 0.15,
    'latam': 0.1,
    'mena': 0.15,
    'oceania': 0.15
}

class DemandScenarioGenerator:
    """Generates demand scenarios for satellite capacity planning."""
    
    def __init__(self, utilization_factor: float = 0.4):
        """
        Initialize generator with utilization factor.
        
        Args:
            utilization_factor: Expected satellite utilization (0-1)
        """
        self.utilization_factor = utilization_factor
        self.spending_columns = [f'{band}-Band Spending' 
                               for band in ['C', 'Ka', 'Ku', 'L', 'X']]

    def generate_scenarios(
        self,
        historical_spending: pd.DataFrame,
        volatility_std: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate all demand scenarios from historical data.
        
        Args:
            historical_spending: Historical spending DataFrame
            volatility_std: Volatility standard deviation
            
        Returns:
            List of scenario dictionaries
        """
        try:
            years = historical_spending['Year']
            base_spending = historical_spending[self.spending_columns]
            
            # Generate all scenarios using vectorized operations
            scenarios = [
                self._generate_single_scenario(
                    scenario_name,
                    base_spending,
                    years,
                    volatility_std
                )
                for scenario_name in SCENARIO_TYPES.values()
            ]
            
            logger.info(f"Generated {len(scenarios)} demand scenarios")
            return scenarios

        except Exception as e:
            logger.error(f"Scenario generation error: {str(e)}")
            raise

    def _generate_single_scenario(
        self,
        scenario_name: str,
        base_spending: pd.DataFrame,
        years: pd.Series,
        volatility_std: float
    ) -> Dict[str, Any]:
        """Generate single demand scenario."""
        
        # Apply scenario adjustments using vectorized operations
        spending = self._apply_scenario_adjustments(
            scenario_name,
            base_spending,
            volatility_std
        )
        
        # Calculate bandwidth using vectorized operations
        purchased_bw = self._calculate_bandwidth(spending, years)
        total_bw = purchased_bw / self.utilization_factor
        
        # Calculate band-specific demand
        band_demand = self._calculate_band_demand(spending, total_bw)
        
        # Distribute demand across regions
        regional_demand = self._distribute_regional_demand(band_demand)
        
        return {
            'Name': scenario_name,
            'Year': years,
            'Spending': spending,
            'Purchased Bandwidth (Mbps)': purchased_bw,
            'Total Bandwidth Needed (Mbps)': total_bw,
            'Band Demand (Mbps)': band_demand,
            'Regional Demand (Mbps)': regional_demand
        }

    def _apply_scenario_adjustments(
        self,
        scenario_name: str,
        base_spending: pd.DataFrame,
        volatility_std: float
    ) -> pd.DataFrame:
        """Apply scenario-specific adjustments to spending."""
        if scenario_name == SCENARIO_TYPES['BASELINE']:
            return base_spending.copy()

        spending = base_spending.copy()
        shape = spending.shape

        # Use numpy's vectorized operations for adjustments
        if scenario_name == SCENARIO_TYPES['HIGH_DEMAND']:
            multiplier = np.random.uniform(1.5, 2.0, size=shape)
            volatility = np.random.normal(0.5, 0.9, size=shape)
            spending *= (1 + multiplier + volatility)
            
        elif scenario_name == SCENARIO_TYPES['MID_LEVEL']:
            multiplier = np.random.uniform(1.2, 1.7, size=shape)
            volatility = np.random.normal(0.1, 0.5, size=shape)
            spending *= (1 + multiplier + volatility)
            
        elif scenario_name == SCENARIO_TYPES['SURGE']:
            surge_indices = np.random.choice(
                spending.index,
                size=7,
                replace=False
            )
            surge_multipliers = np.random.uniform(2.5, 3.5, size=spending.shape[1])
            spending.iloc[surge_indices] *= surge_multipliers

        return spending

    def _calculate_bandwidth(
        self,
        spending: pd.DataFrame,
        years: pd.Series
    ) -> pd.Series:
        """Calculate bandwidth using regression equation."""
        return (COEFFICIENTS['intercept'] +
                COEFFICIENTS['year'] * years +
                sum(COEFFICIENTS[f"{band.lower()}_band"] * spending[f"{band}-Band Spending"]
                    for band in ['C', 'Ka', 'Ku', 'L', 'X']))

    def _calculate_band_demand(
        self,
        spending: pd.DataFrame,
        total_bandwidth: pd.Series
    ) -> pd.DataFrame:
        """Calculate per-band demand."""
        # Vectorized calculation
        total_spending = spending.sum(axis=1)
        band_demand = spending.div(total_spending, axis=0).multiply(
            total_bandwidth, axis=0
        )
        
        # Update column names
        band_demand.columns = [f"{col.split()[0]} Demand (Mbps)" 
                             for col in band_demand.columns]
        
        return band_demand

    def _distribute_regional_demand(
        self,
        band_demand: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Distribute demand across regions."""
        regions = list(REGIONAL_WEIGHTS.keys())
        weights = np.array(list(REGIONAL_WEIGHTS.values()))
        
        # Vectorized regional distribution
        return {
            band: pd.DataFrame(
                band_demand[band].values[:, None] * weights,
                columns=regions
            )
            for band in band_demand.columns
        }

def generate_demand_scenarios(
    historical_spending: pd.DataFrame,
    volatility_std: float = 0.8,
    utilization_factor: float = 0.3
) -> List[Dict[str, Any]]:
    """Convenience function for scenario generation."""
    generator = DemandScenarioGenerator(utilization_factor=utilization_factor)
    return generator.generate_scenarios(historical_spending, volatility_std)


