# options/helpers.py

from typing import Tuple, Dict, Union, Optional
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging, random


"""
Module providing helper functions for satellite integration calculations.

This module contains utilities for calculating integration costs and times
for different satellite types and urgency levels.
"""



logger = logging.getLogger(__name__)

class UrgencyLevel(Enum):
    """Enumeration of integration urgency levels."""
    STANDARD = "Standard"
    ACCELERATED = "Accelerated"
    CRITICAL = "Critical"

class SatelliteType(Enum):
    """Enumeration of satellite types."""
    MILSATCOM = "MILSATCOM"
    INTEGRATED = "Integrated"
    NON_INTEGRATED = "Non-Integrated"
    MIXED = "Mixed"

@dataclass
class IntegrationTimes:
    """Integration times in days for different urgency levels."""
    standard: float
    accelerated: float
    critical: float

@dataclass
class IntegrationCosts:
    """Base integration costs for different components."""
    hardware: float = 250_000_000
    license: float = 90_680.10

class IntegrationCalculator:
    """
    Handles calculations for satellite integration costs and times.
    """

    def __init__(
        self,
        integration_costs: Optional[IntegrationCosts] = None,
        urgency_multipliers: Optional[Dict[UrgencyLevel, float]] = None
    ):
        """
        Initialize the integration calculator.

        Args:
            integration_costs: Base costs for integration
            urgency_multipliers: Cost multipliers for different urgency levels
        """
        self.costs = integration_costs or IntegrationCosts()
        self.urgency_multipliers = urgency_multipliers or {
            UrgencyLevel.STANDARD: 1.0,
            UrgencyLevel.ACCELERATED: 1.25,
            UrgencyLevel.CRITICAL: 1.5
        }
        self._init_integration_times()

    def _init_integration_times(self) -> None:
        """Initialize integration times for different satellite types."""
        self.integration_times = {
            SatelliteType.MILSATCOM: IntegrationTimes(5, 3, 0.1),
            SatelliteType.INTEGRATED: IntegrationTimes(15, 7, 0.167),
            SatelliteType.NON_INTEGRATED: IntegrationTimes(365, 180, 90),
            SatelliteType.MIXED: IntegrationTimes(30, 15, 5)
        }

    def calculate_integration_metrics(
        self,
        satellites: pd.DataFrame,
        urgency_level: str,
        base_config: Dict,
        satellite_type: str
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate integration cost and time for satellites.
        
        MILSATCOM satellites should have very short integration times as they're pre-integrated.
        """
        # Define integration times based on satellite type and urgency
        integration_times = {
            'MILSATCOM': {
                'Standard': random.uniform(5,12),
                'Accelerated': random.uniform(4, 1),
                'Critical': random.uniform(0.1, 0.5)
            },
            'Integrated': {
                'Standard': random.uniform(15,30),
                'Accelerated': random.uniform(7,15),
                'Critical': random.uniform(0.167,5)
            },
            'Non-Integrated': {
                'Standard': random.uniform(180,365),
                'Accelerated': random.uniform(90,180),
                'Critical': random.uniform(45,90)
            },
            'Mixed': {
                'Standard': random.uniform(30,60),
                'Accelerated': random.uniform(15,30),
                'Critical': random.uniform(5,15)
            }
        }
        
        try:
            # Get the appropriate integration time
            if satellite_type not in integration_times:
                logging.error(f"Unknown satellite type: {satellite_type}")
                satellite_type = 'Mixed'  # Default fallback
                
            time_lookup = integration_times[satellite_type]
            integration_time = time_lookup.get(urgency_level, time_lookup['Standard'])
            
            # For MILSATCOM and Integrated satellites, no additional integration cost
            if satellite_type in ['MILSATCOM', 'Integrated']:
                integration_cost = 0
            else:
                # Calculate integration cost for non-integrated satellites
                num_sats = satellites['Satellite Name'].nunique()
                hardware_cost = 250_000_000
                license_cost = 90_680.10
                base_integration_cost = hardware_cost + license_cost
                
                # Cost multiplier based on urgency
                urgency_multipliers = {
                    'Standard': 1.0,
                    'Accelerated': 1.25,
                    'Critical': 1.5
                }
                urgency_multiplier = urgency_multipliers.get(urgency_level, 1.0)
                integration_cost = base_integration_cost * urgency_multiplier * num_sats
            
            # Create integration times DataFrame
            integration_times_df = satellites[['Satellite Name']].drop_duplicates()
            integration_times_df['Integration Time'] = integration_time
            
            logging.debug(
                f"Integration metrics calculated for {satellite_type}:\n"
                f"Urgency Level: {urgency_level}\n"
                f"Integration Time: {integration_time} days\n"
                f"Integration Cost: ${integration_cost:,.2f}"
            )
            
            return integration_cost, integration_times_df
            
        except Exception as e:
            logging.error(f"Error calculating integration metrics: {str(e)}")
            raise

    def _get_integration_time(
        self,
        sat_type: SatelliteType,
        urgency: UrgencyLevel
    ) -> float:
        """Get integration time based on satellite type and urgency level."""
        times = self.integration_times[sat_type]
        if urgency == UrgencyLevel.STANDARD:
            return times.standard
        elif urgency == UrgencyLevel.ACCELERATED:
            return times.accelerated
        else:
            return times.critical

    def _calculate_total_cost(
        self,
        satellites: pd.DataFrame,
        sat_type: SatelliteType,
        urgency: UrgencyLevel
    ) -> float:
        """Calculate total integration cost."""
        if sat_type in [SatelliteType.MILSATCOM, SatelliteType.INTEGRATED]:
            return 0.0
            
        num_sats = satellites['Satellite Name'].nunique()
        base_cost = self.costs.hardware + self.costs.license
        urgency_multiplier = self.urgency_multipliers[urgency]
        
        return base_cost * urgency_multiplier * num_sats

    @staticmethod
    def _create_integration_times_df(
        satellites: pd.DataFrame,
        integration_time: float
    ) -> pd.DataFrame:
        """Create DataFrame with integration times per satellite."""
        unique_satellites = satellites[['Satellite Name']].drop_duplicates()
        unique_satellites['Integration Time'] = integration_time
        return unique_satellites


# Convenience function for backward compatibility
def calculate_integration_cost_time(
    satellites: pd.DataFrame,
    urgency_level: str,
    base_config: Dict,
    satellite_type: str
) -> Tuple[float, pd.DataFrame]:
    """
    Backward compatibility function for integration calculations.
    """
    calculator = IntegrationCalculator()
    return calculator.calculate_integration_metrics(
        satellites,
        urgency_level,
        base_config,
        satellite_type
    )