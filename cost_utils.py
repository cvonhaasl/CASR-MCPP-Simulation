# cost_utils.py
"""
Satellite Operations Cost Calculation Module
=========================================

This module calculates operational and procurement costs for satellite systems,
including both military (MILSATCOM) and commercial (COMSATCOM) satellites.

Financial Models
--------------
1. Net Present Value (NPV):
   NPV = FV / (1 + r)^t
   where:
   - FV: Future Value (annual cost)
   - r: Discount rate (default 10%)
   - t: Time delta from base year

2. MILSATCOM Annual Costs:
   Total = (Operating_Cost * Num_Satellites * Factor) + (Procurement_Cost/simulation period) + (Launch_Cost/sim_period)
   where:
   - Operating_Cost: Uniform random between $17.3M and $24.3M per satellite
   - Procurement_Cost: Uniform random between $442M and $1.6B, divided by simulation period
   - Launch_Cost: Uniform random between $55M and $90M, divided by simulation period

3. COMSATCOM Costs:
   Monthly_Cost_per_Band = (Effective_Usage / Spectral_Efficiency) * Band_Price
   Annual_Cost = Σ(Monthly_Cost_per_Band * 12)
   where:
   - Effective_Usage: min(Available_Capacity, Demand_Met)
   - Spectral_Efficiency: Default 10 Mbps/MHz

Key Assumptions
-------------
1. MILSATCOM:
   - Operating costs are uniformly distributed
   - Procurement and launch costs are amortized over 10 years
   - Costs scale linearly with number of satellites

2. COMSATCOM:
   - Pay-for-use model based on actual capacity utilized
   - Bandwidth is priced per MHz per month
   - Unused capacity is not charged
   - Even distribution across bands if specific band capacity unknown

3. General:
   - All costs are in current year US dollars
   - NPV calculations use annual compounding
   - Band prices remain constant within calculation period
"""

from typing import Dict, Optional, Tuple, Iterable
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MilsatcomCosts:
    """MILSATCOM cost parameters."""
    min_operating: float = 17_307_000
    max_operating: float = 24_268_000
    min_procurement: float = 442_000_000
    max_procurement: float = 1_600_000_000
    min_launch: float = 55_000_000
    max_launch: float = 90_000_000
    amortization_period: int = 11  # Years to spread procurement/launch costs

class CostCalculator:
    """Calculates operational and procurement costs for satellite systems."""
    
    def __init__(self, discount_rate: float = 0.10):
        """
        Initialize calculator with discount rate.
        
        Args:
            discount_rate: Annual discount rate for NPV (default: 10%)
        """
        self.discount_rate = discount_rate
        self.milsatcom_costs = MilsatcomCosts()

    def calculate_npv(
        self,
        annual_cost: float,
        year: int,
        base_year: int
    ) -> float:
        """Calculate Net Present Value."""
        years_from_base = year - base_year
        if years_from_base < 0:
            #logger.warning(f"Year {year} is before base year {base_year}")
            return annual_cost
        
        return annual_cost / ((1 + self.discount_rate) ** years_from_base)

    def calculate_milsatcom_cost(
        self,
        satellite_data: pd.DataFrame,
        year: int,
        base_year: int,
        operating_cost_factor: float = 1.0
    ) -> float:
        """
        Calculate MILSATCOM costs including operations, procurement, and launch.
        
        Args:
            satellite_data: Satellite information DataFrame
            year: Cost calculation year
            base_year: NPV base year
            operating_cost_factor: Operating cost multiplier
        """
        if satellite_data.empty:
            return 0.0

        costs = self.milsatcom_costs
        num_sats = satellite_data['Satellite Name'].nunique()
        
        # Calculate components with numpy for better performance
        operating_cost = (
            np.random.uniform(costs.min_operating, costs.max_operating) * 
            num_sats * 
            operating_cost_factor
        )
        
        procurement_cost = (
            np.random.uniform(costs.min_procurement, costs.max_procurement) / 
            costs.amortization_period
        )
        
        launch_cost = (
            np.random.uniform(costs.min_launch, costs.max_launch) / 
            costs.amortization_period
        )
        
        annual_cost = operating_cost + procurement_cost + launch_cost
        return self.calculate_npv(annual_cost, year, base_year)

    def calculate_comsatcom_cost(
        self,
        satellite_data: pd.DataFrame,
        band_prices: Dict[str, float],
        year: int,
        base_year: int,
        cost_factor: float = 1.0,
        demand_met: Optional[float] = None,
        spectral_efficiency: float = 10.0
    ) -> float:
        """
        Calculate COMSATCOM costs based on bandwidth usage.
        
        Args:
            satellite_data: Satellite information DataFrame
            band_prices: Price per MHz per month for each band
            year: Cost calculation year
            base_year: NPV base year
            cost_factor: Cost multiplier
            demand_met: Actual capacity used (Mbps)
            spectral_efficiency: Mbps per MHz ratio
        """
        if satellite_data.empty or 'Band' not in satellite_data.columns:
            logger.error("Invalid satellite data for COMSATCOM cost calculation")
            return 0.0
        
        spectral_efficiency = np.random.uniform(10, 40)

        # Calculate total available capacity and effective usage
        total_available = satellite_data['Usable Capacity (Mbps)'].sum()
        if total_available <= 0:
            return 0.0

        effective_total = (
            min(total_available, demand_met) 
            if demand_met is not None 
            else total_available
        )

        # Calculate capacity per band
        band_capacities = self._calculate_band_capacities(
            satellite_data, 
            band_prices.keys()
        )

        # Handle case where no capacity matches expected bands
        if sum(band_capacities.values()) <= 0:
            band_capacities = self._distribute_evenly(
                effective_total, 
                len(band_prices)
            )

        # Calculate costs per band
        total_cost = self._calculate_band_costs(
            band_capacities,
            band_prices,
            effective_total,
            spectral_efficiency
        )

        return self.calculate_npv(total_cost * cost_factor, year, base_year)

    def _calculate_band_capacities(
        self,
        data: pd.DataFrame,
        band_keys: Iterable[str]
    ) -> Dict[str, float]:
        """Calculate available capacity per band."""
        return {
            f"{key.upper()}-BAND": data[
                data['Band'].str.contains(f"{key.upper()}-BAND", case=False, na=False)
            ]['Usable Capacity (Mbps)'].sum()
            for key in band_keys
        }

    @staticmethod
    def _distribute_evenly(total: float, num_bands: int) -> Dict[str, float]:
        """Distribute capacity evenly across bands."""
        per_band = total / num_bands
        return {f"{key.upper()}-BAND": per_band for key in 'ckalx'}

    def _calculate_band_costs(
        self,
        capacities: Dict[str, float],
        prices: Dict[str, float],
        effective_total: float,
        spectral_efficiency: float
    ) -> float:
        """Calculate total cost across all bands."""
        total_capacity = sum(capacities.values())
        total_cost = 0.0

        for band_name, capacity in capacities.items():
            if capacity <= 0:
                continue

            band_key = band_name[0].lower()
            if band_key not in prices:
                continue

            proportion = capacity / total_capacity
            effective_usage = (effective_total * proportion) / spectral_efficiency
            monthly_cost = effective_usage * prices[band_key]
            total_cost += monthly_cost * 12

            logger.debug(
                f"{band_name}: Capacity={capacity:.2f}Mbps, "
                f"Usage={effective_usage:.2f}MHz, "
                f"Annual Cost=${monthly_cost*12:,.2f}"
            )

        return total_cost

    def calculate_total_system_cost(
        self,
        satellite_data: pd.DataFrame,
        option_name: str,
        band_prices: Dict[str, float],
        year: int,
        base_year: int,
        milsatcom_factor: float = 1.0,
        comsatcom_factor: float = 1.0
    ) -> float:
        """Calculate total system cost based on satellite type."""
        if satellite_data.empty:
            return 0.0

        try:
            if 'Satellite Type' in satellite_data.columns:
                return self._calculate_with_type(
                    satellite_data,
                    band_prices,
                    year,
                    base_year,
                    milsatcom_factor,
                    comsatcom_factor
                )
            else:
                return self._calculate_by_option(
                    satellite_data,
                    option_name,
                    band_prices,
                    year,
                    base_year,
                    milsatcom_factor,
                    comsatcom_factor
                )

        except Exception as e:
            logger.error(f"Error in total cost calculation: {str(e)}")
            raise

    def _calculate_with_type(
        self,
        data: pd.DataFrame,
        band_prices: Dict[str, float],
        year: int,
        base_year: int,
        milsatcom_factor: float,
        comsatcom_factor: float
    ) -> float:
        """Calculate costs when satellite type is specified."""
        total_cost = 0.0
        
        # Calculate MILSATCOM costs
        milsatcom_sats = data[data['Satellite Type'] == 'MILSATCOM']
        if not milsatcom_sats.empty:
            total_cost += self.calculate_milsatcom_cost(
                milsatcom_sats,
                year,
                base_year,
                milsatcom_factor
            )

        # Calculate COMSATCOM costs
        comsatcom_sats = data[data['Satellite Type'] == 'COMSATCOM']
        if not comsatcom_sats.empty:
            total_cost += self.calculate_comsatcom_cost(
                comsatcom_sats,
                band_prices,
                year,
                base_year,
                comsatcom_factor
            )

        return total_cost

    def _calculate_by_option(
        self,
        data: pd.DataFrame,
        option_name: str,
        band_prices: Dict[str, float],
        year: int,
        base_year: int,
        milsatcom_factor: float,
        comsatcom_factor: float
    ) -> float:
        """Calculate costs based on option when type is not specified."""
        if option_name == 'Option 1':
            return self.calculate_milsatcom_cost(
                data, year, base_year, milsatcom_factor
            )
        elif option_name == 'Option 2':
            return self.calculate_comsatcom_cost(
                data, band_prices, year, base_year, comsatcom_factor
            )
        elif option_name in ['Option 3', 'Option 4']:
            raise ValueError(f"Satellite Type required for {option_name}")
        else:
            logger.warning(f"Unknown option: {option_name}")
            return 0.0