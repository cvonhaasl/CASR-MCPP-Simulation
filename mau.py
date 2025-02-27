# # mau.py
"""
Multi-Attribute Utility (MAU) Analysis Module
==========================================

This module implements Multi-Attribute Utility Theory (MAUT) for evaluating
satellite system configurations across multiple performance attributes.

Mathematical Model
----------------
1. Single Attribute Utility (SAU):
   For benefit attributes:  U(x) = (x - min) / (max - min)
   For cost attributes:    U(x) = (max - x) / (max - min)
   where:
   - x: Attribute value
   - min/max: Attribute bounds
   - U(x): Normalized utility [0,1]

2. Multi-Attribute Utility:
   MAU = Σ(kᵢ * Uᵢ(xᵢ))
   where:
   - kᵢ: Weight of attribute i (Σkᵢ = 1)
   - Uᵢ(xᵢ): Normalized utility of attribute i

"""


import math

def sau_time_to_meet_demand(time_days: float, alpha: float = 0.02) -> float:
    """
    Compute the SAU for time to meet demand using a negative exponential.
    U_time(d) = exp(-alpha * d)

    - At d = 0, utility = 1
    - As d grows, utility smoothly decays toward 0.
    - alpha controls how quickly it decays.

    Parameters:
        time_days (float): Time to meet demand in days.
        alpha (float): Decay rate.

    Returns:
        float: Utility in [0, 1].
    """
    utility = math.exp(-alpha * time_days)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, utility))

def sau_performance(performance: float, demand: float, alpha: float = 2.0) -> float:
    """
    Compute the SAU for performance (mbps) relative to demand using a continuous function.

    - If ratio = performance / demand >= 1, we apply a smooth exponential function
      that starts near some baseline (e.g., 0.8) at ratio=1 and asymptotically approaches 1.
    - If ratio < 1, we use ratio^2 to penalize underperformance more severely.

    Parameters:
        performance (float): Achieved performance in mbps.
        demand (float): Required performance in mbps.
        alpha (float): Controls how quickly the utility approaches 1 above ratio=1.

    Returns:
        float: Utility in [0, 1].
    """
    if demand <= 0:
        return 0.0  

    ratio = performance / demand

    if ratio < 0:
       
        return 0.0

    if ratio < 1.0:
     
        return ratio ** 2
    else:

        baseline = 0.8
        return min(
            1.0,
            baseline + (1.0 - baseline) * (1.0 - math.exp(-alpha * (ratio - 1.0)))
        )

def sau_coverage(coverage: float, target_coverage: float = 100.0, beta: float = 0.7) -> float:
    """
    Compute the SAU for coverage area using a coverage ratio and a smooth power function.
    ratio = coverage / target_coverage

    - If ratio >= 1 => utility = 1
    - Else => ratio^beta, giving a smooth increase as coverage grows.

    Parameters:
        coverage (float): Actual coverage area in km^2.
        target_coverage (float): Target or required coverage area in km^2.
        beta (float): Exponent controlling the curve shape. 0 < beta <= 1 => concave shape.

    Returns:
        float: Utility in [0, 1].
    """
    if target_coverage <= 0:
        return 0.0

    ratio = coverage / target_coverage

    if ratio >= 1.0:
        return 1.0
    else:
        # ratio^beta yields a smooth curve from 0 to 1 as coverage goes 0 to target_coverage
        utility = ratio ** beta
        return max(0.0, min(1.0, utility))

def sau_redundancy(redundancy: float, target_redundancy: float = 3.0, gamma: float = 0.5) -> float:
    """
    Compute the SAU for redundancy factor using a redundancy ratio and a smooth power function.
    ratio = redundancy / target_redundancy

    - If ratio >= 1 => utility = 1
    - Else => ratio^gamma

    Parameters:
        redundancy (float): Actual redundancy factor.
        target_redundancy (float): Desired redundancy factor.
        gamma (float): Exponent controlling the curve shape.

    Returns:
        float: Utility in [0, 1].
    """
    if target_redundancy <= 0:
        return 0.0

    ratio = redundancy / target_redundancy

    if ratio >= 1.0:
        return 1.0
    else:
        utility = ratio ** gamma
        return max(0.0, min(1.0, utility))

def calculate_mau(attributes: dict, weights: dict = None) -> float:
    """
    Compute the overall Multi-Attribute Utility (MAU) as a weighted sum of individual SAU scores,
    using continuous SAU functions for time, performance, coverage, and redundancy.

    Parameters:
        attributes (dict): Must contain:
            - 'time_to_meet_demand' (float)
            - 'performance' (float)
            - 'demand' (float)
            - 'coverage' (float)
            - 'redundancy' (float)
        weights (dict, optional): A dictionary of weights for each attribute.
            Expected keys: 'time', 'performance', 'coverage', 'redundancy'
            Default is equal weighting.

    Returns:
        float: The overall MAU value in [0, 1].
    """
    if weights is None:
        weights = {
            'time': 0.4,
            'performance': 0.3,
            'coverage': 0.15,
            'redundancy': 0.15
        }

    # Calculate SAU values using the new continuous functions
    sau_time = sau_time_to_meet_demand(attributes.get('time_to_meet_demand', 999))
    sau_perf = sau_performance(attributes.get('performance', 0), attributes.get('demand', 1))
    sau_cov = sau_coverage(attributes.get('coverage', 0))
    sau_red = sau_redundancy(attributes.get('redundancy', 0))

    mau = (weights['time'] * sau_time +
           weights['performance'] * sau_perf +
           weights['coverage'] * sau_cov +
           weights['redundancy'] * sau_red)

    return max(0.0, min(1.0, mau))


