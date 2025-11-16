"""
Node Ranking Engine for Power System Siting Analysis

This module provides a fast, vectorized ranking system for ~20,000 power-system nodes
to identify optimal locations for large electric loads (data centers, electrolyzers, etc.).

All operations are fully vectorized using pandas/numpy for maximum performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# PHYSICAL / EMISSIONS CONSTANTS
# ============================================================================

HOURS_PER_YEAR = 8760.0

# Capacity factors for constant vs flexible operating profiles
CF_CONST = 0.95
CF_FLEX = 0.70

# Emissions intensity mapping (kg CO2/MWh)
E_MIN_PHYS = 150.0   # very clean grid
E_MAX_PHYS = 700.0   # coal-heavy grid
E_RAW_MIN = 0.0
E_RAW_MAX = 950.0    # clamp tail of county column

FLEX_INTENSITY_DISCOUNT = 0.10  # flex loads assumed 10% cleaner


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def robust_min_max(series: pd.Series, clip_low: float = 0.05, clip_high: float = 0.95) -> pd.Series:
    """
    Robust min-max normalization with quantile clipping.
    
    Clips outliers at specified quantiles and linearly scales to [0, 1].
    If the series is constant, returns 0.5 for all non-null entries.
    
    Args:
        series: Input series to normalize
        clip_low: Lower quantile for clipping (default 5%)
        clip_high: Upper quantile for clipping (default 95%)
    
    Returns:
        Normalized series in [0, 1] range
    """
    if series.isna().all():
        return pd.Series(0.5, index=series.index)
    
    # Compute quantile bounds
    q_low = series.quantile(clip_low)
    q_high = series.quantile(clip_high)
    
    # Handle constant series
    if q_high - q_low < 1e-9:
        return pd.Series(0.5, index=series.index)
    
    # Clip and normalize
    clipped = series.clip(lower=q_low, upper=q_high)
    normalized = (clipped - q_low) / (q_high - q_low)
    
    return normalized.fillna(0.5)


def invert_score(z: pd.Series) -> pd.Series:
    """
    Inverts a [0,1] score where lower raw values are better.
    
    Converts "lower is better" metrics (like cost, emissions) to 
    "higher is better" scores for consistent combination.
    
    Args:
        z: Score series in [0, 1] range where lower is better
    
    Returns:
        Inverted series where higher is better
    """
    return 1.0 - z


# ============================================================================
# SPATIAL FILTERING
# ============================================================================

def haversine_distance_vectorized(lat1: float, lon1: float, 
                                   lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Vectorized haversine distance calculation in kilometers.
    
    Computes great-circle distance between a point and a series of points.
    
    Args:
        lat1, lon1: Reference point coordinates
        lat2, lon2: Series of target point coordinates
    
    Returns:
        Distance in kilometers for each target point
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371.0
    
    return c * r


def apply_spatial_filter(df: pd.DataFrame, location_filter: Optional[Dict]) -> pd.DataFrame:
    """
    Apply spatial filtering based on states or radial distance.
    
    Args:
        df: DataFrame with latitude, longitude, and state columns
        location_filter: Either {"states": [...]} or {"lat": x, "lon": y, "radius_km": r} or None
    
    Returns:
        Filtered DataFrame
    """
    if location_filter is None:
        return df
    
    if "states" in location_filter:
        # State-based filter
        states = location_filter["states"]
        return df[df["state"].isin(states)].copy()
    
    elif all(k in location_filter for k in ["lat", "lon", "radius_km"]):
        # Radial filter
        lat = location_filter["lat"]
        lon = location_filter["lon"]
        radius_km = location_filter["radius_km"]
        
        distances = haversine_distance_vectorized(lat, lon, df["latitude"], df["longitude"])
        return df[distances <= radius_km].copy()
    
    return df


# ============================================================================
# DATA VALIDATION AND PREPROCESSING
# ============================================================================

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates critical fields and handles missing/bad values.
    
    Drops rows with missing critical fields and coerces numeric columns.
    For non-critical numeric columns, imputes with median.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Critical fields that must be present
    critical_fields = [
        'avg_lmp', 'avg_price_per_acre', 'county_emissions_intensity_kg_per_mwh',
        'latitude', 'longitude', 'state'
    ]
    
    # Drop rows with missing critical fields
    initial_rows = len(df)
    df = df.dropna(subset=critical_fields)
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing critical fields")
    
    # Pre-process land price which may be stored as strings with commas
    if 'avg_price_per_acre' in df.columns:
        df['avg_price_per_acre'] = (
            df['avg_price_per_acre']
            .astype(str)
            .str.replace(',', '', regex=False)
        )

    # Numeric columns to coerce and validate
    numeric_cols = [
        'avg_lmp', 'avg_energy', 'avg_congestion', 'avg_loses',
        'avg_price_per_acre', 'county_emissions_intensity_kg_per_mwh',
        'latitude', 'longitude',
        'is_h2_hub_state', 'state_dc_incentive_level', 'state_clean_energy_friendly',
        'has_hosting_capacity_map', 'policy_fit_electrolyzer', 'policy_fit_datacenter',
        'queue_pending_mw', 'queue_advanced_share', 'queue_renewable_storage_share',
        'queue_pressure_index', 'price_variance_score'
    ]
    
    # Coerce to numeric and handle bad values with median imputation
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # Ensure policy and queue scores are bounded [0, 1] approximately
    bounded_cols = [
        'is_h2_hub_state', 'state_dc_incentive_level', 'state_clean_energy_friendly',
        'has_hosting_capacity_map', 'policy_fit_electrolyzer', 'policy_fit_datacenter',
        'queue_advanced_share', 'queue_renewable_storage_share'
    ]
    
    for col in bounded_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0, upper=1.0)
    
    return df


# ============================================================================
# COMPONENT SCORE CALCULATION
# ============================================================================

def compute_cost_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes cost score (higher is better).
    
    Uses LMP as primary cost proxy, normalizes, and inverts to 
    make lower costs score higher.
    
    Args:
        df: DataFrame with avg_lmp column
    
    Returns:
        Cost score series [0, 1] where higher is better
    """
    # Cost proxy: primarily LMP (could add congestion/losses if needed)
    total_cost_proxy = df['avg_lmp'].copy()
    
    # Normalize using robust method
    cost_norm_raw = robust_min_max(total_cost_proxy)
    
    # Invert: lower cost is better
    cost_score = invert_score(cost_norm_raw)
    
    return cost_score


def compute_land_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes land cost score (higher is better).
    
    Normalizes land price per acre and inverts so cheaper land scores higher.
    
    Args:
        df: DataFrame with avg_price_per_acre column
    
    Returns:
        Land score series [0, 1] where higher is better
    """
    land_cost_norm_raw = robust_min_max(df['avg_price_per_acre'])
    land_score = invert_score(land_cost_norm_raw)
    
    return land_score


def compute_emissions_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes emissions score (higher is better).
    
    Normalizes emissions intensity and inverts so lower emissions score higher.
    
    Args:
        df: DataFrame with county_emissions_intensity_kg_per_mwh column
    
    Returns:
        Emissions score series [0, 1] where higher is better
    """
    emissions_intensity_norm_raw = robust_min_max(df['county_emissions_intensity_kg_per_mwh'])
    emissions_score = invert_score(emissions_intensity_norm_raw)
    
    return emissions_score


def compute_policy_score(df: pd.DataFrame, load_type: str) -> pd.Series:
    """
    Computes policy alignment score (higher is better).
    
    Uses pre-computed policy fit scores for electrolyzer/datacenter loads,
    and combines underlying primitives for other load types.
    
    Args:
        df: DataFrame with policy-related columns
        load_type: Type of load being sited
    
    Returns:
        Policy score series [0, 1] where higher is better
    """
    if load_type in ["h2_electrolyzer_firm"]:
        policy_base = df['policy_fit_electrolyzer'].copy()
    elif load_type in ["data_center_always_on", "data_center_flexible"]:
        policy_base = df['policy_fit_datacenter'].copy()
    else:
        # For other load types, combine underlying policy primitives
        policy_base = (
            0.3 * df['state_clean_energy_friendly'] +
            0.3 * df['state_dc_incentive_level'] +
            0.2 * df['is_h2_hub_state'] +
            0.2 * df['has_hosting_capacity_map']
        )
    
    # Normalize to [0, 1]
    policy_score = robust_min_max(policy_base)
    
    return policy_score


def compute_queue_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes interconnection queue score (higher is better).
    
    Combines multiple queue metrics:
    - Pending MW (lower is better)
    - Advanced share (higher is better)
    - Pressure index (lower is better)
    - Renewable/storage share (higher is better)
    
    Args:
        df: DataFrame with queue-related columns
    
    Returns:
        Queue score series [0, 1] where higher is better
    """
    # Pending MW: less is better
    queue_pending_norm_raw = robust_min_max(df['queue_pending_mw'])
    queue_pending_score = invert_score(queue_pending_norm_raw)
    
    # Advanced share: more is better
    queue_advanced_score = robust_min_max(df['queue_advanced_share'])
    
    # Green share: more is better
    queue_green_share_score = robust_min_max(df['queue_renewable_storage_share'])
    
    # Pressure index: less is better
    queue_pressure_norm_raw = robust_min_max(df['queue_pressure_index'])
    queue_pressure_score = invert_score(queue_pressure_norm_raw)
    
    # Combine into composite score
    queue_score = (
        0.4 * queue_pending_score +
        0.2 * queue_advanced_score +
        0.2 * queue_pressure_score +
        0.2 * queue_green_share_score
    )
    
    # Renormalize to [0, 1]
    queue_score = robust_min_max(queue_score)
    
    return queue_score


def _get_operating_profile(load_type: str) -> Tuple[float, float, bool]:
    """
    Returns (capacity_factor, land_acres_per_mw, is_flexible) for a given load_type.

    This groups detailed load types into two operating modes:
    - Constant: high capacity factor, compact land footprint
    - Flexible: lower capacity factor, larger land footprint
    """
    # Constant-load types (24/7 style)
    constant_types = {
        "data_center_always_on",
        "h2_electrolyzer_firm",
        "industrial_continuous",
        "commercial_campus",
    }
    # Flexible / shiftable types
    flexible_types = {
        "data_center_flexible",
        "industrial_flexible",
    }

    if load_type in flexible_types:
        # Flexible: lower CF, larger land footprint
        return 0.70, 1.50, True
    else:
        # Default to constant behaviour for anything else
        return 0.95, 0.75, False


def compute_physical_and_cost_metrics(
    df: pd.DataFrame,
    load_type: str,
    load_size_mw: float,
) -> pd.DataFrame:
    """
    Computes annual energy, cost, emissions, and land metrics for each node.

    Implements the formula set provided in the spec, using:
    - avg_lmp
    - price_variance_score
    - county_emissions_intensity_kg_per_mwh
    - avg_price_per_acre
    and user input load_size_mw + load_type.
    """
    # Operating profile by load type
    cf_profile, land_acres_per_mw, is_flexible = _get_operating_profile(load_type)

    # ------------------------------------------------------------------
    # 1) Annual MWh using standardized capacity factors
    # ------------------------------------------------------------------
    if is_flexible:
        cf = CF_FLEX
    else:
        cf = CF_CONST

    annual_mwh = float(load_size_mw) * HOURS_PER_YEAR * cf
    df["annual_mwh"] = annual_mwh

    # ------------------------------------------------------------------
    # 2) Volatility normalization: v_norm = clip((V - 1.0) / 4.30, 0, 1)
    #     (unchanged from previous spec)
    # ------------------------------------------------------------------
    if "price_variance_score" in df.columns:
        v_norm = ((df["price_variance_score"] - 1.0) / 4.30).clip(lower=0.0, upper=1.0)
    else:
        # Fallback: no variance information, treat as zero volatility
        v_norm = pd.Series(0.0, index=df.index)

    # 3) Effective energy price P_eff
    if is_flexible:
        # Flexible load: discount with beta = 0.25
        price_multiplier = 1.0 - 0.25 * v_norm
    else:
        # Constant load: penalty with alpha = 0.30
        price_multiplier = 1.0 + 0.30 * v_norm

    df["effective_price_per_mwh"] = df["avg_lmp"] * price_multiplier

    # 4) Annual energy cost
    df["annual_energy_cost"] = df["annual_mwh"] * df["effective_price_per_mwh"]

    # ------------------------------------------------------------------
    # 5) Annual emissions using realistic grid-intensity mapping
    # ------------------------------------------------------------------
    if "county_emissions_intensity_kg_per_mwh" in df.columns:
        # 2) Normalize county emissions column to [0,1]
        E_raw = df["county_emissions_intensity_kg_per_mwh"].clip(
            lower=E_RAW_MIN,
            upper=E_RAW_MAX,
        )
        e_norm = E_raw / E_RAW_MAX  # 0..1

        # 3) Map to realistic grid intensity (kg CO2/MWh)
        E_phys = E_MIN_PHYS + e_norm * (E_MAX_PHYS - E_MIN_PHYS)

        # Optional flex benefit
        if is_flexible:
            E_eff = E_phys * (1.0 - FLEX_INTENSITY_DISCOUNT)
        else:
            E_eff = E_phys
    else:
        # Fallback if the emissions column is missing: use clean grid baseline
        base_intensity = E_MIN_PHYS * (1.0 - FLEX_INTENSITY_DISCOUNT) if is_flexible else E_MIN_PHYS
        E_eff = pd.Series(base_intensity, index=df.index)

    annual_emissions_kg = df["annual_mwh"] * E_eff
    df["annual_emissions_tonnes"] = annual_emissions_kg / 1000.0

    # ------------------------------------------------------------------
    # 6) Land area and land cost
    # ------------------------------------------------------------------
    land_acres = float(load_size_mw) * land_acres_per_mw
    df["land_acres"] = land_acres

    if "avg_price_per_acre" in df.columns:
        p_acre_eff = df["avg_price_per_acre"].clip(lower=0.0, upper=30000.0)
    else:
        p_acre_eff = pd.Series(0.0, index=df.index)

    df["land_cost"] = land_acres * p_acre_eff

    return df


def compute_variability_scores(df: pd.DataFrame, resource_config: str) -> Tuple[pd.Series, pd.Series]:
    """
    Computes baseline and effective price variability penalty scores.
    
    Applies Variability Adjustment Factor (VAF) based on on-site resources.
    Non-RTO nodes (price_variance_score == 1) are never scaled.
    
    Args:
        df: DataFrame with price_variance_score column
        resource_config: One of "none", "solar", "battery", "solar_battery", "firm_gen"
    
    Returns:
        Tuple of (baseline_penalty_score, effective_penalty_score), both [0, 1] higher is better
    """
    # Variability Adjustment Factors
    VAF = {
        "none": 1.0,
        "solar": 0.7,
        "battery": 0.6,
        "solar_battery": 0.4,
        "firm_gen": 0.25,
    }
    
    # Baseline (no on-site resources)
    baseline_price_variance_score = df['price_variance_score'].copy()
    
    # Compute effective variance with VAF
    vaf = VAF.get(resource_config, 1.0)
    is_non_rto = (df['price_variance_score'] == 1.0)
    
    if resource_config == "none":
        effective_price_variance_score = baseline_price_variance_score.copy()
    else:
        # Apply VAF, but never scale non-RTO nodes
        effective_price_variance_score = df['price_variance_score'] * vaf
        effective_price_variance_score[is_non_rto] = 1.0
    
    # Normalize and invert to penalty scores (higher is better)
    baseline_price_variance_norm = robust_min_max(baseline_price_variance_score)
    price_variability_penalty_score = invert_score(baseline_price_variance_norm)
    
    effective_price_variance_norm = robust_min_max(effective_price_variance_score)
    effective_price_variability_penalty_score = invert_score(effective_price_variance_norm)
    
    return price_variability_penalty_score, effective_price_variability_penalty_score


# ============================================================================
# WEIGHT ADJUSTMENT LOGIC
# ============================================================================

def get_load_type_multipliers(load_type: str) -> Dict[str, float]:
    """
    Returns weight multipliers based on load type characteristics.
    
    Different load types have different priorities:
    - Data centers: care about land, variability
    - Electrolyzers: care about queue, emissions
    - Industrial: care about cost, queue
    
    Args:
        load_type: Type of load
    
    Returns:
        Dictionary of multipliers for each weight component
    """
    multipliers = {
        "data_center_always_on": {
            "cost": 1.0,
            "land": 1.7,
            "policy": 1.2,
            "queue": 1.0,
            "emissions": 1.1,
            "variability": 1.4,
        },
        "data_center_flexible": {
            "cost": 1.0,
            "land": 1.5,
            "policy": 1.1,
            "queue": 1.0,
            "emissions": 1.0,
            "variability": 0.7,
        },
        "h2_electrolyzer_firm": {
            "cost": 1.0,
            "land": 0.3,
            "policy": 1.2,
            "queue": 1.6,
            "emissions": 1.7,
            "variability": 1.0,
        },
        "industrial_continuous": {
            "cost": 1.1,
            "land": 1.0,
            "policy": 1.0,
            "queue": 1.3,
            "emissions": 1.0,
            "variability": 1.2,
        },
        "industrial_flexible": {
            "cost": 1.1,
            "land": 0.9,
            "policy": 1.0,
            "queue": 1.2,
            "emissions": 1.0,
            "variability": 0.6,
        },
        "commercial_campus": {
            "cost": 1.0,
            "land": 1.2,
            "policy": 1.0,
            "queue": 1.0,
            "emissions": 1.1,
            "variability": 0.8,
        },
    }
    
    return multipliers.get(load_type, {
        "cost": 1.0,
        "land": 1.0,
        "policy": 1.0,
        "queue": 1.0,
        "emissions": 1.0,
        "variability": 1.0,
    })


def get_size_multipliers(load_size_mw: float) -> Dict[str, float]:
    """
    Returns weight multipliers based on load size.
    
    Larger loads have greater grid impact, increasing importance of
    queue capacity and variability management.
    
    Args:
        load_size_mw: Load size in MW
    
    Returns:
        Dictionary of multipliers for affected components
    """
    if load_size_mw < 50:
        # Small loads: queue and variability less critical
        return {"queue": 0.8, "variability": 0.8}
    elif load_size_mw <= 150:
        # Medium loads: neutral
        return {"queue": 1.0, "variability": 1.0}
    else:
        # Large loads: queue and variability very important, cost slightly less
        return {"queue": 1.4, "variability": 1.3, "cost": 0.9}


def compute_final_weights(load_type: str, load_size_mw: float, 
                          emissions_preference: float) -> Dict[str, float]:
    """
    Computes final normalized weights for all scoring components.
    
    Combines base weights with adjustments for load type, size, and 
    emissions preference, then normalizes to sum to 1.
    
    Args:
        load_type: Type of load
        load_size_mw: Load size in MW
        emissions_preference: User preference 0-100 (0=don't care, 100=very sensitive)
    
    Returns:
        Normalized weight dictionary
    """
    # Base weights (neutral scenario)
    base_weights = {
        "cost": 0.30,
        "land": 0.15,
        "policy": 0.15,
        "queue": 0.15,
        "emissions": 0.15,
        "variability": 0.10,
    }
    
    # Get multipliers
    load_type_m = get_load_type_multipliers(load_type)
    size_m = get_size_multipliers(load_size_mw)
    
    # Emissions factor: maps 0-100 to 0.5-2.0
    emissions_factor = 0.5 + 1.5 * (emissions_preference / 100.0)
    
    # If emissions preference is very high, modestly reduce cost and queue
    if emissions_preference > 80:
        cost_emissions_adjust = 0.85
        queue_emissions_adjust = 0.9
    else:
        cost_emissions_adjust = 1.0
        queue_emissions_adjust = 1.0
    
    # Apply all multipliers
    weights = {}
    for key in base_weights:
        w = base_weights[key]
        w *= load_type_m.get(key, 1.0)
        w *= size_m.get(key, 1.0)
        
        if key == "emissions":
            w *= emissions_factor
        elif key == "cost":
            w *= cost_emissions_adjust
        elif key == "queue":
            w *= queue_emissions_adjust
        
        weights[key] = w
    
    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights


# ============================================================================
# MAIN RANKING FUNCTION
# ============================================================================

def rank_nodes(
    nodes_df: pd.DataFrame,
    load_type: str,
    load_size_mw: float,
    location_filter: Optional[Dict],
    emissions_preference: float,
    resource_config: str,
    top_n: int = 200
) -> pd.DataFrame:
    """
    Ranks power system nodes for siting a large electric load.
    
    This is the main entry point for the ranking engine. It performs:
    1. Data validation and cleaning
    2. Spatial filtering
    3. Component score calculation
    4. Weight adjustment based on load characteristics
    5. Composite scoring with baseline and scenario comparisons
    6. Final ranking and top-N selection
    
    Args:
        nodes_df: DataFrame with node data (see module docstring for required columns)
        load_type: One of "data_center_always_on", "data_center_flexible", 
                   "h2_electrolyzer_firm", "industrial_continuous", 
                   "industrial_flexible", "commercial_campus"
        load_size_mw: Load size in MW
        location_filter: Spatial filter dict or None:
                        - {"states": ["CA", "NY", ...]} for state filter
                        - {"lat": x, "lon": y, "radius_km": r} for radial filter
                        - None for no filter
        emissions_preference: User slider 0-100 (0=don't care, 100=very sensitive)
        resource_config: One of "none", "solar", "battery", "solar_battery", "firm_gen"
        top_n: Number of top-ranked nodes to return (default 200)
    
    Returns:
        DataFrame with top_n ranked nodes, including:
        - All original columns
        - Component scores (cost_score, land_score, etc.)
        - Composite scores (score_baseline, score_scenario)
        - Rankings (rank_baseline, rank_scenario)
    """
    # Validate inputs
    valid_load_types = [
        "data_center_always_on", "data_center_flexible", 
        "h2_electrolyzer_firm", "industrial_continuous",
        "industrial_flexible", "commercial_campus"
    ]
    if load_type not in valid_load_types:
        raise ValueError(f"Invalid load_type. Must be one of {valid_load_types}")
    
    valid_resource_configs = ["none", "solar", "battery", "solar_battery", "firm_gen"]
    if resource_config not in valid_resource_configs:
        raise ValueError(f"Invalid resource_config. Must be one of {valid_resource_configs}")
    
    if not (0 <= emissions_preference <= 100):
        raise ValueError("emissions_preference must be between 0 and 100")
    
    # Step 1: Validate and clean data
    print(f"Starting node ranking for {load_type} ({load_size_mw} MW)")
    df = validate_and_clean_data(nodes_df)
    print(f"Validated data: {len(df)} nodes")
    
    # Step 2: Apply spatial filtering
    df = apply_spatial_filter(df, location_filter)
    print(f"After spatial filter: {len(df)} nodes")
    
    if len(df) == 0:
        print("Warning: No nodes remain after filtering")
        return pd.DataFrame()
    
    # Step 3: Compute component scores
    print("Computing component scores...")
    df['cost_score'] = compute_cost_score(df)
    df['land_score'] = compute_land_score(df)
    df['emissions_score'] = compute_emissions_score(df)
    df['policy_score'] = compute_policy_score(df, load_type)
    df['queue_score'] = compute_queue_score(df)
    
    # Step 4: Compute variability scores (baseline and effective)
    df['price_variability_penalty_score'], df['effective_price_variability_penalty_score'] = \
        compute_variability_scores(df, resource_config)
    
    # Step 5: Compute physical + cost/emissions/land metrics for this load
    print("Computing annual energy, cost, emissions, and land metrics...")
    df = compute_physical_and_cost_metrics(
        df,
        load_type=load_type,
        load_size_mw=load_size_mw,
    )
    
    # Step 6: Optional fast pre-filter to remove obviously poor candidates
    # Keep nodes that have at least one strong component or aren't terrible on all
    pre_filter_mask = (
        (df['cost_score'] >= 0.3) |
        (df['queue_score'] >= 0.3) |
        (df['emissions_score'] >= 0.3) |
        (df['policy_score'] >= 0.3)
    )
    df = df[pre_filter_mask].copy()
    print(f"After quality pre-filter: {len(df)} nodes")
    
    if len(df) == 0:
        print("Warning: No nodes passed quality threshold")
        return pd.DataFrame()
    
    # Step 7: Compute final weights
    weights = compute_final_weights(load_type, load_size_mw, emissions_preference)
    print(f"Final weights: {weights}")
    
    # Step 8: Compute composite scores
    # Baseline score (no on-site resources)
    df['score_baseline'] = (
        weights['cost'] * df['cost_score'] +
        weights['land'] * df['land_score'] +
        weights['policy'] * df['policy_score'] +
        weights['queue'] * df['queue_score'] +
        weights['emissions'] * df['emissions_score'] +
        weights['variability'] * df['price_variability_penalty_score']
    )
    
    # Scenario score (with selected resource_config)
    df['score_scenario'] = (
        weights['cost'] * df['cost_score'] +
        weights['land'] * df['land_score'] +
        weights['policy'] * df['policy_score'] +
        weights['queue'] * df['queue_score'] +
        weights['emissions'] * df['emissions_score'] +
        weights['variability'] * df['effective_price_variability_penalty_score']
    )
    
    # Step 9: Compute ranks (1 = best)
    df['rank_baseline'] = df['score_baseline'].rank(method='min', ascending=False).astype(int)
    df['rank_scenario'] = df['score_scenario'].rank(method='min', ascending=False).astype(int)
    
    # Step 10: Sort by scenario score and return top N
    df_sorted = df.sort_values('score_scenario', ascending=False)
    result = df_sorted.head(top_n).copy()
    
    print(f"Ranking complete. Returning top {len(result)} nodes.")
    print(f"Top node: {result.iloc[0]['node']} in {result.iloc[0]['state']} "
          f"(score: {result.iloc[0]['score_scenario']:.3f})")
    
    return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_nodes_from_csv(filepath: str) -> pd.DataFrame:
    """
    Loads node data from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with node data
    """
    return pd.read_csv(filepath)


def get_ranking_explanation(node_row: pd.Series, weights: Dict[str, float]) -> str:
    """
    Generates human-readable explanation for why a node ranks well.
    
    Args:
        node_row: Series representing a single node's data
        weights: Weight dictionary used in scoring
    
    Returns:
        Explanation string
    """
    explanations = []
    
    # Identify top contributing factors
    contributions = {
        'cost': weights['cost'] * node_row.get('cost_score', 0),
        'land': weights['land'] * node_row.get('land_score', 0),
        'policy': weights['policy'] * node_row.get('policy_score', 0),
        'queue': weights['queue'] * node_row.get('queue_score', 0),
        'emissions': weights['emissions'] * node_row.get('emissions_score', 0),
        'variability': weights['variability'] * node_row.get('effective_price_variability_penalty_score', 0),
    }
    
    # Sort by contribution
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    # Generate explanation for top 3 factors
    factor_names = {
        'cost': 'low energy costs',
        'land': 'affordable land',
        'policy': 'strong policy support',
        'queue': 'low interconnection queue pressure',
        'emissions': 'low emissions intensity',
        'variability': 'low price variability exposure',
    }
    
    for factor, contrib in sorted_contrib[:3]:
        if contrib > 0.1:  # Only mention significant contributions
            explanations.append(factor_names[factor])
    
    node_name = node_row.get('node', 'This node')
    location = f"{node_row.get('county_state_pairs', node_row.get('state', 'Unknown'))}"
    
    return f"{node_name} in {location} ranks highly due to {', '.join(explanations)}."


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Node Ranking Engine - Example Usage\n")
    
    # Load data
    print("Loading node data...")
    nodes_df = load_nodes_from_csv("final_csv_v1_new.csv")
    print(f"Loaded {len(nodes_df)} nodes\n")
    
    # Example 1: Data center in California with solar+battery
    print("=" * 80)
    print("EXAMPLE 1: Data Center in California (250 MW, high emissions sensitivity)")
    print("=" * 80)
    results_1 = rank_nodes(
        nodes_df=nodes_df,
        load_type="data_center_always_on",
        load_size_mw=250,
        location_filter={"states": ["CA"]},
        emissions_preference=80,
        resource_config="solar_battery",
        top_n=10
    )
    
    print("\nTop 10 nodes:")
    print(results_1[['node', 'state', 'county_state_pairs', 'score_scenario', 'rank_scenario', 
                     'avg_lmp', 'emissions_score', 'cost_score']].to_string(index=False))
    
    # Example 2: Electrolyzer with flexible location
    print("\n" + "=" * 80)
    print("EXAMPLE 2: H2 Electrolyzer (100 MW, moderate emissions sensitivity)")
    print("=" * 80)
    results_2 = rank_nodes(
        nodes_df=nodes_df,
        load_type="h2_electrolyzer_firm",
        load_size_mw=100,
        location_filter=None,  # No spatial filter
        emissions_preference=60,
        resource_config="none",
        top_n=10
    )
    
    print("\nTop 10 nodes:")
    print(results_2[['node', 'state', 'iso', 'score_scenario', 'rank_scenario',
                     'queue_score', 'emissions_score', 'policy_score']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Examples complete!")

