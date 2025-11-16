# Node Ranking Engine for Power System Load Siting

A fast, vectorized Python module for ranking ~20,000+ power-system nodes to identify optimal locations for large electric loads (data centers, electrolyzers, industrial facilities, etc.).

## Features

- **Fast & Vectorized**: All operations use pandas/numpy for maximum performance on large datasets
- **Multi-Factor Scoring**: Considers cost, land availability, emissions, policy support, queue pressure, and price variability
- **Load-Specific Weighting**: Automatically adjusts scoring weights based on load type, size, and user preferences
- **Resource Scenarios**: Models impact of on-site generation/storage (solar, battery, firm generation)
- **Flexible Filtering**: Support for state-based or radial geographic filtering
- **Production Ready**: Clean API, comprehensive error handling, and detailed logging

## Quick Start

### Installation

```bash
# No installation required - just Python 3 with pandas and numpy
pip install pandas numpy
```

### Basic Usage

```python
from node_ranking_engine import rank_nodes, load_nodes_from_csv

# Load node data
nodes_df = load_nodes_from_csv("final_csv_v1_new.csv")

# Rank nodes for a data center in California
results = rank_nodes(
    nodes_df=nodes_df,
    load_type="data_center_always_on",
    load_size_mw=250,
    location_filter={"states": ["CA"]},
    emissions_preference=80,  # 0-100 scale
    resource_config="solar_battery",
    top_n=200
)

# View top results
print(results.head(10))
```

## API Reference

### Main Function: `rank_nodes()`

```python
def rank_nodes(
    nodes_df: pd.DataFrame,
    load_type: str,
    load_size_mw: float,
    location_filter: dict | None,
    emissions_preference: float,
    resource_config: str,
    top_n: int = 200
) -> pd.DataFrame
```

#### Parameters

- **nodes_df** (DataFrame): Input data with required columns (see Data Schema below)
- **load_type** (str): One of:
  - `"data_center_always_on"` - Constant load data center
  - `"data_center_flexible"` - Load-flexible data center
  - `"h2_electrolyzer_firm"` - Hydrogen electrolyzer with firm load
  - `"industrial_continuous"` - Continuous industrial process
  - `"industrial_flexible"` - Flexible industrial load
  - `"commercial_campus"` - Commercial facility
  
- **load_size_mw** (float): Load size in MW (affects queue/variability weights)

- **location_filter** (dict or None): Spatial filter options:
  - State filter: `{"states": ["CA", "NY", "TX"]}`
  - Radial filter: `{"lat": 37.77, "lon": -122.42, "radius_km": 200}`
  - No filter: `None`

- **emissions_preference** (float): User sensitivity to emissions (0-100)
  - 0 = Don't care about emissions
  - 100 = Extremely sensitive to emissions

- **resource_config** (str): On-site resource configuration:
  - `"none"` - Grid-only (no on-site resources)
  - `"solar"` - On-site solar generation
  - `"battery"` - On-site battery storage
  - `"solar_battery"` - Combined solar + battery
  - `"firm_gen"` - Firm on-site generation (natural gas, etc.)

- **top_n** (int): Number of top-ranked nodes to return (default: 200)

#### Returns

DataFrame with top_n ranked nodes including:
- All original input columns
- Component scores (0-1 scale, higher is better):
  - `cost_score` - Energy cost competitiveness
  - `land_score` - Land availability/affordability
  - `emissions_score` - Low emissions intensity
  - `policy_score` - Policy support and incentives
  - `queue_score` - Interconnection queue favorability
  - `price_variability_penalty_score` - Baseline price stability
  - `effective_price_variability_penalty_score` - Price stability with resources
- Composite scores and ranks:
  - `score_baseline` - Overall score without on-site resources
  - `rank_baseline` - Rank based on baseline score (1 = best)
  - `score_scenario` - Overall score with selected resource_config
  - `rank_scenario` - Rank based on scenario score (1 = best)

## Data Schema

### Required Columns

The input DataFrame must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `node` | str | Unique node identifier |
| `state` | str | Two-letter state code |
| `latitude` | float | Latitude coordinate |
| `longitude` | float | Longitude coordinate |
| `avg_lmp` | float | Average locational marginal price ($/MWh) |
| `avg_price_per_acre` | float | Average land price ($/acre) |
| `county_emissions_intensity_kg_per_mwh` | float | Grid emissions intensity (kg CO2/MWh) |

### Additional Columns (recommended)

| Column | Type | Description |
|--------|------|-------------|
| `iso` | str | ISO/RTO name |
| `county_state_pairs` | str | County, State for display |
| `avg_energy` | float | Energy component of LMP |
| `avg_congestion` | float | Congestion component of LMP |
| `avg_loses` | float | Loss component of LMP |
| `is_h2_hub_state` | float | H2 hub designation (0-1) |
| `state_dc_incentive_level` | float | Data center incentive level (0-1) |
| `state_clean_energy_friendly` | float | Clean energy policy score (0-1) |
| `has_hosting_capacity_map` | float | Hosting capacity data available (0-1) |
| `policy_fit_electrolyzer` | float | Electrolyzer policy fit score (0-1) |
| `policy_fit_datacenter` | float | Data center policy fit score (0-1) |
| `queue_pending_mw` | float | Pending interconnection queue (MW) |
| `queue_advanced_share` | float | Share of queue in advanced stages (0-1) |
| `queue_renewable_storage_share` | float | Share of green projects in queue (0-1) |
| `queue_pressure_index` | float | Queue congestion index (0-1) |
| `price_variance_score` | float | Price volatility metric (higher = more volatile) |

## How It Works

### 1. Component Score Calculation

The engine computes six normalized component scores (all 0-1, higher is better):

1. **Cost Score**: Based on LMP (lower LMP → higher score)
2. **Land Score**: Based on land price (lower price → higher score)
3. **Emissions Score**: Based on grid emissions intensity (lower emissions → higher score)
4. **Policy Score**: Based on policy support for the load type
5. **Queue Score**: Based on interconnection queue health
6. **Variability Score**: Based on price variability (lower variability → higher score)

### 2. Dynamic Weighting

Weights are adjusted based on:

- **Load Type**: Different priorities for data centers vs. electrolyzers vs. industrial
- **Load Size**: Larger loads increase importance of queue capacity and variability management
- **Emissions Preference**: User slider directly scales emissions weight

Example weights for data center (250 MW, 80% emissions preference):
- Cost: 19.6%
- Land: 18.5%
- Emissions: 20.4%
- Queue: 15.2%
- Policy: 13.1%
- Variability: 13.2%

### 3. Resource Configuration Impact

On-site resources reduce exposure to price variability via Variability Adjustment Factor (VAF):

| Resource Config | VAF | Effect |
|----------------|-----|--------|
| None (grid-only) | 1.0 | No reduction |
| Solar | 0.7 | 30% reduction |
| Battery | 0.6 | 40% reduction |
| Solar + Battery | 0.4 | 60% reduction |
| Firm Generation | 0.25 | 75% reduction |

### 4. Normalization

All raw metrics are normalized using robust min-max scaling with quantile clipping (5th-95th percentile) to handle outliers gracefully.

### 5. Final Scoring

Two composite scores are computed:

- **Baseline Score**: Weighted sum using grid-only variability
- **Scenario Score**: Weighted sum using effective variability (with resource VAF applied)

Nodes are ranked by scenario score, but baseline score is retained for comparison.

## Examples

### Example 1: Data Center in California

```python
results = rank_nodes(
    nodes_df=nodes_df,
    load_type="data_center_always_on",
    load_size_mw=250,
    location_filter={"states": ["CA"]},
    emissions_preference=80,
    resource_config="solar_battery",
    top_n=10
)

# Top node: TEPC_HACKBERRY230-APND in Madera, CA
# - Scenario Score: 0.837
# - LMP: $0.00/MWh
# - Emissions Score: 0.979
# - Cost Score: 1.000
```

### Example 2: H2 Electrolyzer (National Search)

```python
results = rank_nodes(
    nodes_df=nodes_df,
    load_type="h2_electrolyzer_firm",
    load_size_mw=100,
    location_filter=None,  # No filter
    emissions_preference=90,
    resource_config="none",
    top_n=10
)

# Top node: WEST_BPA_NODE_244 in WA
# - Scenario Score: 0.977
# - Emissions Score: 1.000
# - Queue Score: 0.992
# - Policy Score: 1.000
```

### Example 3: Radial Search (Near San Francisco)

```python
results = rank_nodes(
    nodes_df=nodes_df,
    load_type="industrial_continuous",
    load_size_mw=75,
    location_filter={"lat": 37.77, "lon": -122.42, "radius_km": 200},
    emissions_preference=50,
    resource_config="battery",
    top_n=15
)
```

### Example 4: Scenario Comparison

```python
# Compare different resource configurations
configs = ["none", "solar", "battery", "solar_battery", "firm_gen"]

for config in configs:
    results = rank_nodes(
        nodes_df=nodes_df,
        load_type="data_center_flexible",
        load_size_mw=150,
        location_filter={"states": ["TX"]},
        emissions_preference=40,
        resource_config=config,
        top_n=10
    )
    print(f"{config}: {results.iloc[0]['node']} (score: {results.iloc[0]['score_scenario']:.3f})")
```

## Demonstrations

Run comprehensive demonstrations:

```bash
# Run all demos
python demo_ranking.py

# Run specific demo
python -c "from demo_ranking import demo_1_california_datacenter; demo_1_california_datacenter()"
```

Available demos:
1. **demo_1_california_datacenter**: Large always-on data center in CA
2. **demo_2_flexible_electrolyzer**: National H2 electrolyzer search
3. **demo_3_radial_search**: Industrial load near Bay Area
4. **demo_4_scenario_comparison**: Compare resource configurations
5. **demo_5_weight_comparison**: Analyze weights across load types
6. **demo_6_emissions_sensitivity**: Test emissions preference impacts

## Performance

- **Dataset Size**: Handles 20,000+ nodes
- **Processing Time**: ~1-2 seconds for full ranking (on typical hardware)
- **Vectorization**: 100% vectorized using pandas/numpy (no Python loops)
- **Memory**: Efficient in-memory processing

## Architecture

```
node_ranking_engine.py
├── Normalization Helpers
│   ├── robust_min_max()      # Quantile-based normalization
│   └── invert_score()         # Convert "lower is better" to "higher is better"
│
├── Spatial Filtering
│   ├── haversine_distance_vectorized()
│   └── apply_spatial_filter()
│
├── Data Validation
│   └── validate_and_clean_data()
│
├── Component Scores
│   ├── compute_cost_score()
│   ├── compute_land_score()
│   ├── compute_emissions_score()
│   ├── compute_policy_score()
│   ├── compute_queue_score()
│   └── compute_variability_scores()
│
├── Weight Adjustment
│   ├── get_load_type_multipliers()
│   ├── get_size_multipliers()
│   └── compute_final_weights()
│
└── Main Entry Point
    └── rank_nodes()            # Public API
```

## Integration Patterns

### Web API

See `api_server.py` for a Flask-based REST API wrapper.

### Batch Processing

```python
# Process multiple scenarios
scenarios = [
    {"load_type": "data_center_always_on", "size": 250, "states": ["CA"]},
    {"load_type": "h2_electrolyzer_firm", "size": 100, "states": None},
]

results = {}
for scenario in scenarios:
    results[scenario["load_type"]] = rank_nodes(
        nodes_df=nodes_df,
        load_type=scenario["load_type"],
        load_size_mw=scenario["size"],
        location_filter={"states": scenario["states"]} if scenario["states"] else None,
        emissions_preference=50,
        resource_config="solar_battery",
        top_n=100
    )
```

### Interactive Dashboard

The engine is designed to power interactive UIs:
- User adjusts sliders → weights update → ranking refreshes
- User clicks map → radial filter applied → results update
- User selects resource → VAF applied → scores recalculated

## Best Practices

1. **Cache Node Data**: Load CSV once and reuse for multiple rankings
2. **Pre-filter When Possible**: Use location_filter to reduce computation
3. **Adjust top_n**: Return only what you need for better performance
4. **Validate Inputs**: Check load_type and resource_config values
5. **Handle Missing Data**: Engine imputes with median, but cleaner data = better results

## Troubleshooting

### "No nodes remain after filtering"
- Check location_filter parameters
- Verify state codes are correct (e.g., "CA" not "California")
- Ensure radius_km is reasonable for radial filter

### Unexpected Rankings
- Review component scores for top nodes
- Check weight distribution with `compute_final_weights()`
- Verify emissions_preference is on 0-100 scale

### Performance Issues
- Reduce dataset size with pre-filtering
- Ensure numeric columns are properly typed
- Consider using top_n < 1000 for very large datasets

## License

Proprietary - Internal Use Only

## Contact

For questions or support, contact the energy systems modeling team.

