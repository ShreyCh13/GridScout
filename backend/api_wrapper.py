"""
API Wrapper for Frontend JSON Input Format

Handles the exact JSON format from your frontend and translates it to the
node_ranking_engine format.

Usage:
    from api_wrapper import rank_nodes_from_frontend_json
    
    results = rank_nodes_from_frontend_json(frontend_json)
"""

from node_ranking_engine import rank_nodes, load_nodes_from_csv
import pandas as pd
from typing import Dict, List, Any


# State name to code mapping
STATE_NAME_TO_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
}


def map_load_type(type_str: str, sub_type: str = "") -> str:
    """
    Maps frontend load type to node_ranking_engine load_type.
    
    Args:
        type_str: Frontend type (e.g., "commercial", "industrial", "datacenter")
        sub_type: Frontend subType (e.g., "continuous_process", "flexible")
    
    Returns:
        load_type for node_ranking_engine
    """
    type_lower = type_str.lower()
    sub_lower = sub_type.lower() if sub_type else ""
    
    # Data center mappings
    if "data" in type_lower or "datacenter" in type_lower:
        if "flexible" in sub_lower or "interruptible" in sub_lower:
            return "data_center_flexible"
        else:
            return "data_center_always_on"
    
    # Industrial mappings
    elif "industrial" in type_lower:
        if "continuous" in sub_lower or "process" in sub_lower or "24/7" in sub_lower:
            return "industrial_continuous"
        elif "flexible" in sub_lower or "interruptible" in sub_lower:
            return "industrial_flexible"
        else:
            return "industrial_continuous"  # Default
    
    # H2/Electrolyzer mappings
    elif "h2" in type_lower or "hydrogen" in type_lower or "electrolyzer" in type_lower:
        return "h2_electrolyzer_firm"
    
    # Commercial mappings
    elif "commercial" in type_lower or "office" in type_lower or "campus" in type_lower:
        return "commercial_campus"
    
    # Default to commercial campus
    else:
        return "commercial_campus"


def map_resource_config(on_site_generation: str, configuration_type: str) -> str:
    """
    Maps frontend resource configuration to node_ranking_engine resource_config.
    
    Args:
        on_site_generation: "yes" or "no"
        configuration_type: "solar", "battery", "solar_battery", "firm_gen", etc.
    
    Returns:
        resource_config for node_ranking_engine
    """
    if on_site_generation.lower() != "yes":
        return "none"
    
    config_lower = configuration_type.lower() if configuration_type else ""
    
    if "solar" in config_lower and "battery" in config_lower:
        return "solar_battery"
    elif "solar" in config_lower:
        return "solar"
    elif "battery" in config_lower or "storage" in config_lower:
        return "battery"
    elif "firm" in config_lower or "gas" in config_lower or "gen" in config_lower:
        return "firm_gen"
    else:
        return "none"


def convert_state_names_to_codes(state_names: List[str]) -> List[str]:
    """
    Converts full state names to two-letter codes.
    
    Args:
        state_names: List of full state names (e.g., ["Wisconsin", "Nebraska"])
    
    Returns:
        List of state codes (e.g., ["WI", "NE"])
    """
    codes = []
    for name in state_names:
        # Try exact match
        if name in STATE_NAME_TO_CODE:
            codes.append(STATE_NAME_TO_CODE[name])
        # Try case-insensitive match
        else:
            for full_name, code in STATE_NAME_TO_CODE.items():
                if full_name.lower() == name.lower():
                    codes.append(code)
                    break
            else:
                # If it's already a code, use it
                if len(name) == 2:
                    codes.append(name.upper())
                else:
                    print(f"Warning: Unknown state name '{name}', skipping")
    
    return codes


def parse_frontend_json(frontend_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts frontend JSON format to node_ranking_engine parameters.
    
    Args:
        frontend_json: JSON from frontend with loadConfig and location
    
    Returns:
        Dictionary of parameters for rank_nodes()
    """
    load_config = frontend_json.get("loadConfig", {})
    location = frontend_json.get("location", {})
    
    # Map load type
    load_type = map_load_type(
        load_config.get("type", "commercial"),
        load_config.get("subType", "")
    )
    
    # Get load size
    load_size_mw = float(load_config.get("sizeMW", 100))
    
    # Get emissions preference
    emissions_preference = float(load_config.get("carbonEmissions", 50))
    
    # Map resource config
    resource_config = map_resource_config(
        load_config.get("onSiteGeneration", "no"),
        load_config.get("configurationType", "")
    )
    
    # Parse location filter
    location_filter = None
    location_mode = location.get("mode", "states")
    
    if location_mode == "states":
        state_names = location.get("selectedStates", [])
        if state_names:
            state_codes = convert_state_names_to_codes(state_names)
            if state_codes:
                location_filter = {"states": state_codes}
    
    elif location_mode == "points":
        # For points mode, we'll handle multiple points separately
        # This function returns the first point; caller should handle multiple
        selected_points = location.get("selectedPoints", [])
        if selected_points and len(selected_points) > 0:
            first_point = selected_points[0]
            location_filter = {
                "lat": float(first_point["lat"]),
                "lon": float(first_point["lng"]),
                "radius_km": 100  # 100km radius as specified
            }
    
    return {
        "load_type": load_type,
        "load_size_mw": load_size_mw,
        "emissions_preference": emissions_preference,
        "resource_config": resource_config,
        "location_filter": location_filter
    }


def rank_nodes_from_frontend_json(
    frontend_json: Dict[str, Any],
    nodes_df: pd.DataFrame = None,
    top_n: int = 200
) -> pd.DataFrame:
    """
    Ranks nodes using frontend JSON format.
    
    Handles both states and points modes. For points mode with multiple points,
    runs ranking for each point and combines results.
    
    Args:
        frontend_json: JSON from frontend with loadConfig and location
        nodes_df: Pre-loaded node DataFrame (will load if None)
        top_n: Number of top results per point (default 200)
    
    Returns:
        DataFrame with ranked results
    """
    # Load data if not provided
    if nodes_df is None:
        nodes_df = load_nodes_from_csv("final_csv_v1_new.csv")
    
    location = frontend_json.get("location", {})
    location_mode = location.get("mode", "states")
    
    # Parse base parameters
    params = parse_frontend_json(frontend_json)
    
    # Handle states mode (single ranking)
    if location_mode == "states":
        return rank_nodes(
            nodes_df=nodes_df,
            load_type=params["load_type"],
            load_size_mw=params["load_size_mw"],
            location_filter=params["location_filter"],
            emissions_preference=params["emissions_preference"],
            resource_config=params["resource_config"],
            top_n=top_n
        )
    
    # Handle points mode (multiple rankings, one per point)
    elif location_mode == "points":
        selected_points = location.get("selectedPoints", [])
        
        if not selected_points:
            # No points selected, return empty
            return pd.DataFrame()
        
        all_results = []
        
        for i, point in enumerate(selected_points):
            print(f"\nRanking for point {i+1}/{len(selected_points)}: {point.get('id', 'unknown')}")
            print(f"  Location: ({point['lat']:.4f}, {point['lng']:.4f})")
            
            # Create location filter for this point
            point_location_filter = {
                "lat": float(point["lat"]),
                "lon": float(point["lng"]),
                "radius_km": 100
            }
            
            # Run ranking
            results = rank_nodes(
                nodes_df=nodes_df,
                load_type=params["load_type"],
                load_size_mw=params["load_size_mw"],
                location_filter=point_location_filter,
                emissions_preference=params["emissions_preference"],
                resource_config=params["resource_config"],
                top_n=top_n
            )
            
            # Add point ID to results for tracking
            if len(results) > 0:
                results['search_point_id'] = point.get('id', f'point_{i+1}')
                results['search_point_lat'] = point['lat']
                results['search_point_lng'] = point['lng']
                all_results.append(results)
        
        # Combine results from all points
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            
            # Remove duplicates (same node might appear in multiple point searches)
            # Keep the one with the best score
            combined = combined.sort_values('score_scenario', ascending=False)
            combined = combined.drop_duplicates(subset=['node'], keep='first')
            
            # Re-rank the combined results
            combined['rank_scenario'] = combined['score_scenario'].rank(
                method='min', ascending=False
            ).astype(int)
            combined = combined.sort_values('rank_scenario')
            
            # Limit to top_n overall
            combined = combined.head(top_n)
            
            return combined
        else:
            return pd.DataFrame()
    
    else:
        raise ValueError(f"Unknown location mode: {location_mode}")


def format_response_for_frontend(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Formats ranking results for frontend consumption.
    
    Args:
        results: DataFrame from rank_nodes()
    
    Returns:
        JSON-serializable dictionary
    """
    if len(results) == 0:
        return {
            "success": False,
            "message": "No nodes found matching criteria",
            "results": []
        }
    
    # Convert to list of dicts
    nodes = []
    for _, row in results.iterrows():
        node_data = {
            "node": row["node"],
            "location": {
                "state": row["state"],
                "county": row.get("county_state_pairs", ""),
                "latitude": round(float(row["latitude"]), 6),
                "longitude": round(float(row["longitude"]), 6),
                "iso": row.get("iso", "")
            },
            "scores": {
                "overall": round(float(row["score_scenario"]), 4),
                "rank": int(row["rank_scenario"]),
                "components": {
                    "cost": round(float(row["cost_score"]), 3),
                    "land": round(float(row["land_score"]), 3),
                    "emissions": round(float(row["emissions_score"]), 3),
                    "policy": round(float(row["policy_score"]), 3),
                    "queue": round(float(row["queue_score"]), 3),
                    "variability": round(float(row["effective_price_variability_penalty_score"]), 3)
                }
            },
            "metrics": {
                "lmp": round(float(row["avg_lmp"]), 2),
                "landPricePerAcre": round(float(row["avg_price_per_acre"]), 0),
                "emissionsIntensity": round(float(row["county_emissions_intensity_kg_per_mwh"]), 1),
                "queuePendingMW": round(float(row.get("queue_pending_mw", 0)), 1),
                # New physical and cost/emissions metrics
                "annualMWh": round(float(row.get("annual_mwh", 0.0)), 1),
                "effectivePricePerMWh": round(float(row.get("effective_price_per_mwh", 0.0)), 2),
                "annualEnergyCost": round(float(row.get("annual_energy_cost", 0.0)), 0),
                "annualEmissionsTonnes": round(float(row.get("annual_emissions_tonnes", 0.0)), 1),
                "landAcres": round(float(row.get("land_acres", 0.0)), 1),
                "landCost": round(float(row.get("land_cost", 0.0)), 0),
            }
        }
        
        # Add search point info if available
        if "search_point_id" in row:
            node_data["searchPoint"] = {
                "id": row["search_point_id"],
                "lat": float(row["search_point_lat"]),
                "lng": float(row["search_point_lng"])
            }
        
        nodes.append(node_data)
    
    return {
        "success": True,
        "totalResults": len(nodes),
        "results": nodes
    }


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("API WRAPPER TEST - Frontend JSON Format")
    print("="*80)
    
    # Test 1: States mode
    print("\n" + "="*80)
    print("TEST 1: States Mode (Wisconsin, Nebraska)")
    print("="*80)
    
    test_json_1 = {
        "loadConfig": {
            "type": "commercial",
            "subType": "",
            "sizeMW": 500,
            "carbonEmissions": 70,
            "onSiteGeneration": "yes",
            "configurationType": "battery"
        },
        "location": {
            "mode": "states",
            "selectedStates": ["Wisconsin", "Nebraska"],
            "selectedPoints": []
        }
    }
    
    print("\nInput JSON:")
    import json
    print(json.dumps(test_json_1, indent=2))
    
    # Parse and show translation
    params = parse_frontend_json(test_json_1)
    print("\nTranslated parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Run ranking
    print("\nRunning ranking...")
    results_1 = rank_nodes_from_frontend_json(test_json_1, top_n=5)
    
    print(f"\nResults: {len(results_1)} nodes")
    if len(results_1) > 0:
        print("\nTop 3 nodes:")
        for idx, row in results_1.head(3).iterrows():
            print(f"  {int(row['rank_scenario'])}. {row['node']} ({row['state']}) - Score: {row['score_scenario']:.4f}")
    
    # Test 2: Points mode
    print("\n" + "="*80)
    print("TEST 2: Points Mode (3 points in Colorado area)")
    print("="*80)
    
    test_json_2 = {
        "loadConfig": {
            "type": "industrial",
            "subType": "continuous_process",
            "sizeMW": 500,
            "carbonEmissions": 30,
            "onSiteGeneration": "yes",
            "configurationType": "battery"
        },
        "location": {
            "mode": "points",
            "selectedStates": [],
            "selectedPoints": [
                {
                    "id": "point-1763264891452",
                    "lng": -108.72264303766178,
                    "lat": 39.18291624575372
                },
                {
                    "id": "point-1763264892236",
                    "lng": -107.87709391027136,
                    "lat": 41.76655632620444
                },
                {
                    "id": "point-1763264892870",
                    "lng": -106.16089408048168,
                    "lat": 38.88393174385959
                }
            ]
        }
    }
    
    print("\nInput JSON:")
    print(json.dumps(test_json_2, indent=2))
    
    # Run ranking
    print("\nRunning ranking for all points...")
    results_2 = rank_nodes_from_frontend_json(test_json_2, top_n=10)
    
    print(f"\nCombined results: {len(results_2)} nodes")
    if len(results_2) > 0:
        print("\nTop 5 nodes (combined from all points):")
        for idx, row in results_2.head(5).iterrows():
            print(f"  {int(row['rank_scenario'])}. {row['node']} ({row['state']}) - "
                  f"Score: {row['score_scenario']:.4f} - "
                  f"From: {row.get('search_point_id', 'N/A')}")
    
    # Test 3: Format for frontend
    print("\n" + "="*80)
    print("TEST 3: Format Response for Frontend")
    print("="*80)
    
    if len(results_1) > 0:
        frontend_response = format_response_for_frontend(results_1.head(3))
        print("\nFormatted response (first 3 results):")
        print(json.dumps(frontend_response, indent=2))
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)

