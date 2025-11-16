"""
REST API Server for Node Ranking Engine

Provides a web service interface to the node ranking engine.
Supports JSON-based requests and responses for easy frontend integration.

Usage:
    python api_server.py

Then make POST requests to:
    http://localhost:5000/api/rank
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

from node_ranking_engine import (
    rank_nodes,
    load_nodes_from_csv,
    compute_final_weights
)

from api_wrapper import (
    rank_nodes_from_frontend_json,
    format_response_for_frontend
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variable to cache node data
NODES_DF = None
DATA_FILE = "final_csv_v1_new.csv"


def load_data():
    """Load node data into memory on startup."""
    global NODES_DF
    if NODES_DF is None:
        print(f"Loading node data from {DATA_FILE}...")
        NODES_DF = load_nodes_from_csv(DATA_FILE)
        print(f"Loaded {len(NODES_DF)} nodes")
    return NODES_DF


def validate_request(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validates API request parameters.
    
    Returns:
        (is_valid, error_message)
    """
    # Required fields
    required = ["load_type", "load_size_mw", "emissions_preference", "resource_config"]
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate load_type
    valid_load_types = [
        "data_center_always_on", "data_center_flexible",
        "h2_electrolyzer_firm", "industrial_continuous",
        "industrial_flexible", "commercial_campus"
    ]
    if data["load_type"] not in valid_load_types:
        return False, f"Invalid load_type. Must be one of: {', '.join(valid_load_types)}"
    
    # Validate load_size_mw
    try:
        load_size = float(data["load_size_mw"])
        if load_size <= 0 or load_size > 10000:
            return False, "load_size_mw must be between 0 and 10000"
    except (ValueError, TypeError):
        return False, "load_size_mw must be a number"
    
    # Validate emissions_preference
    try:
        emissions_pref = float(data["emissions_preference"])
        if emissions_pref < 0 or emissions_pref > 100:
            return False, "emissions_preference must be between 0 and 100"
    except (ValueError, TypeError):
        return False, "emissions_preference must be a number"
    
    # Validate resource_config
    valid_resources = ["none", "solar", "battery", "solar_battery", "firm_gen"]
    if data["resource_config"] not in valid_resources:
        return False, f"Invalid resource_config. Must be one of: {', '.join(valid_resources)}"
    
    # Validate location_filter if provided
    if "location_filter" in data and data["location_filter"] is not None:
        loc_filter = data["location_filter"]
        
        if "states" in loc_filter:
            if not isinstance(loc_filter["states"], list):
                return False, "location_filter.states must be a list"
        elif all(k in loc_filter for k in ["lat", "lon", "radius_km"]):
            try:
                lat = float(loc_filter["lat"])
                lon = float(loc_filter["lon"])
                radius = float(loc_filter["radius_km"])
                if not (-90 <= lat <= 90):
                    return False, "latitude must be between -90 and 90"
                if not (-180 <= lon <= 180):
                    return False, "longitude must be between -180 and 180"
                if radius <= 0 or radius > 5000:
                    return False, "radius_km must be between 0 and 5000"
            except (ValueError, TypeError):
                return False, "Invalid radial filter parameters"
        else:
            return False, "location_filter must have 'states' or 'lat'/'lon'/'radius_km'"
    
    # Validate top_n if provided
    if "top_n" in data:
        try:
            top_n = int(data["top_n"])
            if top_n < 1 or top_n > 1000:
                return False, "top_n must be between 1 and 1000"
        except (ValueError, TypeError):
            return False, "top_n must be an integer"
    
    return True, ""


def format_results(df: pd.DataFrame) -> list:
    """
    Formats ranking results for JSON response.
    
    Converts DataFrame to list of dicts with clean formatting.
    """
    # Replace NaN with None for JSON serialization
    df = df.replace({np.nan: None})
    
    # Convert to list of dicts
    results = []
    for _, row in df.iterrows():
        results.append({
            # Node identification
            "node": row["node"],
            "state": row["state"],
            "iso": row.get("iso"),
            "county_state_pairs": row.get("county_state_pairs"),
            "latitude": round(row["latitude"], 6) if row["latitude"] is not None else None,
            "longitude": round(row["longitude"], 6) if row["longitude"] is not None else None,
            
            # Scores and ranks
            "score_baseline": round(row["score_baseline"], 4),
            "score_scenario": round(row["score_scenario"], 4),
            "rank_baseline": int(row["rank_baseline"]),
            "rank_scenario": int(row["rank_scenario"]),
            
            # Component scores
            "component_scores": {
                "cost": round(row["cost_score"], 3),
                "land": round(row["land_score"], 3),
                "emissions": round(row["emissions_score"], 3),
                "policy": round(row["policy_score"], 3),
                "queue": round(row["queue_score"], 3),
                "variability_baseline": round(row["price_variability_penalty_score"], 3),
                "variability_scenario": round(row["effective_price_variability_penalty_score"], 3),
            },
            
            # Raw metrics (for display/explanation)
            "raw_metrics": {
                "avg_lmp": round(row["avg_lmp"], 2) if row["avg_lmp"] is not None else None,
                "avg_price_per_acre": round(row["avg_price_per_acre"], 0) if row["avg_price_per_acre"] is not None else None,
                "emissions_intensity": round(row["county_emissions_intensity_kg_per_mwh"], 1) if row["county_emissions_intensity_kg_per_mwh"] is not None else None,
                "queue_pending_mw": round(row["queue_pending_mw"], 1) if row["queue_pending_mw"] is not None else None,
            }
        })
    
    return results


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """Root endpoint - API info."""
    return jsonify({
        "service": "Node Ranking Engine API",
        "version": "1.0",
        "endpoints": {
            "/api/rank": "POST - Rank nodes for load siting",
            "/api/weights": "POST - Get weight breakdown for parameters",
            "/api/health": "GET - Health check"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        nodes_df = load_data()
        return jsonify({
            "status": "healthy",
            "nodes_loaded": len(nodes_df),
            "data_file": DATA_FILE
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/api/rank", methods=["POST"])
def rank():
    """
    Main ranking endpoint.
    
    Request body:
    {
        "load_type": "data_center_always_on",
        "load_size_mw": 250,
        "location_filter": {"states": ["CA", "NY"]} or {"lat": 37.77, "lon": -122.42, "radius_km": 200} or null,
        "emissions_preference": 80,
        "resource_config": "solar_battery",
        "top_n": 200  // optional, default 200
    }
    
    Response:
    {
        "success": true,
        "num_results": 200,
        "weights": {...},
        "results": [...]
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        # Validate request
        is_valid, error_msg = validate_request(data)
        if not is_valid:
            return jsonify({"success": False, "error": error_msg}), 400
        
        # Load data
        nodes_df = load_data()
        
        # Extract parameters
        load_type = data["load_type"]
        load_size_mw = float(data["load_size_mw"])
        emissions_preference = float(data["emissions_preference"])
        resource_config = data["resource_config"]
        location_filter = data.get("location_filter")
        top_n = int(data.get("top_n", 200))
        
        # Compute weights for transparency
        weights = compute_final_weights(load_type, load_size_mw, emissions_preference)
        
        # Run ranking
        results_df = rank_nodes(
            nodes_df=nodes_df,
            load_type=load_type,
            load_size_mw=load_size_mw,
            location_filter=location_filter,
            emissions_preference=emissions_preference,
            resource_config=resource_config,
            top_n=top_n
        )
        
        # Format results
        if len(results_df) == 0:
            return jsonify({
                "success": False,
                "error": "No nodes matched the specified criteria"
            }), 404
        
        results_list = format_results(results_df)
        
        # Build response
        response = {
            "success": True,
            "num_results": len(results_list),
            "parameters": {
                "load_type": load_type,
                "load_size_mw": load_size_mw,
                "emissions_preference": emissions_preference,
                "resource_config": resource_config,
                "location_filter": location_filter,
                "top_n": top_n
            },
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "results": results_list
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route("/api/weights", methods=["POST"])
def get_weights():
    """
    Returns weight breakdown for given parameters (without running full ranking).
    
    Request body:
    {
        "load_type": "data_center_always_on",
        "load_size_mw": 250,
        "emissions_preference": 80
    }
    
    Response:
    {
        "success": true,
        "weights": {...}
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        # Extract parameters
        load_type = data.get("load_type", "data_center_always_on")
        load_size_mw = float(data.get("load_size_mw", 100))
        emissions_preference = float(data.get("emissions_preference", 50))
        
        # Compute weights
        weights = compute_final_weights(load_type, load_size_mw, emissions_preference)
        
        return jsonify({
            "success": True,
            "parameters": {
                "load_type": load_type,
                "load_size_mw": load_size_mw,
                "emissions_preference": emissions_preference
            },
            "weights": {k: round(v, 4) for k, v in weights.items()}
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/submit", methods=["POST"])
def submit_ranking():
    """
    Frontend submission endpoint - Accepts smart-siting frontend JSON format.
    
    This endpoint is designed for the React frontend which sends data in a
    specific format with loadConfig and location objects.
    
    Request body:
    {
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
    
    Response:
    {
        "success": true,
        "totalResults": 10,
        "results": [...]
    }
    """
    try:
        frontend_json = request.get_json()
        
        if not frontend_json:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        print("\n" + "="*80)
        print("Received frontend submission:")
        print("="*80)
        import json
        print(json.dumps(frontend_json, indent=2))
        print("="*80 + "\n")
        
        # Load data
        nodes_df = load_data()
        
        # Use api_wrapper to handle frontend format and run ranking
        results = rank_nodes_from_frontend_json(
            frontend_json,
            nodes_df=nodes_df,
            top_n=200  # Return top 200 results
        )
        
        # Format response for frontend
        response = format_response_for_frontend(results)
        
        print(f"\n✅ Ranking complete: {response.get('totalResults', 0)} results\n")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file '{DATA_FILE}' not found!")
        print("Please ensure the CSV file is in the current directory.")
        exit(1)
    
    # Load data on startup
    print("=" * 80)
    print("NODE RANKING ENGINE - API SERVER")
    print("=" * 80)
    load_data()
    
    print("\nStarting Flask server...")
    print("API endpoints available at:")
    print("  - http://localhost:5001/api/rank (POST)")
    print("  - http://localhost:5001/api/weights (POST)")
    print("  - http://localhost:5001/api/health (GET)")
    print("\n" + "=" * 80)
    
    # Note: Install Flask and flask-cors first:
    # pip install flask flask-cors
    
    # When running locally we default to port 5001.
    # On cloud providers (Render, Railway, etc.) they usually provide a PORT
    # environment variable. Respect it if present so deployment "just works".
    port = int(os.environ.get("PORT", 5001))
    
    # Run server (debug off by default for production safety)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")

