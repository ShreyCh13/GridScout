import React, { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Map, { Source, Layer, Marker } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";

const MAPBOX_TOKEN = "pk.eyJ1IjoicG9ja2V0Y2hhcmdlIiwiYSI6ImNtaHpzeWsweDByNHAycW9meWlxcjJwbG4ifQ.rGEbpo57cirYtnvXfDz5tw"; // <-- INSERT YOUR REAL TOKEN

// Map of state FIPS codes and abbreviations to full names
const STATE_NAMES = {
  // FIPS codes (numeric strings)
  "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
  "08": "Colorado", "09": "Connecticut", "10": "Delaware", "12": "Florida", "13": "Georgia",
  "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
  "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
  "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi", "29": "Missouri",
  "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
  "35": "New Mexico", "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
  "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
  "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
  "51": "Virginia", "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
  "11": "District of Columbia",
  // State abbreviations (as fallback)
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas", CA: "California",
  CO: "Colorado", CT: "Connecticut", DE: "Delaware", FL: "Florida", GA: "Georgia",
  HI: "Hawaii", ID: "Idaho", IL: "Illinois", IN: "Indiana", IA: "Iowa",
  KS: "Kansas", KY: "Kentucky", LA: "Louisiana", ME: "Maine", MD: "Maryland",
  MA: "Massachusetts", MI: "Michigan", MN: "Minnesota", MS: "Mississippi", MO: "Missouri",
  MT: "Montana", NE: "Nebraska", NV: "Nevada", NH: "New Hampshire", NJ: "New Jersey",
  NM: "New Mexico", NY: "New York", NC: "North Carolina", ND: "North Dakota", OH: "Ohio",
  OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania", RI: "Rhode Island", SC: "South Carolina",
  SD: "South Dakota", TN: "Tennessee", TX: "Texas", UT: "Utah", VT: "Vermont",
  VA: "Virginia", WA: "Washington", WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming",
  DC: "District of Columbia"
};

// List of all state abbreviations (for "Select all" in location selection)
const ALL_STATE_ABBRS = Object.keys(STATE_NAMES).filter((key) => {
  // Keep only 2-letter non-numeric keys (e.g., "CA", "NY"), skip FIPS codes like "01"
  return key.length === 2 && isNaN(Number(key));
});

// Mock "nearby clean resources" for a chosen site
function mockNearbyResources(lng, lat) {
  return [
    {
      id: "wind-1",
      type: "Offshore wind",
      lng: lng - 1.2,
      lat: lat + 0.7,
      mw: 800,
    },
    {
      id: "solar-1",
      type: "Solar cluster",
      lng: lng + 0.8,
      lat: lat - 0.5,
      mw: 600,
    },
    {
      id: "bess-1",
      type: "BESS hub",
      lng: lng + 0.2,
      lat: lat + 0.4,
      mw: 400,
    },
  ];
}

// Helper: Calculate distance in km between two lat/lng points (Haversine formula)
function calculateDistance(lat1, lng1, lat2, lng2) {
  const R = 6371; // Earth's radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function App() {
  const [showHero, setShowHero] = useState(true);
  const [spinEnabled, setSpinEnabled] = useState(true);

  // Initial “near-horizon” view, roughly like your image
  const [viewState, setViewState] = useState({
    longitude: -30, // Atlantic / Europe–Africa side, not empty Pacific
    latitude: 15,
    zoom: 1.7, // closer in => more curvature
    bearing: 0,
    pitch: 78, // high pitch => horizon view
  });

  const [loadConfig, setLoadConfig] = useState({
    type: "data_centre",
    subType: "always_on",
    sizeMW: 500,
    carbonEmissions: 50,
    onSiteGeneration: "none",
    configurationType: "",
  });

  // Location selection state
  const [locationMode, setLocationMode] = useState("states"); // "states" or "points"
  const [selectedStates, setSelectedStates] = useState(new Set()); // No states selected by default
  const [selectedPoints, setSelectedPoints] = useState([]); // Array of {id, lng, lat}

  const [selectedSite, setSelectedSite] = useState(null);
  const [resources, setResources] = useState([]);
  const [rankingResults, setRankingResults] = useState(null); // Store backend ranking results
  const [expandedNodeId, setExpandedNodeId] = useState(null); // Track which node in dashboard is expanded

  // Slow Earth rotation while hero is visible (opposite direction)
  useEffect(() => {
    if (!showHero || !spinEnabled) return;

    let frameId;

    const spin = () => {
      setViewState((prev) => ({
        ...prev,
        // negative bearing change => rotates the other way
        bearing: (prev.bearing - 0.06 + 360) % 360,
      }));
      frameId = requestAnimationFrame(spin);
    };

    frameId = requestAnimationFrame(spin);
    return () => cancelAnimationFrame(frameId);
  }, [showHero, spinEnabled]);

  // Reset the app back to a fresh "ready for new input" state
  const handleClearAllResults = () => {
    // Reset load configuration to defaults
    setLoadConfig({
      type: "data_centre",
      subType: "always_on",
      sizeMW: 500,
      carbonEmissions: 50,
      onSiteGeneration: "none",
      configurationType: "",
    });

    // Reset location selection controls
    setLocationMode("states");
    setSelectedStates(new Set());
    setSelectedPoints([]);

    // Clear any selected node/results
    setSelectedSite(null);
    setRankingResults(null);
    setExpandedNodeId(null);

    // Return camera to the main US siting canvas
    setViewState((prev) => ({
      ...prev,
      longitude: -98,
      latitude: 38,
      zoom: 3.3,
      pitch: 20,
      bearing: 0,
    }));
  };

  const handleMapClick = (event) => {
    // Don't handle clicks while hero is showing
    if (showHero) return;

    // When ranking results are visible (output screen), disable selecting/deselecting states/points
    if (rankingResults && rankingResults.length > 0) return;

    const { lng, lat } = event.lngLat;

    // Handle location selection modes
    if (locationMode === "states") {
      // Check if user clicked on a state
      const features = event.target.queryRenderedFeatures(event.point, {
        layers: ["state-fills"],
      });

      if (features.length > 0) {
        // Try different property names that Mapbox might use
        const props = features[0].properties;
        const stateAbbr = props.STUSPS || props.STATE_ID || props.state_abbr || props.postal;

        console.log("Clicked state properties:", props); // Debug log

        if (stateAbbr) {
          setSelectedStates((prev) => {
            const newSet = new Set(prev);
            if (newSet.has(stateAbbr)) {
              newSet.delete(stateAbbr);
            } else {
              newSet.add(stateAbbr);
            }
            return newSet;
          });
        }
      }
      return; // Don't create site marker when in states mode
    } else if (locationMode === "points") {
      // Check if clicking inside an existing circle to remove it
      for (const point of selectedPoints) {
        const distance = calculateDistance(lat, lng, point.lat, point.lng);
        if (distance <= 100) {
          // Clicking within 100km radius removes the point
          setSelectedPoints((prev) => prev.filter((p) => p.id !== point.id));
          return;
        }
      }

      // Check if click is within US boundaries
      const features = event.target.queryRenderedFeatures(event.point, {
        layers: ["state-fills"],
      });

      if (features.length === 0) {
        // Click is outside US boundaries
        console.log("Point must be within US boundaries");
        return;
      }

      // Check if new point is at least 200km from all existing points
      const tooClose = selectedPoints.some((point) => {
        const distance = calculateDistance(lat, lng, point.lat, point.lng);
        return distance < 200;
      });

      if (tooClose) {
        // Could add a visual feedback here (toast/alert)
        console.log("Point too close to existing location (min 200km apart)");
        return;
      }

      // Add new point
      const newPoint = {
        id: `point-${Date.now()}`,
        lng,
        lat,
      };
      console.log("Point added - Latitude:", lat, "Longitude:", lng);
      setSelectedPoints((prev) => [...prev, newPoint]);
      return; // Don't create site marker when placing points
    }

    // Original site selection logic (only when not in location selection mode)
    const site = {
      lng,
      lat,
      name: "Proposed GW-scale load node",
      congestionRelief: "▲ 18% headroom on key tie-lines",
      emissionsImpact: "▼ 0.22 tCO₂/MWh versus BAU siting",
      reliabilityBoost: "N-1 secure with 2× 345-kV paths",
      narrative: `Anchored by ${loadConfig.sizeMW} MW ${labelLoadType(
        loadConfig.type
      )}`,
    };

    setSelectedSite(site);
    setResources(mockNearbyResources(lng, lat));

    setViewState((prev) => ({
      ...prev,
      longitude: lng,
      latitude: lat,
      zoom: 6,
      pitch: 45,
    }));
  };

  // Create GeoJSON for 100km radius circles around selected points
  const pointCirclesGeoJSON = useMemo(() => {
    const features = selectedPoints.map((point) => {
      // Create a circle polygon (approximated with 64 points)
      const radiusInDegrees = 100 / 111; // Rough conversion: 1 degree ≈ 111 km
      const points = 64;
      const coordinates = [[]];

      for (let i = 0; i <= points; i++) {
        const angle = (i * 360) / points;
        const dx = radiusInDegrees * Math.cos((angle * Math.PI) / 180);
        const dy = radiusInDegrees * Math.sin((angle * Math.PI) / 180);
        coordinates[0].push([point.lng + dx, point.lat + dy]);
      }

      return {
        type: "Feature",
        properties: { id: point.id },
        geometry: {
          type: "Polygon",
          coordinates,
        },
      };
    });

    return {
      type: "FeatureCollection",
      features,
    };
  }, [selectedPoints]);

  // Convert selectedStates Set to Array for Mapbox expression
  const selectedStatesArray = useMemo(() => Array.from(selectedStates), [selectedStates]);

  // Create a stable key for layer re-rendering
  const statesKey = useMemo(() => {
    return Array.from(selectedStates).sort().join('-');
  }, [selectedStates]);

  const gradientLegend = useMemo(
    () => [
      { color: "#22c55e", label: "High siting score" },
      { color: "#eab308", label: "Moderate" },
      { color: "#ef4444", label: "Avoid / constrained" },
    ],
    []
  );

  // Generic camera animation helper (from -> to, smoothstep)
  const animateCamera = (fromState, toState, duration, onDone) => {
    const start = performance.now();

    const frame = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const ease = t * t * (3 - 2 * t);

      setViewState((prev) => ({
        ...prev,
        longitude:
          fromState.longitude +
          (toState.longitude - fromState.longitude) * ease,
        latitude:
          fromState.latitude +
          (toState.latitude - fromState.latitude) * ease,
        zoom: fromState.zoom + (toState.zoom - fromState.zoom) * ease,
        pitch: fromState.pitch + (toState.pitch - fromState.pitch) * ease,
        bearing:
          fromState.bearing +
          (toState.bearing - fromState.bearing) * ease,
      }));

      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        setViewState((prev) => ({ ...prev, ...toState }));
        if (onDone) onDone();
      }
    };

    requestAnimationFrame(frame);
  };

  // Hero → slightly zoom out to full Earth → then zoom into US
  const handleHeroStart = () => {
    // stop the continuous spin so it doesn't fight the animation
    setSpinEnabled(false);

    const current = { ...viewState };

    const fullEarthState = {
      longitude: -100, // center roughly on Americas
      latitude: 20,
      zoom: 1.3, // a bit more zoomed out than initial
      pitch: 45, // less horizon, more full globe
      bearing: 0,
    };

    const usState = {
      longitude: -98,
      latitude: 38,
      zoom: 3.3,
      pitch: 20,
      bearing: 0,
    };

    // Stage 1: horizon view -> full Earth
    animateCamera(current, fullEarthState, 900, () => {
      // Stage 2: full Earth -> US siting canvas
      animateCamera(fullEarthState, usState, 1400, () => {
        setShowHero(false);
      });
    });
  };

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-50 overflow-hidden text-[13px] md:text-[14px]">
      {/* Top nav */}
      <div className="absolute top-0 left-0 right-0 z-30 flex items-center justify-between px-6 py-4 bg-gradient-to-b from-slate-950/90 to-transparent">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-emerald-400/20 border border-emerald-300/60 flex items-center justify-center text-sm font-bold">
            GS
          </div>
            <div>
              <div className="text-lg md:text-xl font-semibold tracking-[0.35em] uppercase">
                GridScout
              </div>
              <div className="text-sm text-slate-400">
                Place big loads where they help the grid
              </div>
            </div>
        </div>
        <div className="hidden md:flex items-center gap-4 text-xs text-slate-400">
          <span>MITEC Hackathon 2025</span>
          <span className="h-1 w-1 rounded-full bg-emerald-400" />
          <span>Prototype — siting engine v0.1</span>
        </div>
      </div>

      <div className="flex h-full pt-12">
        {/* Left control panel */}
        <div className="relative z-10 w-full max-w-md border-r border-slate-800 bg-slate-950/95 backdrop-blur-md px-5 py-6 flex flex-col gap-4 overflow-y-auto">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <PanelHeader />
              <LoadForm
                loadConfig={loadConfig}
                setLoadConfig={setLoadConfig}
              />
            </div>
            <button
              type="button"
              onClick={handleClearAllResults}
              className="mt-1 inline-flex items-center whitespace-nowrap rounded-full border border-slate-700 px-3 py-1 text-[10px] font-medium text-slate-300 hover:border-slate-400 hover:bg-slate-900 transition"
            >
              Reset
            </button>
          </div>
          <LocationSelectionForm
            locationMode={locationMode}
            setLocationMode={setLocationMode}
            selectedStates={selectedStates}
            setSelectedStates={setSelectedStates}
          />
          <SubmitButton
            loadConfig={loadConfig}
            locationMode={locationMode}
            selectedStates={selectedStates}
            selectedPoints={selectedPoints}
            rankingResults={rankingResults}
            setRankingResults={setRankingResults}
            setViewState={setViewState}
            setSelectedSite={setSelectedSite}
            setSelectedStates={setSelectedStates}
            setSelectedPoints={setSelectedPoints}
          />
          <Legend gradientLegend={gradientLegend} />
        </div>

        {/* Map area */}
        <div className="relative flex-1">
          <Map
            mapboxAccessToken={MAPBOX_TOKEN}
            mapStyle={
              showHero
                ? "mapbox://styles/mapbox/satellite-v9"
                : "mapbox://styles/mapbox/dark-v11"
            }
            {...viewState}
            onMove={(evt) => setViewState(evt.viewState)}
            onClick={handleMapClick}
            minZoom={1}
            maxZoom={10}
            projection="globe"
            style={{ width: "100%", height: "100%" }}
          >
            {/* US States layer - always present for boundary checking */}
            {!showHero && (
              <Source
                id="state-boundaries"
                type="vector"
                url="mapbox://mapbox.us_census_states_2015"
              >
                <Layer
                  key={`state-fills-${statesKey}`}
                  id="state-fills"
                  type="fill"
                  source-layer="states"
                  paint={{
                    "fill-color": "#10b981", // emerald for all states
                    "fill-opacity": [
                      "case",
                      ["in", ["coalesce", ["get", "STUSPS"], ["get", "STATE_ID"], ""], ["literal", selectedStatesArray]],
                      locationMode === "states" ? 0.4 : 0, // visible when selected in states mode
                      0, // transparent when deselected (blends with map)
                    ],
                  }}
                />
                <Layer
                  key={`state-borders-${statesKey}`}
                  id="state-borders"
                  type="line"
                  source-layer="states"
                  paint={{
                    "line-color": "#10b981", // emerald for all states
                    "line-width": 2,
                    "line-opacity": [
                      "case",
                      ["in", ["coalesce", ["get", "STUSPS"], ["get", "STATE_ID"], ""], ["literal", selectedStatesArray]],
                      locationMode === "states" ? 0.8 : 0, // visible when selected in states mode
                      0, // transparent when deselected (blends with map)
                    ],
                  }}
                />
              </Source>
            )}

            {/* Point circles for specific locations */}
            {!showHero && locationMode === "points" && selectedPoints.length > 0 && (
              <Source id="point-circles-src" type="geojson" data={pointCirclesGeoJSON}>
                <Layer
                  id="point-circles-fill"
                  type="fill"
                  paint={{
                    "fill-color": "#10b981",
                    "fill-opacity": 0.2,
                  }}
                />
                <Layer
                  id="point-circles-outline"
                  type="line"
                  paint={{
                    "line-color": "#10b981",
                    "line-width": 2,
                    "line-opacity": 0.8,
                  }}
                />
              </Source>
            )}

            {/* Point markers (pins) for specific locations */}
            {!showHero && locationMode === "points" && selectedPoints.map((point) => (
              <Marker
                key={point.id}
                longitude={point.lng}
                latitude={point.lat}
                anchor="bottom"
              >
                <div className="relative">
                  <div className="h-4 w-4 rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50 border-2 border-slate-950" />
                  <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-ping" />
                </div>
              </Marker>
            ))}

            {/* Ranking results markers with gradient coloring */}
            {rankingResults && rankingResults.map((node, index) => {
              // Use rank position (not score) for more visible gradient across results
              const totalResults = rankingResults.length;
              const normalizedRank = index / Math.max(totalResults - 1, 1); // 0 (best) to 1 (worst)

              const getGradientColor = (rankPosition) => {
                // Strong visual gradient: bright green → yellow → orange → red
                if (rankPosition < 0.1) {
                  // Top 10%: Bright green
                  return "#10b981";
                } else if (rankPosition < 0.25) {
                  // Top 25%: Green-yellow
                  return "#84cc16";
                } else if (rankPosition < 0.5) {
                  // Top 50%: Yellow
                  return "#eab308";
                } else if (rankPosition < 0.75) {
                  // Top 75%: Orange
                  return "#f97316";
                } else {
                  // Bottom 25%: Red-orange
                  return "#ef4444";
                }
              };

              const markerColor = getGradientColor(normalizedRank);
              const markerSize =
                index === 0 ? 6 : index < 3 ? 5.5 : index < 10 ? 5 : index < 30 ? 4 : 3.5;

              const generateExplanation = (node, loadConfig) => {
                const scores = node.scores.components;
                const metrics = node.metrics;

                // Find top 2 strongest components
                const components = [
                  { name: "cost", score: scores.cost },
                  { name: "emissions", score: scores.emissions },
                  { name: "policy", score: scores.policy },
                  { name: "queue", score: scores.queue },
                  { name: "land", score: scores.land },
                  { name: "variability", score: scores.variability },
                ];

                const topComponents = components
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 2);

                // Build natural explanation
                let parts = [];

                // Opening
                parts.push(
                  `Ranked #${node.scores.rank} with ${(node.scores.overall * 100).toFixed(
                    0
                  )}% overall score.`
                );

                // Add 1-2 key highlights
                const highlights = [];
                topComponents.forEach((c) => {
                  if (c.score > 0.8) {
                    if (c.name === "cost" && metrics.lmp < 35) {
                      highlights.push(`Low energy costs ($${metrics.lmp.toFixed(2)}/MWh)`);
                    } else if (c.name === "emissions" && metrics.emissionsIntensity < 100) {
                      highlights.push(
                        `Clean grid (${metrics.emissionsIntensity.toFixed(0)} kg CO₂/MWh)`
                      );
                    } else if (c.name === "queue" && metrics.queuePendingMW < 500) {
                      highlights.push(`Strong interconnection position`);
                    } else if (c.name === "policy") {
                      highlights.push(`Favorable regulatory environment`);
                    } else if (c.name === "variability") {
                      highlights.push(`Stable pricing`);
                    }
                  }
                });

                if (highlights.length > 0) {
                  parts.push(highlights.slice(0, 2).join(" and ") + ".");
                }

                // Location context
                parts.push(
                  `Located in ${node.location.county}, ${node.location.state} (${node.location.iso} market).`
                );

                return parts.join(" ");
              };

              const selectNodeForDetails = (node) => {
                setSelectedSite({
                  lng: node.location.longitude,
                  lat: node.location.latitude,
                  name: node.node,
                  nodeName: node.node,
                  rank: node.scores.rank,
                  score: node.scores.overall,
                  state: node.location.state,
                  county: node.location.county || node.location.state,
                  iso: node.location.iso,
                  lmp: node.metrics.lmp,
                  emissions: node.metrics.emissionsIntensity,
                  landPrice: node.metrics.landPricePerAcre,
                  queuePending: node.metrics.queuePendingMW,
                  // New physical and cost/emissions metrics
                  annualMWh: node.metrics.annualMWh,
                  effectivePricePerMWh: node.metrics.effectivePricePerMWh,
                  annualEnergyCost: node.metrics.annualEnergyCost,
                  annualEmissionsTonnes: node.metrics.annualEmissionsTonnes,
                  landAcres: node.metrics.landAcres,
                  landCost: node.metrics.landCost,
                  componentScores: node.scores.components,
                  congestionRelief: `Rank #${node.scores.rank} of ${rankingResults.length} shown`,
                  emissionsImpact: `${node.metrics.emissionsIntensity.toFixed(1)} kg CO₂/MWh`,
                  reliabilityBoost: `${node.location.iso} - LMP: $${node.metrics.lmp.toFixed(
                    2
                  )}/MWh`,
                  narrative: generateExplanation(node, loadConfig),
                });
                setViewState((prev) => ({
                  ...prev,
                  longitude: node.location.longitude,
                  latitude: node.location.latitude,
                  zoom: 6,
                  pitch: 45,
                }));
              };

              return (
                <Marker
                  key={node.node}
                  longitude={node.location.longitude}
                  latitude={node.location.latitude}
                  anchor="bottom"
                  onClick={() => selectNodeForDetails(node)}
                >
                  <div className="relative cursor-pointer hover:scale-110 transition-transform">
                    <div 
                      className="rounded-full border-2 border-slate-950 shadow-lg"
                      style={{
                        width: `${markerSize * 4}px`,
                        height: `${markerSize * 4}px`,
                        backgroundColor: markerColor,
                        boxShadow: `0 0 ${markerSize * 3}px ${markerColor}40`
                      }}
                    />
                    {index < 10 && (
                      <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[9px] bg-slate-950/90 px-1.5 py-0.5 rounded-full border border-slate-700 whitespace-nowrap font-semibold">
                        #{node.scores.rank}
                      </span>
                    )}
                  </div>
                </Marker>
              );
            })}

            {selectedSite && (
              <Marker
                longitude={selectedSite.lng}
                latitude={selectedSite.lat}
                anchor="bottom"
              >
                <div className="relative">
                  <div className="h-5 w-5 rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50 border-2 border-slate-950" />
                  <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-ping" />
                </div>
              </Marker>
            )}

            {resources.map((r) => (
              <Marker
                key={r.id}
                longitude={r.lng}
                latitude={r.lat}
                anchor="bottom"
              >
                <div className="flex flex-col items-center gap-1">
                  <div className="h-4 w-4 rounded-full bg-sky-400 border border-slate-950 shadow shadow-sky-300/60" />
                  <span className="text-[9px] bg-slate-950/80 px-1.5 py-0.5 rounded-full border border-slate-700 whitespace-nowrap">
                    {r.type} · {r.mw} MW
                  </span>
                </div>
              </Marker>
            ))}
          </Map>

          {/* Top 5 Nodes Dashboard */}
          <AnimatePresence>
            {rankingResults && rankingResults.length > 0 && !showHero && (
              <motion.div
                initial={{ y: 110, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 110, opacity: 0 }}
                transition={{ type: "spring", stiffness: 80, damping: 18, delay: 0.2 }}
                className="absolute bottom-6 left-6 right-6 pointer-events-auto"
              >
                <div className="bg-slate-950/95 backdrop-blur-xl border border-slate-800 rounded-3xl p-4 shadow-2xl max-w-7xl mx-auto">
                  <div className="flex items-center justify-between mb-3">
                    <div className="text-[11px] uppercase tracking-[0.2em] text-emerald-400 font-semibold">
                      Top 5 Recommended Nodes
                    </div>
                    <div className="text-[10px] text-slate-400">
                      {rankingResults.length} total results • Click to expand
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-5 gap-3">
                    {rankingResults.slice(0, 5).map((node, index) => {
                      const isExpanded = expandedNodeId === node.node;
                      const rankColor = index === 0 ? '#10b981' : index === 1 ? '#84cc16' : index === 2 ? '#eab308' : '#f97316';
                      
                      return (
                        <motion.div
                          key={node.node}
                          layout
                          className={`border border-slate-700 rounded-2xl p-3 cursor-pointer transition-all ${
                            isExpanded ? 'col-span-5 bg-slate-900/80' : 'hover:bg-slate-900/50'
                          }`}
                          onClick={() => {
                            if (isExpanded) {
                              setExpandedNodeId(null);
                            } else {
                              setExpandedNodeId(node.node);
                            }
                            // Also zoom to this node on map
                            setViewState(prev => ({
                              ...prev,
                              longitude: node.location.longitude,
                              latitude: node.location.latitude,
                              zoom: 6,
                              pitch: 45
                            }));
                          }}
                        >
                          <div className="flex items-center gap-3">
                            <div 
                              className="h-3 w-3 rounded-full border-2 border-slate-950 flex-shrink-0"
                              style={{ backgroundColor: rankColor }}
                            />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1.5">
                                <span 
                                  className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                                  style={{ 
                                    backgroundColor: `${rankColor}20`,
                                    color: rankColor
                                  }}
                                >
                                  #{node.scores.rank}
                                </span>
                                <span className="text-[11px] font-semibold text-slate-200 truncate">
                                  {node.node}
                                </span>
                              </div>
                              <div className="text-[10px] text-slate-400 truncate">
                                {node.location.state} • {(node.scores.overall * 100).toFixed(0)}%
                              </div>
                            </div>
                            {!isExpanded && (
                              <svg className="w-3 h-3 text-slate-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                              </svg>
                            )}
                          </div>
                          
                          {/* Expanded details */}
                          <AnimatePresence>
                            {isExpanded && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                              >
                                <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                                  <div className="text-[10px] text-slate-300">
                                    <span className="text-slate-400">Location:</span>{" "}
                                    {node.location.county || node.location.state}
                                  </div>
                                  <div className="text-[10px] text-slate-300">
                                    <span className="text-slate-400">ISO:</span> {node.location.iso}
                                  </div>
                                  {/* Two-column layout: big annual metrics vs context */}
                                  <div className="grid grid-cols-2 gap-4 pt-2">
                                    {/* Left: big annual numbers */}
                                    <div className="space-y-1.5">
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">Annual energy cost:</span>{" "}
                                        $
                                        {Math.round(node.metrics.annualEnergyCost / 1_000_000)}
                                        M
                                      </div>
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">Annual emissions:</span>{" "}
                                        {Math.round(node.metrics.annualEmissionsTonnes / 1_000)}k tCO₂
                                      </div>
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">Land cost:</span>{" "}
                                        $
                                        {Math.round(node.metrics.landCost / 1_000)}k{" "}
                                        ({node.metrics.landAcres.toFixed(1)} acres)
                                      </div>
                                    </div>

                                    {/* Right: context metrics */}
                                    <div className="space-y-1.5">
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">AVG LMP:</span>{" "}
                                        ${node.metrics.lmp.toFixed(2)}/MWh
                                      </div>
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">ISO:</span> {node.location.iso || "—"}
                                      </div>
                                      <div className="text-[10px] text-slate-300">
                                        <span className="text-slate-400">Emissions intensity:</span>{" "}
                                        {node.metrics.emissionsIntensity.toFixed(1)} kg CO₂/MWh
                                      </div>
                                    </div>
                                  </div>

                                  {/* Component scores block, restored to bottom */}
                                  <div className="pt-3 border-t border-slate-700 mt-2">
                                    <div className="text-[10px] text-slate-400 mb-1.5">
                                      Component Scores (higher % = better fit for this load)
                                    </div>
                                    <div className="grid grid-cols-3 gap-x-3 gap-y-2 text-[10px]">
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Cost</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.cost * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Emissions</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.emissions * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Policy</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.policy * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Queue</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.queue * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Land</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.land * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span className="text-slate-400">Variability</span>
                                        <span className="font-semibold" style={{ color: rankColor }}>
                                          {(node.scores.components.variability * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                    </div>
                                  </div>
                                  
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setExpandedNodeId(null);
                                    }}
                                    className="w-full mt-3 text-[10px] py-1.5 rounded-lg border border-slate-700 hover:bg-slate-800 transition text-slate-300"
                                  >
                                    Collapse
                                  </button>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Site detail panel - show when a node is selected (from map marker or dashboard) */}
          <AnimatePresence>
            {selectedSite && !showHero && (
              <motion.div
                initial={{ y: 240, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 240, opacity: 0 }}
                transition={{ type: "spring", stiffness: 80, damping: 18 }}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 w-[95%] max-w-3xl"
              >
                <div className="rounded-3xl border border-slate-800 bg-slate-950/95 backdrop-blur-xl p-4 md:p-5 grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                  <div className="md:col-span-4">
                    <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-400 mb-1">
                      {selectedSite.rank ? `Recommended Node · Rank #${selectedSite.rank}` : 'Recommended node'}
                    </div>
                    <div className="text-sm font-semibold mb-2">
                      {selectedSite.name}
                    </div>
                    <div className="text-[11px] text-slate-300 leading-relaxed whitespace-pre-line bg-slate-900/30 rounded-xl p-3 border border-slate-800/50">
                      {selectedSite.narrative}
                    </div>
                  </div>
                  <MetricBox
                    label="Overall Score & Rank"
                    value={selectedSite.score ? `${(selectedSite.score * 100).toFixed(1)}% - ${selectedSite.congestionRelief}` : selectedSite.congestionRelief}
                  />
                  {/* Main metrics layout, aligned with Top 5 expanded cards */}
                  <div className="md:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-[10px]">
                    {/* Left: big annual numbers */}
                    <div className="space-y-1.5 border border-slate-800 rounded-2xl p-3 bg-slate-950/70">
                      <div className="text-[9px] uppercase tracking-[0.18em] text-slate-400 mb-1">
                        Annual Energy & Land
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">Annual energy cost:</span>{" "}
                        {selectedSite.annualEnergyCost != null
                          ? `$${Math.round(selectedSite.annualEnergyCost / 1_000_000)}M / year`
                          : "—"}
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">Annual emissions:</span>{" "}
                        {selectedSite.annualEmissionsTonnes != null
                          ? `${Math.round(selectedSite.annualEmissionsTonnes / 1_000)}k tCO₂/year`
                          : "—"}
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">Land cost:</span>{" "}
                        {selectedSite.landCost != null
                          ? `$${Math.round(selectedSite.landCost / 1_000)}k (${selectedSite.landAcres?.toFixed(1)} acres)`
                          : "—"}
                      </div>
                    </div>

                    {/* Right: context metrics */}
                    <div className="space-y-1.5 border border-slate-800 rounded-2xl p-3 bg-slate-950/70">
                      <div className="text-[9px] uppercase tracking-[0.18em] text-slate-400 mb-1">
                        Market & Grid Context
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">AVG LMP:</span>{" "}
                        {selectedSite.lmp != null
                          ? `$${selectedSite.lmp.toFixed(2)}/MWh`
                          : "—"}
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">ISO:</span> {selectedSite.iso || "—"}
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">Emissions intensity:</span>{" "}
                        {selectedSite.emissions != null
                          ? `${selectedSite.emissions.toFixed(1)} kg CO₂/MWh`
                          : "—"}
                      </div>
                      <div className="text-slate-300">
                        <span className="text-slate-400">Queue pending:</span>{" "}
                        {selectedSite.queuePending != null
                          ? `${selectedSite.queuePending.toFixed(1)} MW`
                          : "—"}
                      </div>
                    </div>
                  </div>
                  
                  {/* Component scores if available */}
                  {selectedSite.componentScores && (
                    <div className="md:col-span-4 border-t border-slate-800 pt-3 mt-2">
                      <div className="text-[9px] uppercase tracking-[0.2em] text-slate-400 mb-2">
                        Component Scores
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-[10px]">
                        <div>
                          <span className="text-slate-400">Cost:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.cost * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Emissions:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.emissions * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Policy:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.policy * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Queue:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.queue * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Land:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.land * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Variability:</span>
                          <span className="ml-1 font-semibold text-emerald-400">
                            {(selectedSite.componentScores.variability * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className="text-[9px] text-slate-400 mt-2">
                        Higher scores indicate better suitability for this load configuration.
                      </div>
                    </div>
                  )}
                  <div className="md:col-span-4 flex justify-between items-center mt-1">
                    <div className="text-[10px] text-slate-400">
                      Click another region on the map to compare siting options.
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSelectedSite(null)}
                        className="text-[10px] px-2.5 py-1 rounded-full border border-slate-700 hover:border-slate-500 hover:bg-slate-900 transition"
                      >
                        Clear selection
                      </button>
                      {rankingResults && (
                        <button
                          onClick={handleClearAllResults}
                          className="text-[10px] px-2.5 py-1 rounded-full border border-red-700 hover:border-red-500 hover:bg-red-900/20 transition text-red-400"
                        >
                          Clear all results
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Globe intro overlay */}
          <AnimatePresence>
            {showHero && <SpaceIntro onStart={handleHeroStart} />}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

/* ---------- UI subcomponents ---------- */

function PanelHeader() {
  return (
    <div className="mb-1">
      <div className="mt-3 text-xs md:text-sm uppercase tracking-[0.25em] text-slate-400">
        Siting engine
      </div>
      <div className="text-xl font-semibold mt-2">
        Clean siting for big loads
      </div>
    </div>
  );
}

function LoadForm({ loadConfig, setLoadConfig }) {
  const update = (field, value) => {
    setLoadConfig((prev) => {
      const newConfig = { ...prev, [field]: value };

      // Reset subType when load type changes
      if (field === "type") {
        // Set default subType based on new type
        if (value === "data_centre") {
          newConfig.subType = "always_on";
        } else if (value === "industrial") {
          newConfig.subType = "continuous_process";
        } else {
          newConfig.subType = "";
        }
      }

      return newConfig;
    });
  };

  // Determine which subtypes to show based on load type
  const getSubTypeOptions = () => {
    switch (loadConfig.type) {
      case "data_centre":
        return [
          { value: "always_on", label: "Always-On" },
          { value: "flexible_batch", label: "Flexible/Batch" }
        ];
      case "industrial":
        return [
          { value: "continuous_process", label: "Continuous Process" },
          { value: "flexible_shiftable", label: "Flexible/Shiftable Process" }
        ];
      default:
        return [];
    }
  };

  const subTypeOptions = getSubTypeOptions();
  const showSubType = subTypeOptions.length > 0;

  return (
    <div className="space-y-3 text-[13px]">
      <div className="flex flex-col gap-1">
        <label className="text-slate-300 text-[13px]">Load Type</label>
        <select
          value={loadConfig.type}
          onChange={(e) => update("type", e.target.value)}
          className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[13px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
        >
          <option value="data_centre">Data Centre</option>
          <option value="hydrogen_electrolyzer">Hydrogen Electrolyzer</option>
          <option value="industrial">Industrial</option>
          <option value="commercial">Commercial</option>
        </select>
      </div>

      {showSubType && (
        <div className="flex flex-col gap-1">
          <label className="text-slate-300 text-[13px]">Sub Type</label>
          <select
            value={loadConfig.subType}
            onChange={(e) => update("subType", e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[13px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
          >
            {subTypeOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="flex flex-col gap-1">
        <label className="text-slate-300 text-[13px]">Load Size</label>
        <div className="relative">
          <input
            type="number"
            min={0}
            max={5000}
            step={10}
            value={loadConfig.sizeMW}
            onChange={(e) => {
              const inputValue = e.target.value;
              // Allow empty string for when user is typing
              if (inputValue === "") {
                update("sizeMW", "");
                return;
              }
              const value = Number(inputValue);
              if (value >= 0 && value <= 5000) {
                update("sizeMW", value);
              }
            }}
            onBlur={(e) => {
              // If field is empty on blur, set to 0
              if (e.target.value === "") {
                update("sizeMW", 0);
              }
            }}
            className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 pr-20 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60 w-full"
            placeholder="Enter size"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 text-[12px]">
            MW
          </span>
        </div>
        <div className="text-[12px] text-slate-400">
          Enter value between 0 and 5,000 MegaWatts
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-slate-300 text-[13px]">
          How important are carbon emissions to you?
        </label>
        <div className="w-4/5">
          <input
            type="range"
            min={0}
            max={100}
            step={10}
            value={loadConfig.carbonEmissions}
            onChange={(e) => update("carbonEmissions", Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-slate-300">On-Site Generation</label>
        <select
          value={loadConfig.onSiteGeneration}
          onChange={(e) => {
            update("onSiteGeneration", e.target.value);
            // Reset configuration type when changing on-site generation
            if (e.target.value === "none") {
              update("configurationType", "");
            }
          }}
          className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
        >
          <option value="none">None</option>
          <option value="yes">Yes</option>
        </select>
      </div>

      {loadConfig.onSiteGeneration === "yes" && (
        <div className="flex flex-col gap-1">
          <label className="text-slate-300">Configuration Type</label>
          <select
            value={loadConfig.configurationType}
            onChange={(e) => update("configurationType", e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
          >
            <option value=""></option>
            <option value="solar">Solar</option>
            <option value="battery">Battery</option>
            <option value="solar_battery">Solar Battery</option>
            <option value="firm_gen">Firm Gen</option>
          </select>
        </div>
      )}
    </div>
  );
}

function LocationSelectionForm({
  locationMode,
  setLocationMode,
  selectedStates,
  setSelectedStates,
}) {
  return (
    <div className="space-y-3 text-[11px] border border-slate-800 rounded-2xl p-3 bg-slate-950">
      <div className="text-xs font-semibold text-slate-300 mb-2">
        Location Selection
      </div>

      {/* Mode selection radio buttons */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="locationMode"
            value="states"
            checked={locationMode === "states"}
            onChange={(e) => setLocationMode(e.target.value)}
            className="w-4 h-4 text-emerald-400 bg-slate-900 border-slate-700 focus:ring-emerald-400/60"
          />
          <span className="text-slate-300">Select States on the Map</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="locationMode"
            value="points"
            checked={locationMode === "points"}
            onChange={(e) => setLocationMode(e.target.value)}
            className="w-4 h-4 text-emerald-400 bg-slate-900 border-slate-700 focus:ring-emerald-400/60"
          />
          <span className="text-slate-300">Choose Specific Locations on the Map</span>
        </label>
      </div>

      {/* State mode controls */}
      {locationMode === "states" && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center justify-between gap-3">
            <div className="text-[10px] text-slate-400">
              Click on states to select or deselect them.
            </div>
            <button
              type="button"
              onClick={() => setSelectedStates(new Set(ALL_STATE_ABBRS))}
              className="inline-flex items-center whitespace-nowrap rounded-full border border-slate-700 px-2.5 py-0.5 text-[9px] font-medium text-slate-200 hover:border-slate-400 hover:bg-slate-900 transition"
            >
              Select all
            </button>
          </div>
          <div className="text-[10px] text-slate-400">
            {selectedStates.size} state{selectedStates.size !== 1 ? "s" : ""} selected
          </div>
        </div>
      )}

      {/* Points mode info */}
      {locationMode === "points" && (
        <div className="mt-3 space-y-2">
          <div className="text-[10px] text-slate-400">
            Click on the map to place location markers with 100km radius. Click inside a circle to remove it. Minimum 200km between locations.
          </div>
        </div>
      )}
    </div>
  );
}

function SubmitButton({
  loadConfig,
  locationMode,
  selectedStates,
  selectedPoints,
  rankingResults,
  setRankingResults,
  setViewState,
  setSelectedSite,
  setSelectedStates,
  setSelectedPoints,
}) {
  // Check if at least one state or point is selected
  const hasSelection =
    (locationMode === "states" && selectedStates.size > 0) ||
    (locationMode === "points" && selectedPoints.length > 0);

  return (
    <div className="space-y-2">
      <button
        onClick={async () => {
          // Convert state abbreviations to full names
          const stateNames = locationMode === "states"
            ? Array.from(selectedStates).map(abbr => STATE_NAMES[abbr] || abbr)
            : [];

          // Prepare complete data to send to backend
          const dataToSend = {
            // Load configuration
            loadConfig: {
              type: loadConfig.type,
              subType: loadConfig.subType,
              sizeMW: loadConfig.sizeMW,
              carbonEmissions: loadConfig.carbonEmissions,
              onSiteGeneration: loadConfig.onSiteGeneration,
              configurationType: loadConfig.configurationType,
            },
            // Location selection
            location: {
              mode: locationMode,
              selectedStates: stateNames,
              selectedPoints: locationMode === "points" ? selectedPoints : [],
            },
          };

          console.log("Sending data to backend:");
          console.log(JSON.stringify(dataToSend, null, 2));

          // Send to backend
          // Replace with the backend server's IP address (e.g., 'http://192.168.1.100:5001/api/submit')
          const backendUrl = 'http://localhost:5001/api/submit';

          try {
            const response = await fetch(backendUrl, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(dataToSend)
            });

            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Success:', result);
            
            // Display results on the map
            if (result.success && result.results && result.results.length > 0) {
              // Clear any state/point selections used as input now that results are displayed
              setSelectedStates(new Set());
              setSelectedPoints([]);

              setRankingResults(result.results);
              
              // Zoom to the first (best) result
              const topNode = result.results[0];
              setViewState({
                longitude: topNode.location.longitude,
                latitude: topNode.location.latitude,
                zoom: 6,
                pitch: 45,
                bearing: 0
              });
              
              // Generate explanation for top node
              const generateExplanation = (node) => {
                const scores = node.scores.components;
                const metrics = node.metrics;
                
                // Find top 2 strongest components
                const components = [
                  { name: 'cost', score: scores.cost },
                  { name: 'emissions', score: scores.emissions },
                  { name: 'policy', score: scores.policy },
                  { name: 'queue', score: scores.queue },
                  { name: 'land', score: scores.land },
                  { name: 'variability', score: scores.variability }
                ];
                
                const topComponents = components
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 2);
                
                // Build natural explanation
                let parts = [];
                
                // Opening
                parts.push(`Ranked #${node.scores.rank} with ${(node.scores.overall * 100).toFixed(0)}% overall score.`);
                
                // Add 1-2 key highlights
                const highlights = [];
                topComponents.forEach(c => {
                  if (c.score > 0.8) {
                    if (c.name === 'cost' && metrics.lmp < 35) {
                      highlights.push(`Low energy costs ($${metrics.lmp.toFixed(2)}/MWh)`);
                    } else if (c.name === 'emissions' && metrics.emissionsIntensity < 100) {
                      highlights.push(`Clean grid (${metrics.emissionsIntensity.toFixed(0)} kg CO₂/MWh)`);
                    } else if (c.name === 'queue' && metrics.queuePendingMW < 500) {
                      highlights.push(`Strong interconnection position`);
                    } else if (c.name === 'policy') {
                      highlights.push(`Favorable regulatory environment`);
                    } else if (c.name === 'variability') {
                      highlights.push(`Stable pricing`);
                    }
                  }
                });
                
                if (highlights.length > 0) {
                  parts.push(highlights.slice(0, 2).join(' and ') + '.');
                }
                
                // Location context
                parts.push(`Located in ${node.location.county}, ${node.location.state} (${node.location.iso} market).`);
                
                return parts.join(' ');
              };
              
              // Dashboard will show automatically - no need for separate detail panel
            } else {
              alert('⚠️ No results found. Try adjusting your criteria.');
            }
          } catch (error) {
            console.error('Error sending data to backend:', error);
            alert('❌ Error connecting to backend. Make sure the server is running on http://localhost:5001');
          }
        }}
        disabled={!hasSelection}
        className={`w-full px-4 py-2 rounded-xl font-semibold text-[11px] transition border shadow ${
          hasSelection
            ? 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 border-emerald-300 shadow-emerald-500/40 cursor-pointer'
            : 'bg-slate-700 text-slate-400 border-slate-600 shadow-slate-700/20 cursor-not-allowed opacity-50'
        }`}
      >
        Enter
      </button>
      {!hasSelection && (
        <div className="text-[10px] text-slate-400 text-center">
          {locationMode === "states"
            ? "Please select at least one state"
            : "Please select at least one location point"}
        </div>
      )}
      {rankingResults && (
        <div className="mt-2 text-[10px] text-center p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
          <span className="text-emerald-400 font-semibold">
            ✓ {rankingResults.length} optimal nodes found
          </span>
          <span className="text-slate-400 ml-1">
            (click markers on map to explore)
          </span>
        </div>
      )}
    </div>
  );
}

function Legend({ gradientLegend }) {
  return (
    <div className="mt-auto pt-3 border-t border-slate-800 text-[10px]">
      <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">
        Siting score legend
      </div>
      <div className="flex items-center gap-3">
        <div className="flex-1 h-1.5 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-emerald-400" />
        <div className="flex gap-3">
          {gradientLegend.map((g) => (
            <div key={g.label} className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full border border-slate-900"
                style={{ backgroundColor: g.color }}
              />
              <span className="text-slate-400">{g.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricBox({ label, value }) {
  return (
    <div className="border border-slate-800 rounded-2xl p-3 bg-slate-950/70">
      <div className="text-[9px] uppercase tracking-[0.2em] text-slate-400 mb-1">
        {label}
      </div>
      <div className="text-[11px] text-slate-100">{value}</div>
    </div>
  );
}

// Intro overlay on top of the spinning globe
function SpaceIntro({ onStart }) {
  return (
    <motion.div
      className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none"
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.8, ease: "easeInOut" }}
    >
      <div className="absolute inset-x-0 bottom-0 h-1/2 pointer-events-none bg-gradient-to-t from-slate-950/80 via-slate-950/40 to-transparent" />

      <div className="relative pointer-events-auto max-w-xl mx-4 rounded-3xl border border-slate-700/30 bg-slate-950/30 backdrop-blur-xl px-6 py-5">
        <div className="text-[11px] uppercase tracking-[0.28em] text-slate-400 mb-2">
          From space to substation
        </div>
        <div className="text-2xl font-semibold mb-2">
          Ready to optimize your next gigawatt of demand?
        </div>
        <div className="text-[11px] text-slate-300 mb-4">
          We start where the ISS would see it: a planet full of growing loads.
          Then we drop into the US grid and place your data centers,
          electrolyzers, and hubs where they{" "}
          <span className="text-emerald-300 font-semibold">
            relieve congestion and cut emissions
          </span>{" "}
          instead of blowing them up.
        </div>
        <div className="flex items-center justify-between gap-3 text-[11px]">
          <div className="text-slate-400">
            Click once. We&apos;ll rotate the Earth and land you on the US siting
            canvas.
          </div>
          <button
            onClick={onStart}
            className="shrink-0 px-4 py-2 rounded-full bg-emerald-500 text-slate-950 font-semibold text-[11px] hover:bg-emerald-400 transition border border-emerald-300 shadow shadow-emerald-500/40"
          >
            Select Locations
          </button>
        </div>
      </div>
    </motion.div>
  );
}

function labelLoadType(type) {
  switch (type) {
    case "data_center":
      return "data center";
    case "electrolyzer":
      return "hydrogen electrolyzer";
    case "ev_hub":
      return "EV fast-charging hub";
    case "industrial":
      return "industrial process load";
    default:
      return "large load";
  }
}

export default App;

// import React, { useState, useMemo, useEffect } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import Map, { Source, Layer, Marker } from "react-map-gl/mapbox";
// import "mapbox-gl/dist/mapbox-gl.css";

// const MAPBOX_TOKEN = "pk.eyJ1IjoicG9ja2V0Y2hhcmdlIiwiYSI6ImNtaHpzeWsweDByNHAycW9meWlxcjJwbG4ifQ.rGEbpo57cirYtnvXfDz5tw"; // <-- INSERT YOUR REAL TOKEN

// // Heat layer style: red = bad, green = good
// const heatLayer = {
//   id: "siting-heat",
//   type: "circle",
//   paint: {
//     "circle-radius": [
//       "interpolate",
//       ["linear"],
//       ["get", "score"],
//       0,
//       4,
//       1,
//       28,
//     ],
//     "circle-color": [
//       "interpolate",
//       ["linear"],
//       ["get", "score"],
//       0,
//       "#ef4444",
//       0.5,
//       "#eab308",
//       1,
//       "#22c55e",
//     ],
//     "circle-opacity": 0.4,
//   },
// };

// // Mock “nearby clean resources” for a chosen site (still dummy for now)
// function mockNearbyResources(lng, lat) {
//   return [
//     {
//       id: "wind-1",
//       type: "Offshore wind",
//       lng: lng - 1.2,
//       lat: lat + 0.7,
//       mw: 800,
//     },
//     {
//       id: "solar-1",
//       type: "Solar cluster",
//       lng: lng + 0.8,
//       lat: lat - 0.5,
//       mw: 600,
//     },
//     {
//       id: "bess-1",
//       type: "BESS hub",
//       lng: lng + 0.2,
//       lat: lat + 0.4,
//       mw: 400,
//     },
//   ];
// }

// // Haversine distance in km
// function haversineDistance(lat1, lon1, lat2, lon2) {
//   const R = 6371; // km
//   const toRad = (d) => (d * Math.PI) / 180;
//   const dLat = toRad(lat2 - lat1);
//   const dLon = toRad(lon2 - lon1);
//   const a =
//     Math.sin(dLat / 2) ** 2 +
//     Math.cos(toRad(lat1)) *
//       Math.cos(toRad(lat2)) *
//       Math.sin(dLon / 2) ** 2;
//   const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
//   return R * c;
// }

// function App() {
//   const [showHero, setShowHero] = useState(true);
//   const [spinEnabled, setSpinEnabled] = useState(true);

//   // Initial “near-horizon” view
//   const [viewState, setViewState] = useState({
//     longitude: -30, // Atlantic / Europe–Africa side, not empty Pacific
//     latitude: 15,
//     zoom: 1.7, // closer => more curvature
//     bearing: 0,
//     pitch: 78, // near-horizon view
//   });

//   const [loadConfig, setLoadConfig] = useState({
//     type: "data_center",
//     sizeMW: 500,
//     profile: "flat",
//     carbonTarget: "90",
//   });

//   const [gridConfig, setGridConfig] = useState({
//     interconnection: "iso_ne",
//     voltageLevel: "115kv",
//     distanceToSubstation: 10,
//   });

//   const [landConfig, setLandConfig] = useState({
//     siteSize: 50,
//     landUseType: "greenfield",
//     waterAvailability: "high",
//   });

//   const [economicConfig, setEconomicConfig] = useState({
//     capexBudget: 1000,
//     opexTarget: 50,
//     incentivePreference: "federal_itc",
//   });

//   const [timelineConfig, setTimelineConfig] = useState({
//     targetOnlineDate: "2027",
//     constructionDuration: 24,
//     permitPriority: "expedited",
//   });

//   const [selectedSite, setSelectedSite] = useState(null);
//   const [resources, setResources] = useState([]);

//   // Real ISO-NE nodes from CSV
//   const [nodes, setNodes] = useState([]);

//   // Load CSV on mount
//   useEffect(() => {
//     const loadCsv = async () => {
//       try {
//         const res = await fetch("/iso_ne_dalmp_with_latlon_only.csv");
//         const text = await res.text();
//         const lines = text.trim().split("\n");
//         if (lines.length <= 1) return;

//         const headers = lines[0].split(",");
//         const idxName = headers.indexOf("Location Name");
//         const idxEnergy = headers.indexOf("Avg Energy Component");
//         const idxCong = headers.indexOf("Avg Congestion Component");
//         const idxLat = headers.indexOf("Latitude");
//         const idxLng = headers.indexOf("Longitude");

//         const rawNodes = lines
//           .slice(1)
//           .map((line) => line.trim())
//           .filter((line) => line.length > 0)
//           .map((line) => {
//             const cols = line.split(",");
//             const name = cols[idxName];
//             const energy = parseFloat(cols[idxEnergy]);
//             const congestion = parseFloat(cols[idxCong]);
//             const lat = parseFloat(cols[idxLat]);
//             const lng = parseFloat(cols[idxLng]);
//             return { name, energy, congestion, lat, lng };
//           })
//           .filter(
//             (n) =>
//               !Number.isNaN(n.lat) &&
//               !Number.isNaN(n.lng) &&
//               !Number.isNaN(n.energy) &&
//               !Number.isNaN(n.congestion)
//           );

//         if (!rawNodes.length) return;

//         const energies = rawNodes.map((n) => n.energy);
//         const congs = rawNodes.map((n) => n.congestion);
//         const minE = Math.min(...energies);
//         const maxE = Math.max(...energies);
//         const minC = Math.min(...congs);
//         const maxC = Math.max(...congs);
//         const rangeE = maxE - minE || 1;
//         const rangeC = maxC - minC || 1;

//         const scoredNodes = rawNodes.map((n, idx) => {
//           const normEnergy = (n.energy - minE) / rangeE; // 0=cheapest
//           const normCong = (n.congestion - minC) / rangeC; // 0=most negative/least congested
//           const cost = 0.6 * normEnergy + 0.4 * normCong; // lower = better
//           const score = 1 - cost; // higher = better
//           return {
//             id: idx,
//             ...n,
//             score,
//           };
//         });

//         setNodes(scoredNodes);
//       } catch (e) {
//         console.error("Failed to load ISO-NE CSV", e);
//       }
//     };

//     loadCsv();
//   }, []);

//   // Slow Earth rotation while hero is visible (reverse direction)
//   useEffect(() => {
//     if (!showHero || !spinEnabled) return;

//     let frameId;

//     const spin = () => {
//       setViewState((prev) => ({
//         ...prev,
//         bearing: (prev.bearing - 0.06 + 360) % 360,
//       }));
//       frameId = requestAnimationFrame(spin);
//     };

//     frameId = requestAnimationFrame(spin);
//     return () => cancelAnimationFrame(frameId);
//   }, [showHero, spinEnabled]);

//   const gradientLegend = useMemo(
//     () => [
//       { color: "#22c55e", label: "High siting score" },
//       { color: "#eab308", label: "Moderate" },
//       { color: "#ef4444", label: "Avoid / constrained" },
//     ],
//     []
//   );

//   // GeoJSON for heat layer from actual nodes
//   const heatGeojson = useMemo(
//     () => ({
//       type: "FeatureCollection",
//       features: nodes.map((n) => ({
//         type: "Feature",
//         properties: { score: n.score },
//         geometry: {
//           type: "Point",
//           coordinates: [n.lng, n.lat],
//         },
//       })),
//     }),
//     [nodes]
//   );

//   // Generic camera animation helper
//   const animateCamera = (fromState, toState, duration, onDone) => {
//     const start = performance.now();

//     const frame = (now) => {
//       const t = Math.min((now - start) / duration, 1);
//       const ease = t * t * (3 - 2 * t);

//       setViewState((prev) => ({
//         ...prev,
//         longitude:
//           fromState.longitude +
//           (toState.longitude - fromState.longitude) * ease,
//         latitude:
//           fromState.latitude +
//           (toState.latitude - fromState.latitude) * ease,
//         zoom: fromState.zoom + (toState.zoom - fromState.zoom) * ease,
//         pitch: fromState.pitch + (toState.pitch - fromState.pitch) * ease,
//         bearing:
//           fromState.bearing +
//           (toState.bearing - fromState.bearing) * ease,
//       }));

//       if (t < 1) {
//         requestAnimationFrame(frame);
//       } else {
//         setViewState((prev) => ({ ...prev, ...toState }));
//         if (onDone) onDone();
//       }
//     };

//     requestAnimationFrame(frame);
//   };

//   // Hero → slightly zoom out to full Earth → then zoom into US
//   const handleHeroStart = () => {
//     setSpinEnabled(false);

//     const current = { ...viewState };

//     const fullEarthState = {
//       longitude: -100,
//       latitude: 20,
//       zoom: 1.3,
//       pitch: 45,
//       bearing: 0,
//     };

//     const usState = {
//       longitude: -71, // center roughly on ISO-NE
//       latitude: 42.5,
//       zoom: 4.5,
//       pitch: 25,
//       bearing: 0,
//     };

//     animateCamera(current, fullEarthState, 900, () => {
//       animateCamera(fullEarthState, usState, 1400, () => {
//         setShowHero(false);
//       });
//     });
//   };

//   // On map click: choose best nearby node based on score + distance
//   const handleMapClick = (event) => {
//     if (showHero) return;
//     if (!nodes.length) return;

//     const { lng, lat } = event.lngLat;

//     const radiusKm = 150; // ~100 miles
//     let bestInRadius = null;
//     let bestScore = -Infinity;
//     let closestOverall = null;
//     let closestDist = Infinity;

//     for (const n of nodes) {
//       const d = haversineDistance(lat, lng, n.lat, n.lng);
//       if (d < closestDist) {
//         closestDist = d;
//         closestOverall = n;
//       }
//       if (d <= radiusKm && n.score > bestScore) {
//         bestScore = n.score;
//         bestInRadius = n;
//       }
//     }

//     const chosen = bestInRadius || closestOverall;
//     if (!chosen) return;

//     const site = {
//       lng: chosen.lng,
//       lat: chosen.lat,
//       nodeName: chosen.name,
//       avgEnergy: chosen.energy,
//       avgCongestion: chosen.congestion,
//       score: chosen.score,
//       // county: chosen.county || null, // add once your CSV has a County column
//       congestionRelief: "Prototype: higher headroom vs ISO-NE average.",
//       emissionsImpact:
//         "Prototype: lower LMP when paired with nearby clean supply.",
//       reliabilityBoost: "Prototype: strong nodal connectivity in ISO-NE.",
//       narrative: `Anchored by ${loadConfig.sizeMW} MW ${labelLoadType(
//         loadConfig.type
//       )} at an ISO-NE node with low energy prices and mild/negative congestion.`,
//     };

//     setSelectedSite(site);
//     setResources(mockNearbyResources(chosen.lng, chosen.lat));

//     setViewState((prev) => ({
//       ...prev,
//       longitude: chosen.lng,
//       latitude: chosen.lat,
//       zoom: 6,
//       pitch: 45,
//     }));
//   };

//   return (
//     <div className="h-screen w-screen bg-slate-950 text-slate-50 overflow-hidden">
//       {/* Top nav */}
//       <div className="absolute top-0 left-0 right-0 z-30 flex items-center justify-between px-6 py-4 bg-gradient-to-b from-slate-950/90 to-transparent">
//         <div className="flex items-center gap-2">
//           <div className="h-8 w-8 rounded-xl bg-emerald-400/20 border border-emerald-300/60 flex items-center justify-center text-xs font-bold">
//             GW
//           </div>
//           <div>
//             <div className="text-sm font-semibold tracking-widest uppercase">
//               Smart Siting
//             </div>
//             <div className="text-xs text-slate-400">
//               Place big loads where they help the grid
//             </div>
//           </div>
//         </div>
//         <div className="hidden md:flex items-center gap-4 text-xs text-slate-400">
//           <span>RWE Hackathon 2025</span>
//           <span className="h-1 w-1 rounded-full bg-emerald-400" />
//           <span>Prototype — siting engine v0.1</span>
//         </div>
//       </div>

//       <div className="flex h-full pt-12">
//         {/* Left control panel */}
//         <div className="relative z-10 w-full max-w-md border-r border-slate-800 bg-slate-950/95 backdrop-blur-md px-5 py-6 flex flex-col gap-4 overflow-y-auto">
//           <PanelHeader />
//           <LoadForm loadConfig={loadConfig} setLoadConfig={setLoadConfig} />
//           <GridConnectionForm
//             gridConfig={gridConfig}
//             setGridConfig={setGridConfig}
//           />
//           <LandRequirementsForm
//             landConfig={landConfig}
//             setLandConfig={setLandConfig}
//           />
//           <EconomicConstraintsForm
//             economicConfig={economicConfig}
//             setEconomicConfig={setEconomicConfig}
//           />
//           <ProjectTimelineForm
//             timelineConfig={timelineConfig}
//             setTimelineConfig={setTimelineConfig}
//           />
//           <HowItWorks />
//           <Legend gradientLegend={gradientLegend} />
//         </div>

//         {/* Map area */}
//         <div className="relative flex-1">
//           <Map
//             mapboxAccessToken={MAPBOX_TOKEN}
//             mapStyle={
//               showHero
//                 ? "mapbox://styles/mapbox/satellite-v9"
//                 : "mapbox://styles/mapbox/dark-v11"
//             }
//             {...viewState}
//             onMove={(evt) => setViewState(evt.viewState)}
//             onClick={handleMapClick}
//             minZoom={1}
//             maxZoom={10}
//             projection="globe"
//             style={{ width: "100%", height: "100%" }}
//           >
//             {/* Real node scores as heat layer */}
//             {nodes.length > 0 && (
//               <Source
//                 id="siting-heat-src"
//                 type="geojson"
//                 data={heatGeojson}
//               >
//                 <Layer {...heatLayer} />
//               </Source>
//             )}

//             {selectedSite && (
//               <Marker
//                 longitude={selectedSite.lng}
//                 latitude={selectedSite.lat}
//                 anchor="bottom"
//               >
//                 <div className="relative">
//                   <div className="h-5 w-5 rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50 border-2 border-slate-950" />
//                   <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-ping" />
//                 </div>
//               </Marker>
//             )}

//             {resources.map((r) => (
//               <Marker
//                 key={r.id}
//                 longitude={r.lng}
//                 latitude={r.lat}
//                 anchor="bottom"
//               >
//                 <div className="flex flex-col items-center gap-1">
//                   <div className="h-4 w-4 rounded-full bg-sky-400 border border-slate-950 shadow shadow-sky-300/60" />
//                   <span className="text-[9px] bg-slate-950/80 px-1.5 py-0.5 rounded-full border border-slate-700 whitespace-nowrap">
//                     {r.type} · {r.mw} MW
//                   </span>
//                 </div>
//               </Marker>
//             ))}
//           </Map>

//           {/* Site detail panel */}
//           <AnimatePresence>
//             {selectedSite && !showHero && (
//               <motion.div
//                 initial={{ y: 240, opacity: 0 }}
//                 animate={{ y: 0, opacity: 1 }}
//                 exit={{ y: 240, opacity: 0 }}
//                 transition={{ type: "spring", stiffness: 80, damping: 18 }}
//                 className="absolute bottom-4 left-1/2 -translate-x-1/2 w-[95%] max-w-3xl"
//               >
//                 <div className="rounded-3xl border border-slate-800 bg-slate-950/95 backdrop-blur-xl p-4 md:p-5 grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
//                   <div className="md:col-span-2">
//                     <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-400 mb-1">
//                       Recommended ISO-NE node
//                     </div>
//                     <div className="text-sm font-semibold mb-1">
//                       {selectedSite.nodeName}
//                     </div>
//                     <div className="text-[10px] text-slate-400 mb-1">
//                       Lat {selectedSite.lat.toFixed(3)}, Lon{" "}
//                       {selectedSite.lng.toFixed(3)}
//                     </div>
//                     <div className="text-[11px] text-slate-300">
//                       {selectedSite.narrative}
//                     </div>
//                   </div>

//                   <MetricBox label="Price + congestion">
//                     <div>
//                       Avg energy component:{" "}
//                       {selectedSite.avgEnergy != null
//                         ? `${selectedSite.avgEnergy.toFixed(2)} $/MWh`
//                         : "–"}
//                     </div>
//                     <div>
//                       Avg congestion component:{" "}
//                       {selectedSite.avgCongestion != null
//                         ? `${selectedSite.avgCongestion.toFixed(2)} $/MWh`
//                         : "–"}
//                     </div>
//                     <div className="text-[10px] text-slate-400 mt-1">
//                       Higher score = cheaper energy + less congestion.
//                     </div>
//                   </MetricBox>

//                   <MetricBox label="Grid impact (prototype)">
//                     <div>{selectedSite.congestionRelief}</div>
//                     <div>{selectedSite.emissionsImpact}</div>
//                   </MetricBox>

//                   <MetricBox label="Reliability (prototype)">
//                     <div>{selectedSite.reliabilityBoost}</div>
//                   </MetricBox>

//                   <div className="md:col-span-4 flex justify-between items-center mt-1">
//                     <div className="text-[10px] text-slate-400">
//                       Click another region on the map to compare siting options.
//                     </div>
//                     <button
//                       onClick={() => setSelectedSite(null)}
//                       className="text-[10px] px-2.5 py-1 rounded-full border border-slate-700 hover:border-slate-500 hover:bg-slate-900 transition"
//                     >
//                       Clear selection
//                     </button>
//                   </div>
//                 </div>
//               </motion.div>
//             )}
//           </AnimatePresence>

//           {/* Globe intro overlay */}
//           <AnimatePresence>
//             {showHero && <SpaceIntro onStart={handleHeroStart} />}
//           </AnimatePresence>
//         </div>
//       </div>
//     </div>
//   );
// }

// /* ---------- UI subcomponents ---------- */

// function PanelHeader() {
//   return (
//     <div className="mb-1">
//       <div className="text-xs uppercase tracking-[0.25em] text-slate-400">
//         Siting engine
//       </div>
//       <div className="text-lg font-semibold mt-1">
//         Turn the next gigawatt into a climate asset.
//       </div>
//       <div className="text-[11px] text-slate-400 mt-2">
//         Configure a large electro-intensive load, click a region on the US map,
//         and the engine will zoom into a candidate node and show nearby clean
//         resources that make that siting defensible.
//       </div>
//     </div>
//   );
// }

// function LoadForm({ loadConfig, setLoadConfig }) {
//   const update = (field, value) =>
//     setLoadConfig((prev) => ({ ...prev, [field]: value }));

//   return (
//     <div className="space-y-3 text-[11px]">
//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Load type</label>
//         <select
//           value={loadConfig.type}
//           onChange={(e) => update("type", e.target.value)}
//           className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//         >
//           <option value="data_center">Data center</option>
//           <option value="electrolyzer">Hydrogen electrolyzer</option>
//           <option value="ev_hub">EV fast-charging hub</option>
//           <option value="industrial">Process heat / industrial load</option>
//         </select>
//       </div>

//       <div className="flex gap-3">
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Size (MW)</label>
//           <input
//             type="number"
//             min={10}
//             max={2000}
//             value={loadConfig.sizeMW}
//             onChange={(e) => update("sizeMW", Number(e.target.value || 0))}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Load profile</label>
//           <select
//             value={loadConfig.profile}
//             onChange={(e) => update("profile", e.target.value)}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           >
//             <option value="flat">Flat 24×7</option>
//             <option value="solar_following">Solar-following</option>
//             <option value="wind_following">Wind-following</option>
//             <option value="flexible">Interruptible / flexible</option>
//           </select>
//         </div>
//       </div>

//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Carbon-free energy target (%)</label>
//         <input
//           type="range"
//           min={0}
//           max={100}
//           step={10}
//           value={loadConfig.carbonTarget}
//           onChange={(e) => update("carbonTarget", e.target.value)}
//           className="w-full"
//         />
//         <div className="text-[10px] text-slate-400 flex justify-between">
//           <span>Grid-average</span>
//           <span className="font-semibold text-slate-200">
//             {loadConfig.carbonTarget}% CFE
//           </span>
//         </div>
//       </div>
//     </div>
//   );
// }

// function HowItWorks() {
//   return (
//     <div className="mt-3 border border-slate-800 rounded-2xl p-3 text-[10px] bg-slate-950">
//       <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">
//         Flow
//       </div>
//       <ol className="space-y-1.5">
//         <li>
//           <span className="font-semibold text-slate-200">1. Set the load</span>{" "}
//           – choose type, size, and carbon target.
//         </li>
//         <li>
//           <span className="font-semibold text-slate-200">2. Scan the map</span>{" "}
//           – the gradient shows where adding load unlocks clean resources instead
//           of stressing the grid.
//         </li>
//         <li>
//           <span className="font-semibold text-slate-200">3. Click a region</span>{" "}
//           – zoom into an optimal node and see the portfolio of nearby wind,
//           solar, and storage that can anchor the siting choice.
//         </li>
//       </ol>
//     </div>
//   );
// }

// function Legend({ gradientLegend }) {
//   return (
//     <div className="mt-auto pt-3 border-t border-slate-800 text-[10px]">
//       <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">
//         Siting score legend
//       </div>
//       <div className="flex items-center gap-3">
//         <div className="flex-1 h-1.5 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-emerald-400" />
//         <div className="flex gap-3">
//           {gradientLegend.map((g) => (
//             <div key={g.label} className="flex items-center gap-1.5">
//               <span
//                 className="h-2 w-2 rounded-full border border-slate-900"
//                 style={{ backgroundColor: g.color }}
//               />
//               <span className="text-slate-400">{g.label}</span>
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// }

// function GridConnectionForm({ gridConfig, setGridConfig }) {
//   const update = (field, value) =>
//     setGridConfig((prev) => ({ ...prev, [field]: value }));

//   return (
//     <div className="space-y-3 text-[11px]">
//       <div className="text-xs font-semibold text-slate-300 mb-2">
//         Grid Connection
//       </div>
//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Interconnection queue</label>
//         <select
//           value={gridConfig.interconnection}
//           onChange={(e) => update("interconnection", e.target.value)}
//           className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//         >
//           <option value="iso_ne">ISO-NE</option>
//           <option value="pjm">PJM</option>
//           <option value="caiso">CAISO</option>
//           <option value="ercot">ERCOT</option>
//           <option value="miso">MISO</option>
//         </select>
//       </div>

//       <div className="flex gap-3">
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Voltage level</label>
//           <select
//             value={gridConfig.voltageLevel}
//             onChange={(e) => update("voltageLevel", e.target.value)}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           >
//             <option value="69kv">69 kV</option>
//             <option value="115kv">115 kV</option>
//             <option value="230kv">230 kV</option>
//             <option value="345kv">345 kV</option>
//           </select>
//         </div>
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Distance to sub (mi)</label>
//           <input
//             type="number"
//             min={1}
//             max={100}
//             value={gridConfig.distanceToSubstation}
//             onChange={(e) =>
//               update("distanceToSubstation", Number(e.target.value || 1))
//             }
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// function LandRequirementsForm({ landConfig, setLandConfig }) {
//   const update = (field, value) =>
//     setLandConfig((prev) => ({ ...prev, [field]: value }));

//   return (
//     <div className="space-y-3 text-[11px]">
//       <div className="text-xs font-semibold text-slate-300 mb-2">
//         Land & Resources
//       </div>
//       <div className="flex gap-3">
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Site size (acres)</label>
//           <input
//             type="number"
//             min={10}
//             max={500}
//             value={landConfig.siteSize}
//             onChange={(e) => update("siteSize", Number(e.target.value || 10))}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Land use type</label>
//           <select
//             value={landConfig.landUseType}
//             onChange={(e) => update("landUseType", e.target.value)}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           >
//             <option value="greenfield">Greenfield</option>
//             <option value="brownfield">Brownfield</option>
//             <option value="industrial">Industrial</option>
//           </select>
//         </div>
//       </div>

//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Water availability</label>
//         <select
//           value={landConfig.waterAvailability}
//           onChange={(e) => update("waterAvailability", e.target.value)}
//           className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//         >
//           <option value="high">High availability</option>
//           <option value="medium">Medium availability</option>
//           <option value="low">Low / dry cooling required</option>
//         </select>
//       </div>
//     </div>
//   );
// }

// function EconomicConstraintsForm({ economicConfig, setEconomicConfig }) {
//   const update = (field, value) =>
//     setEconomicConfig((prev) => ({ ...prev, [field]: value }));

//   return (
//     <div className="space-y-3 text-[11px]">
//       <div className="text-xs font-semibold text-slate-300 mb-2">
//         Economic Constraints
//       </div>
//       <div className="flex gap-3">
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">CapEx budget ($M)</label>
//           <input
//             type="number"
//             min={100}
//             max={5000}
//             step={100}
//             value={economicConfig.capexBudget}
//             onChange={(e) =>
//               update("capexBudget", Number(e.target.value || 100))
//             }
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">OpEx target ($/MWh)</label>
//           <input
//             type="number"
//             min={10}
//             max={200}
//             value={economicConfig.opexTarget}
//             onChange={(e) =>
//               update("opexTarget", Number(e.target.value || 10))
//             }
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//       </div>

//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Incentive preference</label>
//         <select
//           value={economicConfig.incentivePreference}
//           onChange={(e) => update("incentivePreference", e.target.value)}
//           className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//         >
//           <option value="federal_itc">Federal ITC</option>
//           <option value="federal_ptc">Federal PTC</option>
//           <option value="state_programs">State programs</option>
//           <option value="tax_credits_45v">Tax credits (45V)</option>
//         </select>
//       </div>
//       <div className="text-[10px] text-slate-400">
//         Maximize available federal and state incentives for clean energy projects.
//       </div>
//     </div>
//   );
// }

// function ProjectTimelineForm({ timelineConfig, setTimelineConfig }) {
//   const update = (field, value) =>
//     setTimelineConfig((prev) => ({ ...prev, [field]: value }));

//   return (
//     <div className="space-y-3 text-[11px]">
//       <div className="text-xs font-semibold text-slate-300 mb-2">
//         Project Timeline
//       </div>
//       <div className="flex gap-3">
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Target online date</label>
//           <select
//             value={timelineConfig.targetOnlineDate}
//             onChange={(e) => update("targetOnlineDate", e.target.value)}
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           >
//             <option value="2026">2026</option>
//             <option value="2027">2027</option>
//             <option value="2028">2028</option>
//             <option value="2029">2029</option>
//             <option value="2030">2030+</option>
//           </select>
//         </div>
//         <div className="flex-1 flex flex-col gap-1">
//           <label className="text-slate-300">Construction (mo)</label>
//           <input
//             type="number"
//             min={12}
//             max={48}
//             value={timelineConfig.constructionDuration}
//             onChange={(e) =>
//               update("constructionDuration", Number(e.target.value || 12))
//             }
//             className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//           />
//         </div>
//       </div>

//       <div className="flex flex-col gap-1">
//         <label className="text-slate-300">Permitting priority</label>
//         <select
//           value={timelineConfig.permitPriority}
//           onChange={(e) => update("permitPriority", e.target.value)}
//           className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-[11px] focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
//         >
//           <option value="expedited">Expedited / fast-track</option>
//           <option value="standard">Standard review</option>
//           <option value="flexible">Flexible timeline</option>
//         </select>
//       </div>
//       <div className="text-[10px] text-slate-400">
//         Prioritize sites with favorable permitting environments and streamlined approval processes.
//       </div>
//     </div>
//   );
// }

// // Metric box now takes children, not a single string
// function MetricBox({ label, children }) {
//   return (
//     <div className="border border-slate-800 rounded-2xl p-3 bg-slate-950/70">
//       <div className="text-[9px] uppercase tracking-[0.2em] text-slate-400 mb-1">
//         {label}
//       </div>
//       <div className="text-[11px] text-slate-100 space-y-0.5">{children}</div>
//     </div>
//   );
// }

// // Intro overlay on top of the spinning globe
// function SpaceIntro({ onStart }) {
//   return (
//     <motion.div
//       className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none"
//       initial={{ opacity: 1 }}
//       animate={{ opacity: 1 }}
//       exit={{ opacity: 0 }}
//       transition={{ duration: 0.8, ease: "easeInOut" }}
//     >
//       <div className="absolute inset-x-0 bottom-0 h-1/2 pointer-events-none bg-gradient-to-t from-slate-950/80 via-slate-950/40 to-transparent" />

//       <div className="relative pointer-events-auto max-w-xl mx-4 rounded-3xl border border-slate-700/30 bg-slate-950/30 backdrop-blur-xl px-6 py-5">
//         <div className="text-[11px] uppercase tracking-[0.28em] text-slate-400 mb-2">
//           From space to substation
//         </div>
//         <div className="text-2xl font-semibold mb-2">
//           Ready to optimize your next gigawatt of demand?
//         </div>
//         <div className="text-[11px] text-slate-300 mb-4">
//           We start where the ISS would see it: a planet full of growing loads.
//           Then we drop into the US grid and place your data centers,
//           electrolyzers, and hubs where they{" "}
//           <span className="text-emerald-300 font-semibold">
//             relieve congestion and cut emissions
//           </span>{" "}
//           instead of blowing them up.
//         </div>
//         <div className="flex items-center justify-between gap-3 text-[11px]">
//           <div className="text-slate-400">
//             Click once. We&apos;ll rotate the Earth and land you on the US siting
//             canvas.
//           </div>
//           <button
//             onClick={onStart}
//             className="shrink-0 px-4 py-2 rounded-full bg-emerald-500 text-slate-950 font-semibold text-[11px] hover:bg-emerald-400 transition border border-emerald-300 shadow shadow-emerald-500/40"
//           >
//             Let&apos;s go
//           </button>
//         </div>
//       </div>
//     </motion.div>
//   );
// }

// function labelLoadType(type) {
//   switch (type) {
//     case "data_center":
//       return "data center";
//     case "electrolyzer":
//       return "hydrogen electrolyzer";
//     case "ev_hub":
//       return "EV fast-charging hub";
//     case "industrial":
//       return "industrial process load";
//     default:
//       return "large load";
//   }
// }

// export default App;