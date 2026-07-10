import { getPreset } from "./presets.mjs";
import { recommendationForRam } from "./sizing.mjs";

function addMessage(collection, level, text) {
  collection.push({ level, text });
}

export function validateCase(caseConfig) {
  const errors = [];
  const warnings = [];
  const ramPlan = recommendationForRam(caseConfig.ramGb);
  const targetPoints = Math.max(caseConfig.eWe, caseConfig.eSn);
  const preset = getPreset(caseConfig.presetId);

  if (!caseConfig.geogDataPath.startsWith("/")) {
    addMessage(errors, "error", "geog_data_path should point at a Linux path such as /home/YOUR_USER/WPS_GEOG, not a Windows path.");
  }

  if (targetPoints > ramPlan.aggressiveMax) {
    addMessage(errors, "error", `The requested ${targetPoints} x ${targetPoints} domain is beyond the aggressive 3 km ceiling for ${caseConfig.ramGb} GB.`);
  } else if (targetPoints > ramPlan.defaultMax) {
    addMessage(warnings, "warning", `The requested ${targetPoints} x ${targetPoints} domain is above the comfortable 3 km target for ${caseConfig.ramGb} GB.`);
  }

  if (targetPoints < 140) {
    addMessage(warnings, "warning", "This is a very small 3 km domain. It is fine for a first success, but it may be too cramped for broad severe-weather setups.");
  }

  if (caseConfig.historyMinutes < 30 && targetPoints >= 260) {
    addMessage(warnings, "warning", "Very frequent output on a large domain will fill disk fast. Use 60 minutes unless you have a concrete reason not to.");
  }

  if (caseConfig.runHours > 12) {
    addMessage(warnings, "warning", "Long first runs multiply every mistake. For a first success, a 3 to 6 hour sanity case is still the safer move.");
  }

  if (preset && caseConfig.ramGb < preset.defaults.ramGb) {
    addMessage(warnings, "warning", `The selected ${preset.label} preset assumes more host RAM than you reported.`);
  }

  if ((caseConfig.source === "hrrr" || caseConfig.source === "rap")
      && (caseConfig.centerLat < 15 || caseConfig.centerLat > 60 || caseConfig.centerLon < -140 || caseConfig.centerLon > -55)) {
    addMessage(warnings, "warning", "HRRR and RAP are CONUS-focused. Re-check the source choice if your domain is far outside that area.");
  }

  if (caseConfig.numMetgridLevels <= caseConfig.numMetgridSoilLevels) {
    addMessage(errors, "error", "num_metgrid_levels should be comfortably larger than num_metgrid_soil_levels.");
  }

  return { errors, warnings };
}
