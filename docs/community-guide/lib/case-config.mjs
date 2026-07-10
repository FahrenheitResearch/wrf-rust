import { clamp, round } from "./sizing.mjs";
import { getSourceProfile } from "./source-profiles.mjs";

export const DEFAULT_START_UTC = "2026-04-03T00:00";

function asInteger(value, fallback) {
  const parsed = parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function partsFromDate(date) {
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
    hour: date.getUTCHours(),
    minute: date.getUTCMinutes(),
    second: date.getUTCSeconds(),
  };
}

export function parseStartUtc(value = DEFAULT_START_UTC) {
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/);
  if (!match) {
    return parseStartUtc(DEFAULT_START_UTC);
  }

  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
    hour: Number(match[4]),
    minute: Number(match[5]),
    second: 0,
  };
}

export function addHours(parts, hours) {
  const date = new Date(
    Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second || 0),
  );
  date.setUTCHours(date.getUTCHours() + hours);
  return partsFromDate(date);
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

export function formatStamp(parts) {
  return `${parts.year}-${pad2(parts.month)}-${pad2(parts.day)}_${pad2(parts.hour)}:${pad2(parts.minute)}:${pad2(parts.second || 0)}`;
}

export function deriveProjection(centerLat, centerLon) {
  return {
    mapProj: "lambert",
    refLat: round(centerLat, 4),
    refLon: round(centerLon, 4),
    truelat1: round(clamp(centerLat - 5, -60, 60), 4),
    truelat2: round(clamp(centerLat + 25, -60, 60), 4),
    standLon: round(centerLon, 4),
  };
}

export function normalizeCaseConfig(rawState) {
  const sourceProfile = getSourceProfile(rawState.source);
  const start = parseStartUtc(rawState.startUtc);
  const runHours = clamp(asInteger(rawState.runHours, 6), 1, 72);
  const end = addHours(start, runHours);
  const eWe = clamp(asInteger(rawState.e_we, 200), 80, 480);
  const eSn = clamp(asInteger(rawState.e_sn, 200), 80, 480);
  const centerLat = round(clamp(asNumber(rawState.centerLat, 35), -60, 60), 4);
  const centerLon = round(clamp(asNumber(rawState.centerLon, -97), -180, 180), 4);
  const historyMinutes = clamp(asInteger(rawState.historyMinutes, 60), 15, 180);
  const numMetgridLevels = clamp(
    asInteger(rawState.numMetgridLevels, sourceProfile.numMetgridLevels),
    1,
    99,
  );
  const numMetgridSoilLevels = clamp(
    asInteger(rawState.numMetgridSoilLevels, sourceProfile.numMetgridSoilLevels),
    1,
    20,
  );
  const projection = deriveProjection(centerLat, centerLon);

  return {
    presetId: rawState.presetId,
    platform: rawState.platform || "windows",
    ramGb: clamp(asInteger(rawState.ramGb, 32), 8, 192),
    diskGb: clamp(asInteger(rawState.diskGb, 200), 20, 4000),
    workspaceLocation: rawState.workspaceLocation || "linux-home",
    source: sourceProfile.id,
    sourceProfile,
    startUtc: `${start.year}-${pad2(start.month)}-${pad2(start.day)}T${pad2(start.hour)}:${pad2(start.minute)}`,
    start,
    end,
    startStamp: formatStamp(start),
    endStamp: formatStamp(end),
    runHours,
    historyMinutes,
    intervalSeconds: sourceProfile.defaultIntervalSeconds,
    dxMeters: 3000,
    dyMeters: 3000,
    eWe,
    eSn,
    centerLat,
    centerLon,
    geogDataPath: String(rawState.geogDataPath || "/home/YOUR_USER/WPS_GEOG").trim() || "/home/YOUR_USER/WPS_GEOG",
    numMetgridLevels,
    numMetgridSoilLevels,
    ...projection,
  };
}

export function formatDateParts(parts) {
  return {
    year: String(parts.year),
    month: pad2(parts.month),
    day: pad2(parts.day),
    hour: pad2(parts.hour),
  };
}
