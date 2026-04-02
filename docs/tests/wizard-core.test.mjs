import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { normalizeCaseConfig } from "../lib/case-config.mjs";
import { renderNamelistInput, renderNamelistWps } from "../lib/namelist-render.mjs";
import { applyPreset, getPreset } from "../lib/presets.mjs";
import { evaluatePreflight } from "../lib/preflight-core.mjs";
import { recommendationForRam } from "../lib/sizing.mjs";
import { buildSourcePlan } from "../lib/source-profiles.mjs";
import { validateCase } from "../lib/validate-case.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsDir = path.resolve(__dirname, "..");

test("default ECMWF preset matches the starter namelist.wps fixture exactly", async () => {
  const state = applyPreset(
    {
      presetId: "starter-32",
      platform: "windows",
      ramGb: 32,
      diskGb: 200,
      workspaceLocation: "linux-home",
      source: "ecmwf",
      startUtc: "2026-04-03T00:00",
      runHours: 6,
      historyMinutes: 60,
      e_we: 200,
      e_sn: 200,
      centerLat: 35,
      centerLon: -97,
      geogDataPath: "/home/YOUR_USER/WPS_GEOG",
      numMetgridLevels: 33,
      numMetgridSoilLevels: 4,
    },
    "starter-32",
  );
  const config = normalizeCaseConfig({ ...state, geogDataPath: "/home/YOUR_USER/WPS_GEOG" });
  const expected = await fs.readFile(path.join(docsDir, "starter-files/namelist.wps.ecmwf-starter"), "utf8");
  assert.equal(renderNamelistWps(config), expected);
});

test("default ECMWF preset matches the starter namelist.input fixture exactly", async () => {
  const state = applyPreset(
    {
      presetId: "starter-32",
      platform: "windows",
      ramGb: 32,
      diskGb: 200,
      workspaceLocation: "linux-home",
      source: "ecmwf",
      startUtc: "2026-04-03T00:00",
      runHours: 6,
      historyMinutes: 60,
      e_we: 200,
      e_sn: 200,
      centerLat: 35,
      centerLon: -97,
      geogDataPath: "/home/YOUR_USER/WPS_GEOG",
      numMetgridLevels: 33,
      numMetgridSoilLevels: 4,
    },
    "starter-32",
  );
  const config = normalizeCaseConfig({ ...state, geogDataPath: "/home/YOUR_USER/WPS_GEOG" });
  const expected = await fs.readFile(path.join(docsDir, "starter-files/namelist.input.ecmwf-starter"), "utf8");
  assert.equal(renderNamelistInput(config), expected);
});

test("presets scale the case conservatively across spec tiers", () => {
  const learning = getPreset("learning-16");
  const starter = getPreset("starter-32");
  const larger = getPreset("larger-64");

  assert.equal(learning.defaults.source, "gfs");
  assert.equal(starter.defaults.source, "ecmwf");
  assert.ok(learning.defaults.e_we < starter.defaults.e_we);
  assert.ok(starter.defaults.e_we < larger.defaults.e_we);
  assert.ok(learning.defaults.runHours < larger.defaults.runHours);
});

test("source plans stay source-aware", () => {
  const ecmwf = buildSourcePlan(normalizeCaseConfig({ source: "ecmwf" }));
  const gfs = buildSourcePlan(normalizeCaseConfig({ source: "gfs" }));
  const hrrr = buildSourcePlan(normalizeCaseConfig({ source: "hrrr" }));

  assert.equal(ecmwf.vtable, "Vtable.ECMWF_opendata");
  assert.equal(ecmwf.fgName, "'SFILE', 'SOILFILE', 'FILE'");
  assert.equal(gfs.vtable, "Vtable.GFS");
  assert.equal(gfs.fgName, "'FILE'");
  assert.equal(hrrr.vtable, "Vtable.raphrrr");
});

test("preflight catches Windows-mounted workspaces and low disk", () => {
  const config = normalizeCaseConfig({
    platform: "windows",
    ramGb: 32,
    diskGb: 60,
    workspaceLocation: "windows-drive",
    source: "ecmwf",
    e_we: 200,
    e_sn: 200,
    centerLat: 35,
    centerLon: -97,
  });
  const result = evaluatePreflight({
    ...config,
    hasWsl: true,
    hasVirt: true,
    hasWrf: false,
    hasWps: false,
    hasGeog: false,
    wslVersion: 2,
    wslConfigApplied: true,
  });

  assert.equal(result.overall.label, "Blocked");
  assert.ok(result.items.some((item) => item.title === "Workspace path" && item.status === "fail"));
  assert.ok(result.items.some((item) => item.title === "Disk headroom" && item.status === "fail"));
});

test("validator warns when the domain is too large for the RAM tier", () => {
  const config = normalizeCaseConfig({
    presetId: "learning-16",
    ramGb: 16,
    source: "gfs",
    e_we: 240,
    e_sn: 240,
    geogDataPath: "/home/YOUR_USER/WPS_GEOG",
  });
  const result = validateCase(config);
  assert.ok(result.errors.length >= 1);
});

test("validator catches Windows-style geography paths", () => {
  const config = normalizeCaseConfig({
    source: "ecmwf",
    geogDataPath: "C:\\Users\\drew\\WPS_GEOG",
  });
  const result = validateCase(config);
  assert.ok(result.errors.some((entry) => /geog_data_path/.test(entry.text)));
});

test("RAM recommendations expose the expected 32 GB starter tier", () => {
  const rec = recommendationForRam(32);
  assert.equal(rec.memory, 24);
  assert.equal(rec.processors, 12);
  assert.equal(rec.swap, 16);
  assert.equal(rec.default3km, "220 to 300");
});
