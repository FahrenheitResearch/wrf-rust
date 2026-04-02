export const PRESETS = [
  {
    id: "learning-16",
    label: "16 GB learning",
    description: "Smallest practical 3 km starter. Good for learning the workflow without pretending the machine is bigger than it is.",
    defaults: {
      ramGb: 16,
      source: "gfs",
      startUtc: "2026-04-03T00:00",
      runHours: 3,
      historyMinutes: 60,
      e_we: 160,
      e_sn: 160,
      centerLat: 35,
      centerLon: -97,
      numMetgridLevels: 34,
      numMetgridSoilLevels: 4,
    },
  },
  {
    id: "starter-32",
    label: "32 GB 3 km starter",
    description: "The repo default. Balanced for a short convection-allowing sanity run on the kind of gaming PC most community users own.",
    defaults: {
      ramGb: 32,
      source: "ecmwf",
      startUtc: "2026-04-03T00:00",
      runHours: 6,
      historyMinutes: 60,
      e_we: 200,
      e_sn: 200,
      centerLat: 35,
      centerLon: -97,
      numMetgridLevels: 33,
      numMetgridSoilLevels: 4,
    },
  },
  {
    id: "larger-64",
    label: "64 GB larger-domain",
    description: "For users who already have room to grow and want a broader single-domain 3 km case without immediately reaching for nests.",
    defaults: {
      ramGb: 64,
      source: "ecmwf",
      startUtc: "2026-04-03T00:00",
      runHours: 12,
      historyMinutes: 60,
      e_we: 320,
      e_sn: 320,
      centerLat: 35,
      centerLon: -97,
      numMetgridLevels: 33,
      numMetgridSoilLevels: 4,
    },
  },
];

export function getPreset(id) {
  return PRESETS.find((preset) => preset.id === id) || PRESETS[1];
}

export function applyPreset(state, presetId) {
  const preset = getPreset(presetId);
  return {
    ...state,
    ...preset.defaults,
    presetId: preset.id,
  };
}
