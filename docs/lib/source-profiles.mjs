const SOURCE_PROFILES = {
  ecmwf: {
    id: "ecmwf",
    label: "ECMWF open data",
    shortLabel: "ECMWF",
    defaultIntervalSeconds: 3600,
    vtable: "Vtable.ECMWF_opendata",
    numMetgridLevels: 33,
    numMetgridSoilLevels: 4,
    splitPrefixes: true,
    steps: [
      "Point Vtable at the repo starter file Vtable.ECMWF_opendata.",
      "Run ungrib.exe once for pressure files with prefix = 'FILE'.",
      "Run ungrib.exe again for surface files with prefix = 'SFILE'.",
      "Run ungrib.exe a third time for soil files with prefix = 'SOILFILE'.",
      "Restore fg_name = 'SFILE', 'SOILFILE', 'FILE' and run metgrid.exe.",
    ],
    notes: [
      "Use the Free and Open Data Portal for the normal hobbyist real-time path. That route usually does not need an API key.",
      "If recent ECMWF GRIB2 files fail in ungrib.exe, rewrite them first with grib_set to change packingType from grid_ccsds to grid_simple.",
      "The repo starter pack and the wizard default both follow this split-prefix ECMWF pattern.",
    ],
  },
  gfs: {
    id: "gfs",
    label: "GFS via NOMADS",
    shortLabel: "GFS",
    defaultIntervalSeconds: 3600,
    vtable: "Vtable.GFS",
    numMetgridLevels: 34,
    numMetgridSoilLevels: 4,
    splitPrefixes: false,
    steps: [
      "Point Vtable at the stock WPS Vtable.GFS file.",
      "Link the GFS GRIB2 files into GRIBFILE.* names.",
      "Keep prefix = 'FILE', run ungrib.exe once, then run metgrid.exe.",
    ],
    notes: [
      "This is the simplest public fallback for most users.",
      "If you download a 3-hourly GFS product instead of hourly files, change interval_seconds to 10800 in both namelists.",
    ],
  },
  hrrr: {
    id: "hrrr",
    label: "HRRR",
    shortLabel: "HRRR",
    defaultIntervalSeconds: 3600,
    vtable: "Vtable.raphrrr",
    numMetgridLevels: 51,
    numMetgridSoilLevels: 8,
    splitPrefixes: false,
    steps: [
      "Point Vtable at the stock WPS Vtable.raphrrr file.",
      "Link the HRRR GRIB2 files into GRIBFILE.* names.",
      "Keep prefix = 'FILE', run ungrib.exe once, then run metgrid.exe.",
    ],
    notes: [
      "Use HRRR for CONUS short-range storm work, not as the universal beginner default.",
      "The vertical-level defaults here are a practical starter. If real.exe complains, match num_metgrid_* to a met_em header.",
    ],
  },
  rap: {
    id: "rap",
    label: "RAP",
    shortLabel: "RAP",
    defaultIntervalSeconds: 3600,
    vtable: "Vtable.raphrrr",
    numMetgridLevels: 51,
    numMetgridSoilLevels: 8,
    splitPrefixes: false,
    steps: [
      "Point Vtable at the stock WPS Vtable.raphrrr file.",
      "Link the RAP GRIB2 files into GRIBFILE.* names.",
      "Keep prefix = 'FILE', run ungrib.exe once, then run metgrid.exe.",
    ],
    notes: [
      "Use RAP when you want broader fast-refresh coverage than HRRR.",
      "The vertical-level defaults here are a practical starter. If real.exe complains, match num_metgrid_* to a met_em header.",
    ],
  },
};

export function listSourceProfiles() {
  return Object.values(SOURCE_PROFILES);
}

export function getSourceProfile(id) {
  return SOURCE_PROFILES[id] || SOURCE_PROFILES.ecmwf;
}

export function buildSourcePlan(caseConfig) {
  const source = getSourceProfile(caseConfig.source);
  return {
    ...source,
    fgName: source.splitPrefixes ? "'SFILE', 'SOILFILE', 'FILE'" : "'FILE'",
    notes: [...source.notes],
    steps: [...source.steps],
  };
}
