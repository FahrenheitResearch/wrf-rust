export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function round(value, places = 0) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

export function recommendationForRam(ramRaw) {
  const ram = clamp(Math.round(Number(ramRaw) || 32), 8, 192);

  let windowsHeadroom = 6;
  if (ram >= 64) windowsHeadroom = 12;
  else if (ram >= 32) windowsHeadroom = 8;

  const memory = clamp(ram - windowsHeadroom, 8, Math.max(8, ram - 4));
  const processors = ram >= 64 ? 16 : ram >= 32 ? 12 : ram >= 24 ? 8 : 6;
  const swap = ram <= 16 ? 8 : 16;

  let defaultMin = 140;
  let defaultMax = 180;
  let aggressiveMax = 180;
  let tier = "learning";
  let tone =
    "Below the pleasant range for a big severe-weather setup. Keep the domain modest, shorten the run, and expect more compromise.";

  if (ram >= 24 && ram < 48) {
    defaultMin = 220;
    defaultMax = 300;
    aggressiveMax = 300;
    tier = "starter";
    tone =
      "Usable starter tier. Stay CPU-first, keep output intervals conservative, and scale only after a clean short run.";
  } else if (ram >= 48 && ram < 96) {
    defaultMin = 300;
    defaultMax = 350;
    aggressiveMax = 350;
    tier = "sweet-spot";
    tone = "This is the sweet spot for a hobbyist 3 km single domain.";
  } else if (ram >= 96) {
    defaultMin = 350;
    defaultMax = 400;
    aggressiveMax = 400;
    tier = "aggressive";
    tone = "Aggressive hobbyist tier. Disk and output volume now matter as much as RAM.";
  }

  return {
    ram,
    memory,
    processors,
    swap,
    defaultMin,
    defaultMax,
    aggressiveMax,
    default3km: `${defaultMin} to ${defaultMax}`,
    aggressive3km: `around ${aggressiveMax}`,
    tier,
    tone,
  };
}
