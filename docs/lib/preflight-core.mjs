import { buildSourcePlan } from "./source-profiles.mjs";
import { recommendationForRam } from "./sizing.mjs";

function item(status, title, detail, action) {
  return { status, title, detail, action };
}

export function evaluatePreflight(profile) {
  const ramPlan = recommendationForRam(profile.ramGb);
  const sourcePlan = buildSourcePlan(profile);
  const targetPoints = Math.max(profile.eWe, profile.eSn);
  const items = [];

  if (profile.platform === "windows") {
    if (!profile.hasWsl) {
      items.push(item("fail", "WSL base", "WSL is not installed yet.", "Run wsl --install -d Ubuntu from an elevated PowerShell, then update and reopen the distro."));
    } else if (Number(profile.wslVersion) !== 2) {
      items.push(item("fail", "WSL version", "The distro is not on WSL 2.", "Run wsl -l -v, then convert the distro with wsl --set-version <DistroName> 2."));
    } else if (!profile.hasVirt) {
      items.push(item("fail", "Virtualization", "BIOS or Windows virtualization support is still missing.", "Turn on virtualization in BIOS and confirm the Windows virtualization features are enabled."));
    } else {
      items.push(item("pass", "WSL base", "WSL 2 and virtualization look sane.", "Keep working inside Ubuntu, not inside PowerShell."));
    }
  } else {
    items.push(item("pass", "Platform base", "Linux path selected. No WSL layer is required.", "Keep the project inside the Linux filesystem and continue with the normal build path."));
  }

  if (profile.workspaceLocation === "windows-drive") {
    items.push(item("fail", "Workspace path", "The project is sitting on /mnt/c or another Windows-mounted path.", "Move the workflow under /home/<user> and access it from Windows through \\\\wsl$."));
  } else {
    items.push(item("pass", "Workspace path", "The workflow is pointed at the Linux filesystem.", "Keep the repo, WPS_GEOG, and run directories there."));
  }

  if (profile.platform === "windows" && !profile.wslConfigApplied) {
    items.push(item("warn", "WSL memory", "WSL is probably still using the default half-RAM limit.", "Drop the generated .wslconfig into %UserProfile%, then run wsl --shutdown before reopening Ubuntu."));
  } else if (profile.platform === "windows") {
    items.push(item("pass", "WSL memory", "A custom .wslconfig is in play.", "Keep that file aligned with the RAM calculator above."));
  }

  if (profile.diskGb < 80) {
    items.push(item("fail", "Disk headroom", "There is not enough free disk for geog data, builds, and outputs.", "Free at least 80 GB before you start staging geography and met_em files."));
  } else if (profile.diskGb < 150) {
    items.push(item("warn", "Disk headroom", "The disk is workable for a short case, but output files will become the limiter fast.", "Keep the run short and the history interval at 60 minutes until you have more room."));
  } else {
    items.push(item("pass", "Disk headroom", "Disk space looks sane for a short first case.", "Still check df -h before launching the real run."));
  }

  if (profile.ramGb < 16) {
    items.push(item("fail", "Spec tier", "This is below the practical floor for the workflow documented here.", "Use a larger machine or keep to analysis and visualization instead of full local WRF runs."));
  } else if (targetPoints > ramPlan.aggressiveMax) {
    items.push(item("fail", "Domain target", `The requested ${targetPoints} x ${targetPoints} 3 km domain is beyond the aggressive ceiling for ${profile.ramGb} GB.`, `Drop the domain closer to ${ramPlan.default3km} or choose a smaller preset.`));
  } else if (targetPoints > ramPlan.defaultMax || profile.ramGb < 24) {
    items.push(item("warn", "Domain target", `The target domain is above the comfortable range for ${profile.ramGb} GB.`, `Use the ${ramPlan.default3km} range for a calmer first run.`));
  } else {
    items.push(item("pass", "Domain target", `The current ${targetPoints} x ${targetPoints} target fits the ${profile.ramGb} GB tier reasonably well.`, "Keep the first run short before you scale up."));
  }

  if (!profile.hasWrf || !profile.hasWps) {
    const missing = [
      !profile.hasWrf ? "WRF" : null,
      !profile.hasWps ? "WPS" : null,
    ].filter(Boolean).join(" and ");
    items.push(item("warn", "Compiled binaries", `${missing} still needs to be built.`, "Build CPU WRF first, then WPS, and do not move on until wrf.exe, real.exe, geogrid.exe, ungrib.exe, and metgrid.exe exist."));
  } else {
    items.push(item("pass", "Compiled binaries", "WRF and WPS are both marked as built.", "Go straight into geogrid.exe, then the source-specific ungrib/metgrid flow."));
  }

  if (!profile.hasGeog) {
    items.push(item("warn", "Static geography", "WPS geographical data is still missing.", "Download the mandatory WPS geog package and point geog_data_path at the Linux-side directory."));
  } else {
    items.push(item("pass", "Static geography", "WPS geographical data is marked present.", "Double-check the geog_data_path in the generated namelist.wps before running geogrid.exe."));
  }

  items.push(item("pass", "Source plan", `${sourcePlan.label} is selected with ${sourcePlan.vtable}.`, sourcePlan.steps[0]));

  const failCount = items.filter((entry) => entry.status === "fail").length;
  const warnCount = items.filter((entry) => entry.status === "warn").length;
  let overall = {
    status: "pass",
    label: "Ready",
    detail: "The baseline looks sane for a short single-domain run.",
  };

  if (failCount > 0) {
    overall = {
      status: "fail",
      label: "Blocked",
      detail: `Fix ${failCount} blocker${failCount === 1 ? "" : "s"} before you spend more time on compile or data prep.`,
    };
  } else if (!profile.hasWrf || !profile.hasWps || !profile.hasGeog) {
    overall = {
      status: "warn",
      label: "Build next",
      detail: "The platform looks sane enough. Finish the missing prep stages before you try a real case.",
    };
  } else if (warnCount > 0) {
    overall = {
      status: "warn",
      label: "Needs work",
      detail: "You can keep moving, but a few fixes now will save a bad run later.",
    };
  }

  return { overall, items, ramPlan, sourcePlan };
}

export function buildSupportTemplate(profile) {
  const sourcePlan = buildSourcePlan(profile);
  const platformLabel = profile.platform === "windows"
    ? `Windows + WSL ${profile.wslVersion || "?"}`
    : "Linux";
  const workspaceLabel = profile.workspaceLocation === "windows-drive"
    ? "/mnt/c or another Windows-mounted path"
    : "/home/<user> Linux filesystem";

  return `OS: ${platformLabel}
Host RAM: ${profile.ramGb} GB
Free disk: ${profile.diskGb} GB
Workspace: ${workspaceLabel}
Source: ${sourcePlan.label}
Target domain: ${profile.eWe} x ${profile.eSn} at 3 km
WRF built: ${profile.hasWrf ? "yes" : "no"}
WPS built: ${profile.hasWps ? "yes" : "no"}
WPS geog present: ${profile.hasGeog ? "yes" : "no"}
Last successful step:
Exact error:
`;
}
