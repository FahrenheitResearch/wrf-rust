import { normalizeCaseConfig } from "./case-config.mjs";
import { renderNamelistInput, renderNamelistWps } from "./namelist-render.mjs";
import { applyPreset, getPreset } from "./presets.mjs";
import { buildSupportTemplate, evaluatePreflight } from "./preflight-core.mjs";
import { recommendationForRam } from "./sizing.mjs";
import { buildSourcePlan, getSourceProfile, listSourceProfiles } from "./source-profiles.mjs";
import { validateCase } from "./validate-case.mjs";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function yesNo(value) {
  return value ? "yes" : "no";
}

function statusLabel(status) {
  return status === "fail" ? "Blocker" : status === "warn" ? "Watch" : "Good";
}

function renderMessages(messages, emptyText) {
  if (!messages.length) {
    return `<p class="small">${escapeHtml(emptyText)}</p>`;
  }

  return `<ul class="checklist">${messages
    .map((message) => `<li>${escapeHtml(message.text)}</li>`)
    .join("")}</ul>`;
}

function copyText(text, button) {
  const finish = (label) => {
    const original = button.dataset.originalLabel || button.textContent;
    button.dataset.originalLabel = original;
    button.textContent = label;
    window.setTimeout(() => {
      button.textContent = original;
    }, 1400);
  };

  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(() => finish("Copied"), () => finish("Copy failed"));
    return;
  }

  const input = document.createElement("textarea");
  input.value = text;
  input.setAttribute("readonly", "readonly");
  input.style.position = "absolute";
  input.style.left = "-9999px";
  document.body.appendChild(input);
  input.select();
  try {
    document.execCommand("copy");
    finish("Copied");
  } catch {
    finish("Copy failed");
  } finally {
    document.body.removeChild(input);
  }
}

export function initApp() {
  const state = {
    presetId: "starter-32",
    platform: "windows",
    ramGb: 32,
    diskGb: 200,
    workspaceLocation: "linux-home",
    hasWsl: true,
    wslVersion: 2,
    hasVirt: true,
    wslConfigApplied: true,
    hasWrf: false,
    hasWps: false,
    hasGeog: false,
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
  };

  const els = {
    ramRange: document.querySelector("#ram-range"),
    ramNumber: document.querySelector("#ram-number"),
    presetSummary: document.querySelector("#preset-summary"),
    presetCards: [...document.querySelectorAll("[data-preset]")],
    outMemory: document.querySelector("#out-memory"),
    outProcessors: document.querySelector("#out-processors"),
    outSwap: document.querySelector("#out-swap"),
    outDefault: document.querySelector("#out-default"),
    outAggressive: document.querySelector("#out-aggressive"),
    outTone: document.querySelector("#out-tone"),
    wslConfigBlock: document.querySelector("#wsl-config-block"),
    preflightPlatform: document.querySelector("#pf-platform"),
    preflightDisk: document.querySelector("#pf-disk"),
    preflightWorkspace: document.querySelector("#pf-workspace"),
    preflightHasWsl: document.querySelector("#pf-has-wsl"),
    preflightWslVersion: document.querySelector("#pf-wsl-version"),
    preflightHasVirt: document.querySelector("#pf-has-virt"),
    preflightWslConfigApplied: document.querySelector("#pf-wsl-config"),
    preflightHasWrf: document.querySelector("#pf-has-wrf"),
    preflightHasWps: document.querySelector("#pf-has-wps"),
    preflightHasGeog: document.querySelector("#pf-has-geog"),
    preflightSummary: document.querySelector("#preflight-summary"),
    preflightBadge: document.querySelector("#preflight-badge"),
    preflightStatusGrid: document.querySelector("#preflight-status-grid"),
    preflightConfigCard: document.querySelector("#preflight-config-card"),
    preflightWslConfig: document.querySelector("#preflight-wsl-config"),
    supportTemplate: document.querySelector("#support-template"),
    sourceSelect: document.querySelector("#wiz-source"),
    startUtc: document.querySelector("#wiz-start-utc"),
    runHours: document.querySelector("#wiz-run-hours"),
    historyMinutes: document.querySelector("#wiz-history-minutes"),
    eWe: document.querySelector("#wiz-e-we"),
    eSn: document.querySelector("#wiz-e-sn"),
    centerLat: document.querySelector("#wiz-center-lat"),
    centerLon: document.querySelector("#wiz-center-lon"),
    geogDataPath: document.querySelector("#wiz-geog-data-path"),
    numMetgridLevels: document.querySelector("#wiz-num-metgrid-levels"),
    numMetgridSoilLevels: document.querySelector("#wiz-num-metgrid-soil-levels"),
    wizardSource: document.querySelector("#wizard-current-source"),
    wizardPreset: document.querySelector("#wizard-current-preset"),
    wizardSummary: document.querySelector("#wizard-summary"),
    wizardProjection: document.querySelector("#wizard-projection"),
    wizardValidation: document.querySelector("#wizard-validation"),
    wizardVtable: document.querySelector("#wizard-vtable"),
    wizardSourceSteps: document.querySelector("#wizard-source-steps"),
    wizardSourceNotes: document.querySelector("#wizard-source-notes"),
    generatedWps: document.querySelector("#generated-wps"),
    generatedInput: document.querySelector("#generated-input"),
    copyWsl: document.querySelector("#copy-wsl"),
    copySupport: document.querySelector("#copy-support"),
    copyWps: document.querySelector("#copy-wps"),
    copyInput: document.querySelector("#copy-input"),
    downloadWps: document.querySelector("#download-wps"),
    downloadInput: document.querySelector("#download-input"),
    windowsOnly: [...document.querySelectorAll(".windows-only")],
  };

  for (const source of listSourceProfiles()) {
    const option = document.createElement("option");
    option.value = source.id;
    option.textContent = source.label;
    els.sourceSelect?.appendChild(option);
  }

  const downloadUrls = {
    wps: null,
    input: null,
  };

  function updateDownloadLink(anchor, key, filename, text) {
    if (!anchor) return;
    if (downloadUrls[key]) {
      URL.revokeObjectURL(downloadUrls[key]);
    }
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    downloadUrls[key] = url;
    anchor.href = url;
    anchor.download = filename;
  }

  function syncFormValues() {
    if (els.ramRange) els.ramRange.value = String(state.ramGb);
    if (els.ramNumber) els.ramNumber.value = String(state.ramGb);
    if (els.preflightPlatform) els.preflightPlatform.value = state.platform;
    if (els.preflightDisk) els.preflightDisk.value = String(state.diskGb);
    if (els.preflightWorkspace) els.preflightWorkspace.value = state.workspaceLocation;
    if (els.preflightHasWsl) els.preflightHasWsl.value = yesNo(state.hasWsl);
    if (els.preflightWslVersion) els.preflightWslVersion.value = String(state.wslVersion);
    if (els.preflightHasVirt) els.preflightHasVirt.value = yesNo(state.hasVirt);
    if (els.preflightWslConfigApplied) els.preflightWslConfigApplied.value = yesNo(state.wslConfigApplied);
    if (els.preflightHasWrf) els.preflightHasWrf.value = yesNo(state.hasWrf);
    if (els.preflightHasWps) els.preflightHasWps.value = yesNo(state.hasWps);
    if (els.preflightHasGeog) els.preflightHasGeog.value = yesNo(state.hasGeog);
    if (els.sourceSelect) els.sourceSelect.value = state.source;
    if (els.startUtc) els.startUtc.value = state.startUtc;
    if (els.runHours) els.runHours.value = String(state.runHours);
    if (els.historyMinutes) els.historyMinutes.value = String(state.historyMinutes);
    if (els.eWe) els.eWe.value = String(state.e_we);
    if (els.eSn) els.eSn.value = String(state.e_sn);
    if (els.centerLat) els.centerLat.value = String(state.centerLat);
    if (els.centerLon) els.centerLon.value = String(state.centerLon);
    if (els.geogDataPath) els.geogDataPath.value = state.geogDataPath;
    if (els.numMetgridLevels) els.numMetgridLevels.value = String(state.numMetgridLevels);
    if (els.numMetgridSoilLevels) els.numMetgridSoilLevels.value = String(state.numMetgridSoilLevels);
  }

  function render() {
    syncFormValues();

    const preset = getPreset(state.presetId);
    const rec = recommendationForRam(state.ramGb);
    const caseConfig = normalizeCaseConfig(state);
    const validation = validateCase(caseConfig);
    const preflight = evaluatePreflight({
      ...state,
      ...caseConfig,
      hasWsl: state.hasWsl,
      hasVirt: state.hasVirt,
      hasWrf: state.hasWrf,
      hasWps: state.hasWps,
      hasGeog: state.hasGeog,
      wslVersion: state.wslVersion,
      wslConfigApplied: state.wslConfigApplied,
    });
    const sourcePlan = buildSourcePlan(caseConfig);
    const wslConfig = `[wsl2]
memory=${rec.memory}GB
processors=${rec.processors}
swap=${rec.swap}GB`;
    const namelistWps = renderNamelistWps(caseConfig);
    const namelistInput = renderNamelistInput(caseConfig);

    els.outMemory.textContent = `${rec.memory} GB`;
    els.outProcessors.textContent = `${rec.processors} threads`;
    els.outSwap.textContent = `${rec.swap} GB`;
    els.outDefault.textContent = rec.default3km;
    els.outAggressive.textContent = rec.aggressive3km;
    els.outTone.textContent = rec.tone;
    els.wslConfigBlock.textContent = wslConfig;
    els.presetSummary.textContent = `${preset.label}: ${preset.description}`;

    for (const card of els.presetCards) {
      card.classList.toggle("is-active", card.dataset.preset === state.presetId);
    }

    for (const block of els.windowsOnly) {
      block.classList.toggle("is-hidden", state.platform !== "windows");
    }

    els.preflightBadge.textContent = preflight.overall.label;
    els.preflightBadge.className = `tag status-tag status-${preflight.overall.status}`;
    els.preflightSummary.textContent = preflight.overall.detail;
    els.preflightStatusGrid.innerHTML = preflight.items
      .map((entry) => `
        <article class="card status-card status-${entry.status}">
          <div class="tag-row">
            <span class="tag status-tag status-${entry.status}">${statusLabel(entry.status)}</span>
          </div>
          <h3>${escapeHtml(entry.title)}</h3>
          <p>${escapeHtml(entry.detail)}</p>
          <p><strong>Next action:</strong> ${escapeHtml(entry.action)}</p>
        </article>
      `)
      .join("");

    if (els.preflightConfigCard) {
      els.preflightConfigCard.classList.toggle("is-hidden", state.platform !== "windows");
    }
    if (els.preflightWslConfig) {
      els.preflightWslConfig.textContent = wslConfig;
    }

    const supportTemplate = buildSupportTemplate({
      ...caseConfig,
      platform: state.platform,
      diskGb: state.diskGb,
      hasWrf: state.hasWrf,
      hasWps: state.hasWps,
      hasGeog: state.hasGeog,
      wslVersion: state.wslVersion,
      workspaceLocation: state.workspaceLocation,
    });
    els.supportTemplate.textContent = supportTemplate;

    els.wizardSource.textContent = sourcePlan.label;
    els.wizardPreset.textContent = preset.label;
    els.wizardSummary.textContent = `${caseConfig.runHours}-hour ${sourcePlan.shortLabel} case, ${caseConfig.eWe} x ${caseConfig.eSn} at 3 km, centered on ${caseConfig.centerLat.toFixed(1)}, ${caseConfig.centerLon.toFixed(1)}.`;
    els.wizardProjection.textContent = `Projection: Lambert, truelat1 ${caseConfig.truelat1.toFixed(1)}, truelat2 ${caseConfig.truelat2.toFixed(1)}, stand_lon ${caseConfig.standLon.toFixed(1)}. Use UTC in the start-time field; the wizard does not convert local timezones for you.`;
    els.wizardValidation.innerHTML = `
      <div class="validation-block">
        <h4>Errors</h4>
        ${renderMessages(validation.errors, "No hard configuration errors are showing.")}
      </div>
      <div class="validation-block">
        <h4>Warnings</h4>
        ${renderMessages(validation.warnings, "No extra warnings beyond the normal starter cautions.")}
      </div>
    `;
    els.wizardVtable.textContent = sourcePlan.vtable;
    els.wizardSourceSteps.innerHTML = sourcePlan.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("");
    els.wizardSourceNotes.innerHTML = `<ul class="checklist">${sourcePlan.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`;
    els.generatedWps.textContent = namelistWps;
    els.generatedInput.textContent = namelistInput;
    updateDownloadLink(els.downloadWps, "wps", "namelist.wps", namelistWps);
    updateDownloadLink(els.downloadInput, "input", "namelist.input", namelistInput);

    els.copyWsl.onclick = () => copyText(wslConfig, els.copyWsl);
    els.copySupport.onclick = () => copyText(supportTemplate, els.copySupport);
    els.copyWps.onclick = () => copyText(namelistWps, els.copyWps);
    els.copyInput.onclick = () => copyText(namelistInput, els.copyInput);
  }

  function bindNumberInput(element, key) {
    element?.addEventListener("input", () => {
      const value = parseInt(element.value || "", 10);
      if (Number.isFinite(value)) {
        state[key] = value;
        render();
      }
    });
  }

  function bindTextInput(element, key) {
    element?.addEventListener("input", () => {
      state[key] = element.value;
      render();
    });
  }

  function bindSelectInput(element, key) {
    element?.addEventListener("change", () => {
      state[key] = element.value;
      render();
    });
  }

  function bindYesNoSelect(element, key) {
    element?.addEventListener("change", () => {
      state[key] = element.value === "yes";
      render();
    });
  }

  els.ramRange?.addEventListener("input", () => {
    const value = parseInt(els.ramRange.value || "", 10);
    if (Number.isFinite(value)) {
      state.ramGb = value;
      render();
    }
  });

  els.ramNumber?.addEventListener("input", () => {
    const value = parseInt(els.ramNumber.value || "", 10);
    if (Number.isFinite(value)) {
      state.ramGb = value;
      render();
    }
  });

  els.preflightPlatform?.addEventListener("change", () => {
    state.platform = els.preflightPlatform.value;
    render();
  });

  bindNumberInput(els.preflightDisk, "diskGb");
  bindSelectInput(els.preflightWorkspace, "workspaceLocation");
  bindYesNoSelect(els.preflightHasWsl, "hasWsl");
  els.preflightWslVersion?.addEventListener("change", () => {
    state.wslVersion = parseInt(els.preflightWslVersion.value || "2", 10);
    render();
  });
  bindYesNoSelect(els.preflightHasVirt, "hasVirt");
  bindYesNoSelect(els.preflightWslConfigApplied, "wslConfigApplied");
  bindYesNoSelect(els.preflightHasWrf, "hasWrf");
  bindYesNoSelect(els.preflightHasWps, "hasWps");
  bindYesNoSelect(els.preflightHasGeog, "hasGeog");
  els.sourceSelect?.addEventListener("change", () => {
    const profile = getSourceProfile(els.sourceSelect.value);
    state.source = profile.id;
    state.numMetgridLevels = profile.numMetgridLevels;
    state.numMetgridSoilLevels = profile.numMetgridSoilLevels;
    render();
  });
  bindTextInput(els.startUtc, "startUtc");
  bindNumberInput(els.runHours, "runHours");
  bindNumberInput(els.historyMinutes, "historyMinutes");
  bindNumberInput(els.eWe, "e_we");
  bindNumberInput(els.eSn, "e_sn");
  bindTextInput(els.centerLat, "centerLat");
  bindTextInput(els.centerLon, "centerLon");
  bindTextInput(els.geogDataPath, "geogDataPath");
  bindNumberInput(els.numMetgridLevels, "numMetgridLevels");
  bindNumberInput(els.numMetgridSoilLevels, "numMetgridSoilLevels");

  for (const card of els.presetCards) {
    card.addEventListener("click", () => {
      Object.assign(state, applyPreset(state, card.dataset.preset));
      render();
    });
  }

  document.querySelectorAll("[data-jump]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.querySelector(button.getAttribute("data-jump"));
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  render();
}
