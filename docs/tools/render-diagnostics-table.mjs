// Regenerate the static diagnostics table in docs/diagnostics.html from
// docs/data/variables.json (itself produced by
// `cargo run -p wrf-core --example dump_variables -- <sha>`).
//
// Usage: node docs/tools/render-diagnostics-table.mjs

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = path.join(docsDir, "data", "variables.json");
const pagePath = path.join(docsDir, "diagnostics.html");

const START = "<!-- diagnostics-table:start -->";
const END = "<!-- diagnostics-table:end -->";

function esc(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const data = JSON.parse(await fs.readFile(dataPath, "utf8"));
if (!Array.isArray(data.variables) || data.variables.length !== data.count) {
  throw new Error(
    `variables.json count field (${data.count}) does not match array length (${data.variables?.length})`,
  );
}

const rows = data.variables
  .map((v) => {
    const aliases = v.aliases.length
      ? v.aliases.map((a) => `<code>${esc(a)}</code>`).join(", ")
      : "&#8212;";
    return [
      "    <tr>",
      `      <td><code>${esc(v.name)}</code></td>`,
      `      <td>${aliases}</td>`,
      `      <td>${esc(v.description)}</td>`,
      `      <td>${esc(v.units)}</td>`,
      `      <td>${esc(v.dim.split(" ")[0])}</td>`,
      `      <td>${esc(v.group)}</td>`,
      "    </tr>",
    ].join("\n");
  })
  .join("\n");

const page = await fs.readFile(pagePath, "utf8");
const startIdx = page.indexOf(START);
const endIdx = page.indexOf(END);
if (startIdx === -1 || endIdx === -1 || endIdx < startIdx) {
  throw new Error("table markers not found in diagnostics.html");
}

const next =
  page.slice(0, startIdx + START.length) + "\n" + rows + "\n" + page.slice(endIdx);
await fs.writeFile(pagePath, next);
console.log(
  `wrote ${data.count} rows into diagnostics.html (registry @ ${data.git_commit})`,
);
