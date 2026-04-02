import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsDir = path.resolve(__dirname, "..");

test("all internal section anchors in index.html resolve to an element id", async () => {
  const html = await fs.readFile(path.join(docsDir, "index.html"), "utf8");
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));
  const hrefs = [...html.matchAll(/href="#([^"]+)"/g)].map((match) => match[1]);

  for (const id of hrefs) {
    assert.ok(ids.has(id), `Missing target id for #${id}`);
  }
});

test("all data-jump targets in index.html resolve to an element id", async () => {
  const html = await fs.readFile(path.join(docsDir, "index.html"), "utf8");
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));
  const targets = [...html.matchAll(/data-jump="#([^"]+)"/g)].map((match) => match[1]);

  for (const id of targets) {
    assert.ok(ids.has(id), `Missing data-jump target #${id}`);
  }
});

test("all local linked files in index.html exist", async () => {
  const html = await fs.readFile(path.join(docsDir, "index.html"), "utf8");
  const linkedFiles = [...html.matchAll(/href="(\.\/[^"#]+)"/g)].map((match) => match[1]);

  for (const relativePath of linkedFiles) {
    const fullPath = path.join(docsDir, relativePath.replace(/^\.\//, ""));
    const stat = await fs.stat(fullPath);
    assert.ok(stat.isFile(), `Missing linked file ${relativePath}`);
  }
});

test("the Pages entry script exists", async () => {
  const stat = await fs.stat(path.join(docsDir, "app.js"));
  assert.ok(stat.isFile());
});
