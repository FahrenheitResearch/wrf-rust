import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const PAGES = [
  "index.html",
  "diagnostics.html",
  "validation.html",
  "performance.html",
  "changelog.html",
];

async function read(name) {
  return fs.readFile(path.join(docsDir, name), "utf8");
}

test("variables.json is valid and self-consistent", async () => {
  const data = JSON.parse(await read(path.join("data", "variables.json")));
  assert.equal(typeof data.count, "number");
  assert.ok(Array.isArray(data.variables));
  assert.equal(
    data.variables.length,
    data.count,
    "count field must match array length",
  );
  const names = new Set();
  for (const v of data.variables) {
    assert.ok(v.name && v.description && v.units && v.dim && v.group, v.name);
    assert.ok(!names.has(v.name), `duplicate name ${v.name}`);
    names.add(v.name);
  }
});

test("diagnostics table contains exactly one row per registry entry", async () => {
  const data = JSON.parse(await read(path.join("data", "variables.json")));
  const html = await read("diagnostics.html");
  const start = html.indexOf("<!-- diagnostics-table:start -->");
  const end = html.indexOf("<!-- diagnostics-table:end -->");
  assert.ok(start !== -1 && end > start, "table markers present");
  const body = html.slice(start, end);
  const rows = [...body.matchAll(/<tr>/g)].length;
  assert.equal(rows, data.count, `table rows (${rows}) must equal registry count`);
  for (const v of data.variables) {
    assert.ok(
      body.includes(`<td><code>${v.name}</code></td>`),
      `missing row for ${v.name}`,
    );
  }
});

test("stated variable counts on the pages match the registry", async () => {
  const data = JSON.parse(await read(path.join("data", "variables.json")));
  const diagnostics = await read("diagnostics.html");
  const index = await read("index.html");
  assert.ok(
    diagnostics.includes(`<strong>${data.count}</strong>`),
    "diagnostics.html lede count",
  );
  assert.ok(
    diagnostics.includes(`${data.count} of ${data.count} variables`),
    "diagnostics.html row-count label",
  );
  assert.ok(
    index.includes(`<td class="num">${data.count}</td>`),
    "index.html facts table count",
  );
});

test("every internal link and asset on the site pages resolves", async () => {
  for (const page of PAGES) {
    const html = await read(page);
    const refs = [
      ...html.matchAll(/(?:href|src)="([^"]+)"/g),
    ].map((m) => m[1]);
    for (const ref of refs) {
      if (/^(https?:|mailto:)/.test(ref)) continue;
      const target = ref.split("#")[0];
      if (target === "") {
        // Pure fragment: must resolve to an id on the same page.
        const id = ref.slice(1);
        assert.ok(html.includes(`id="${id}"`), `${page}: missing #${id}`);
        continue;
      }
      await assert.doesNotReject(
        fs.access(path.join(docsDir, target)),
        `${page}: broken reference ${ref}`,
      );
    }
  }
});

test("all pages share nav, footer, and mark the current page", async () => {
  for (const page of PAGES) {
    const html = await read(page);
    assert.ok(html.includes('nav class="site-nav"'), `${page}: nav`);
    assert.ok(
      html.includes(`<a href="${page}" aria-current="page">`),
      `${page}: aria-current marker`,
    );
    assert.ok(
      html.includes('href="community-guide/index.html"'),
      `${page}: community guide footer link`,
    );
    for (const other of PAGES) {
      assert.ok(html.includes(`href="${other}"`), `${page}: nav link to ${other}`);
    }
  }
});

test("site pages contain no emoji", async () => {
  // House rule: no decorative emojis. Covers pictographs, symbols-and-
  // pictographs blocks, dingbats, and variation selector-16.
  const emoji = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u;
  for (const page of PAGES) {
    const html = await read(page);
    assert.ok(!emoji.test(html), `${page}: emoji found`);
  }
});
