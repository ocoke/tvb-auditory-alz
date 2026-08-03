#!/usr/bin/env node
// Package the canonical Data Analytics portable reader with one scoped fix for
// its 100vw sticky-header overflow under non-overlay desktop scrollbars.

import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";


function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]?.replace(/^--/, "");
    const value = argv[index + 1];
    if (!key || !value) throw new Error("Expected paired --key value arguments.");
    values[key] = value;
  }
  for (const key of ["input", "output", "builder", "extractor"]) {
    if (!values[key]) throw new Error(`Missing --${key}.`);
  }
  return values;
}


const options = parseArguments(process.argv.slice(2));
const builderModule = await import(pathToFileURL(options.builder));
const extractorModule = await import(pathToFileURL(options.extractor));
const artifact = JSON.parse(readFileSync(options.input, "utf8"));
let html = builderModule.buildPortableArtifact(artifact);
const temporaryHtml = `${options.output}.pre-static.html`;
writeFileSync(temporaryHtml, html, "utf8");

try {
  const staticCharts = await extractorModule.extractPortableChartSvgs({
    actionTimeoutMs: 5_000,
    htmlPath: temporaryHtml,
    readyTimeoutMs: 10_000,
  });
  html = builderModule.buildPortableArtifact(artifact, { staticCharts });
  const overflowFix = [
    '<style data-tvb379-portable-overflow-fix="true">',
    ".analytics-top-bar{width:100%!important;margin-right:0!important;margin-left:0!important}",
    "@media(max-width:760px){.chart-legend-wrap--bottom{overflow-x:hidden!important}.recharts-default-legend.chart-legend{box-sizing:border-box!important;display:flex!important;flex-wrap:wrap!important;justify-content:center!important;width:100%!important;max-width:100%!important}}",
    "</style>",
  ].join("");
  if (!html.includes("</head>")) throw new Error("Portable HTML has no head terminator.");
  html = html.replace("</head>", `${overflowFix}</head>`);
  writeFileSync(options.output, html, "utf8");
  process.stdout.write(`${JSON.stringify({
    ok: true,
    output: options.output,
    staticCharts: Object.keys(staticCharts).length,
    scopedOverflowFix: true,
  })}\n`);
} finally {
  rmSync(temporaryHtml, { force: true });
}
