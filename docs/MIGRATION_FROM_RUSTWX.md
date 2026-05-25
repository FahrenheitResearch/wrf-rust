# Migration From rustwx

This repository takes over the focused WRF product surface. `rustwx` remains the
multi-model research and incubation workspace.

## Migration Strategy

Port small WRF-relevant capabilities into focused crates:

- `rustwx-contour` -> `wrf-contour`
- WRF-relevant rendering concepts from `rustwx-render` -> `wrf-render`
- WRF product recipe ideas from `rustwx-products` -> `wrf-products`
- WRF sounding column and selection ideas from `rustwx-sounding` -> `wrf-sounding`
- wxstore-style local run/file discovery ideas -> `wrf-store`

Avoid depending on the whole `rustwx` workspace. That would preserve the same
agent and maintainer context overload that this split is meant to fix.

## Current Port Status

- `wrf-contour` contains the renderer-agnostic contour topology engine ported
  from `rustwx-contour`, renamed for WRF ownership.
- `wrf-render` contains WRF-native render request, palette, contour-overlay, and
  PNG output primitives. It intentionally accepts data instead of opening WRF
  files.
- `wrf-products` contains initial WRF product recipes and field assembly.
- `wrf-sounding` contains WRF sounding selection and validated column types.
- `wrf-store` contains local run/file path and discovery primitives.

## Backport Guidance

As `wrf-rust` becomes the diagnostic and WRF product source of truth, WRF-specific
paths in `rustwx` should either call into `wrf-rust` or be deprecated in place.
Do not delete unrelated non-WRF `rustwx` capabilities as part of this migration.
