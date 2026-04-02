# Specs And Paths

This file gives the practical recommendation ladder for hobbyist WRF users.

## Stable default

Start new users on:

- single-domain `3 km`
- CPU-first
- modest output cadence
- short sanity run before any real event

## Host RAM tiers

These are practical community recommendations, not vendor guarantees.

### 16 GB host RAM

- Below the comfortable range for large real-data convection-allowing WRF on Windows.
- Recommend:
  - tiny learning domains
  - very short runs
  - or using HRRR/RAP instead of self-running WRF

### 24 to 32 GB host RAM

- Viable starter tier.
- Recommend:
  - `3 km`
  - about `200 to 250` grid points each direction
  - conservative output intervals

### 48 to 64 GB host RAM

- Best value tier for hobbyist real-data WRF.
- Recommend:
  - `3 km`
  - about `250 to 350` grid points each direction
  - single domain first
  - one inner nest only after baseline success

### 96 GB and up

- Aggressive hobbyist tier.
- Can explore:
  - `3 km` domains near the `350 to 400` class
  - nested experiments
  - more expensive output

## Width math at 3 km

- `200 x 200` -> about `600 km x 600 km`
- `250 x 250` -> about `750 km x 750 km`
- `300 x 300` -> about `900 km x 900 km`
- `350 x 350` -> about `1050 km x 1050 km`
- `400 x 400` -> about `1200 km x 1200 km`

## Local evidence from this repo owner's runs

- Verified starter case: `3 km`, `200 x 200 x 80`, about `1.86 GB` domain allocation.
- Verified aggressive case: `1 km`, `400 x 400 x 80`, about `6.99 GB` domain allocation.
- Verified extreme-output case: a `401 x 401` idealized stress run produced a single `56 GB` output file.

Use those as sanity anchors when explaining why "it fits in RAM" is not the same as "this is a good beginner choice."

## Recommended branching

- If the user has `16 GB` and wants large 3 km severe-weather WRF, redirect them.
- If the user has `32 GB`, aim for `200 to 250` first.
- If the user has `64 GB`, `300 to 350` becomes realistic.
- If the user wants `400 x 400` at `3 km`, label it aggressive, not normal.
