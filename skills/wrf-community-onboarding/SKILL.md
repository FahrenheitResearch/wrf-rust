---
name: wrf-community-onboarding
description: Help weather-hobbyist and Discord users install, size, initialize, run, and troubleshoot WRF on consumer hardware, especially Windows users on WSL 2. Use this when the user needs CPU-first WRF/WPS setup guidance, ECMWF/GFS/HRRR/RAP initialization advice, domain sizing for 3 km convection-allowing runs, or branching troubleshooting for compile, WPS, real.exe, wrf.exe, or disk issues.
---

# WRF Community Onboarding

Use this skill for hobbyist or community WRF support, especially when the user:

- is on Windows and needs WSL 2
- wants a practical 3 km severe-weather setup on consumer hardware
- needs help choosing ECMWF, GFS, HRRR, or RAP init data
- needs a decision tree for build, prep, runtime, or disk failures

This skill is intentionally biased toward getting the user to a stable first run quickly, not toward the fanciest possible configuration.

## Core stance

- Prefer a stable CPU-first path.
- For Windows users, prefer WSL 2 and keep all WRF/WPS files inside the Linux filesystem, not `/mnt/c/...`.
- Push users toward a small sanity run before a large real-data storm run.
- Be explicit about disk budgeting. Hobbyist users usually get surprised by output size before they get surprised by CPU time.
- Clearly separate verified facts from inference when discussing specs, ECMWF access, or platform limits.

## Hard rules

- If the user says "Windows", load [windows-wsl.md](references/windows-wsl.md).
- If the user needs help deciding whether their hardware is enough, load [specs-and-paths.md](references/specs-and-paths.md).
- If the user needs a practical WRF/WPS workflow, load [build-and-run.md](references/build-and-run.md).
- If the user asks about ECMWF, GFS, HRRR, or RAP, load [init-data.md](references/init-data.md).
- If the user is already failing somewhere, load [troubleshooting.md](references/troubleshooting.md).

## Workflow

1. Classify the user.
   Determine:
   - OS: Windows + WSL, Linux, or something unsupported
   - Host RAM
   - CPU threads
   - disk free space
   - target region
   - preferred init data
   - whether they want a first successful run or a maxed-out build

2. Put them on the right path.
   Default path:
   - Windows user -> WSL 2 + CPU-first + single-domain 3 km
   - Linux user -> CPU-first + single-domain 3 km

3. Recommend a domain tier.
   Use the spec tiers in [specs-and-paths.md](references/specs-and-paths.md).
   Default recommendation for hobbyist severe-weather users:
   - single-domain 3 km
   - around 200 to 250 grid points each direction for a low-risk starter
   - around 300 to 350 each direction only if RAM and disk headroom are clearly there
   - 400 x 400 class domains are aggressive and should be framed that way

4. Choose init data.
   Use [init-data.md](references/init-data.md).
   Default order:
   - ECMWF open data for hobbyist real-time Euro/IFS initialization
   - GFS for universal fallback
   - HRRR or RAP only when the region and use case justify them

5. Guide the actual workflow.
   Use [build-and-run.md](references/build-and-run.md).
   The stable pipeline is:
   - install WSL 2 if needed
   - install compilers and libraries
   - compile WRF first
   - compile WPS second
   - run `geogrid.exe`
   - run `ungrib.exe`
   - run `metgrid.exe`
   - run `real.exe`
   - run `wrf.exe`
   - sanity-check output with `wrf-rust`

6. Troubleshoot by phase, not by vague symptoms.
   Use [troubleshooting.md](references/troubleshooting.md).
   Start with the earliest failing phase:
   - compile
   - WPS
   - `real.exe`
   - `wrf.exe` startup
   - numerical stability
   - disk exhaustion

## Communication guidance

- Use exact dates when clarifying ECMWF access rules or any "current" platform guidance.
- For ECMWF, explicitly state that as of `2026-04-01` the simplest real-time open-data path is the Free & Open Data Portal, and hobbyist users usually do not need an API key for that path.
- When recommending against a user plan, replace it with a better one immediately.
  Example:
  - not "1 km will be hard"
  - but "start with 3 km 220 x 220, confirm the pipeline, then nest or tighten resolution later"
- If the user is on 16 GB RAM and wants a giant 3 km domain, say plainly that they are below the pleasant operating range.

## Default response pattern

When the user needs setup help, structure the answer as:

1. Recommended path
2. Exact first commands or actions
3. What to check before moving on
4. The next branch if that step fails

When the user needs troubleshooting, structure the answer as:

1. What phase failed
2. Most likely root causes
3. Fastest checks
4. Minimal corrective actions

## Do not do this

- Do not recommend `/mnt/c/...` for the main build or run tree.
- Do not tell users to start with the largest domain their RAM might barely fit.
- Do not bury the fact that WSL defaults to only half the host RAM.
- Do not imply ECMWF open-data real-time access requires the same credentials as CDS/archive access.
