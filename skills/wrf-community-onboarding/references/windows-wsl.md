# Windows And WSL 2

Use this file whenever the user is on Windows.

## Verified current baseline

As of `2026-04-01`, Microsoft documents the current `wsl --install` flow for:

- `Windows 11`
- `Windows 10 version 2004+`, build `19041+`

## Recommended install flow

Run in elevated PowerShell:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --status
wsl -l -v
```

If `wsl --install` only shows help, WSL may already be present:

```powershell
wsl --list --online
wsl --install -d Ubuntu
```

If installation stalls at `0.0%`, Microsoft documents:

```powershell
wsl --install --web-download -d Ubuntu
```

## Non-negotiable WRF advice

- Use `WSL 2`, not WSL 1.
- Keep the WRF/WPS tree in the Linux filesystem:
  - good: `/home/<user>/wrf-work`
  - bad: `/mnt/c/Users/<user>/wrf-work`
- Access files from Windows through `\\wsl$` or `explorer.exe .`

## Why WSL surprises hobbyist users

By default, WSL 2 does not just "use all your RAM."
Microsoft documents defaults roughly as:

- memory: `50%` of host RAM
- processors: all logical processors
- swap: `25%` of host RAM, rounded up

That means a `32 GB` Windows box may only give WSL about `16 GB` unless the user sets `.wslconfig`.

## Recommended `.wslconfig` starting points

File location:

- `%UserProfile%\.wslconfig`

### 32 GB host

```ini
[wsl2]
memory=24GB
processors=12
swap=16GB
```

### 64 GB host

```ini
[wsl2]
memory=48GB
processors=16
swap=16GB
```

### 16 GB host

```ini
[wsl2]
memory=10GB
processors=6
swap=8GB
```

Then apply:

```powershell
wsl --shutdown
```

Wait a few seconds before reopening Ubuntu. Microsoft notes an "8 second rule" in practice: config changes do not fully apply until WSL actually stops.

## Common documented failure modes

### `0x80370102`

Usually means:

- BIOS/UEFI virtualization is off
- Virtual Machine Platform is missing
- hypervisor launch is disabled

Fast checks:

- enable virtualization in BIOS
- run `systeminfo.exe`
- run:

```powershell
bcdedit /enum | findstr -i hypervisorlaunchtype
```

If disabled:

```powershell
bcdedit /set hypervisorlaunchtype Auto
```

### `0x8007019e`

Usually means the Windows Subsystem for Linux feature itself is not enabled.

Fast fix:

- enable the WSL Windows feature
- reboot
- retry the install

### `0x80070003`

Microsoft documents that WSL must run from the system drive, usually `C:`.

### Compressed or encrypted distro storage

WSL can fail if the distro `LocalState` directory is compressed or encrypted on NTFS.

### `.wslconfig` changes do not seem to apply

Common cause:

- the user edited `%UserProfile%\.wslconfig` but never fully stopped WSL

Fast fix:

```powershell
wsl --shutdown
```

Then wait a few seconds and relaunch Ubuntu.

### WSL has no internet or DNS fails

Common on managed PCs, VPN-heavy setups, and some firewall configurations.

Fast first move:

- `wsl --shutdown`
- relaunch the distro
- if still broken, check Microsoft's WSL troubleshooting guidance around DNS tunneling and firewall behavior
