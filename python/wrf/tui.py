"""
wrf-rust Terminal UI.

A clean TUI for browsing WRF files, selecting variables, and exporting
data to files. No ASCII art. Just a practical file/variable picker.

Launch:
    python -m wrf tui [directory_or_file]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Static,
    Button,
    Checkbox,
)
from textual.widgets.option_list import Option

from rich.panel import Panel
from rich.table import Table
from rich import box


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_wrf_files(path: str) -> list[str]:
    """Find wrfout files in a directory, or return [path] if it's a file."""
    if os.path.isfile(path):
        return [path]
    files = []
    for pattern in ("wrfout*", "wrfout_*", "*.nc", "*.nc4"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    # Sort by name (usually sorts by time)
    return sorted(set(files))


def _load_wrf(path: str):
    """Open a WRF file."""
    from wrf import WrfFile
    return WrfFile(path)


def _get_var_list() -> list[dict]:
    from wrf import list_variables
    return list_variables()


def _compute_stats(wf, varname: str, timeidx: int, units: str | None) -> dict:
    """Compute a variable and return stats dict."""
    from wrf import getvar
    kwargs = {}
    if units:
        kwargs["units"] = units
    data = getvar(wf, varname, timeidx=timeidx, **kwargs)
    is_3d = data.ndim == 3
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return {"shape": data.shape, "is_3d": is_3d, "error": "No valid data"}
    return {
        "shape": data.shape,
        "is_3d": is_3d,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def _export_var(wf, varname: str, timeidx: int, units: str | None, outpath: str) -> str:
    """Compute a variable and save to .npy file. Returns status message."""
    from wrf import getvar
    kwargs = {}
    if units:
        kwargs["units"] = units
    data = getvar(wf, varname, timeidx=timeidx, **kwargs)
    np.save(outpath, data)
    return f"Saved {outpath}  {data.shape}  {data.dtype}"


# ── App ──────────────────────────────────────────────────────────────────────

class WrfTui(App):
    """WRF file browser and variable selector."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        grid-gutter: 1;
    }

    #files-panel {
        height: 100%;
        padding: 0 1;
    }

    #vars-panel {
        height: 100%;
        padding: 0 1;
    }

    #detail-panel {
        height: 100%;
        padding: 0 1;
    }

    #file-list { height: 1fr; }
    #var-list { height: 1fr; }
    #var-filter { margin-bottom: 1; }
    #file-filter { margin-bottom: 1; }

    #file-info {
        height: auto;
        margin-bottom: 1;
    }

    #var-detail {
        height: auto;
        margin-bottom: 1;
    }

    #action-bar {
        height: auto;
        padding: 1;
        margin-top: 1;
    }

    #export-status {
        height: auto;
        margin-top: 1;
        color: $success;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
    }

    #selected-list {
        height: 1fr;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("e", "export_selected", "Export"),
        Binding("s", "show_stats", "Stats"),
        Binding("p", "plot_selected", "Plot"),
        Binding("a", "select_all", "Select all"),
        Binding("c", "clear_selected", "Clear"),
    ]

    TITLE = "wrf-rust"

    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.start_path = start_path or os.getcwd()
        self.wf = None
        self.wf_path: str | None = None
        self.all_vars: list[dict] = []
        self.selected_vars: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()

        # Left: file browser
        with Vertical(id="files-panel"):
            yield Label("[bold]Files[/bold]", classes="panel-title")
            yield Input(placeholder="Filter files...", id="file-filter")
            yield OptionList(id="file-list")
            yield Static(id="file-info")

        # Center: variable selector
        with Vertical(id="vars-panel"):
            yield Label("[bold]Variables[/bold]", classes="panel-title")
            yield Input(placeholder="Filter variables...", id="var-filter")
            yield OptionList(id="var-list")

        # Right: detail + actions
        with Vertical(id="detail-panel"):
            yield Label("[bold]Detail[/bold]", classes="panel-title")
            yield Static(id="var-detail")
            yield Label("[bold]Selected[/bold]  [dim](space to toggle)[/dim]", classes="panel-title")
            yield OptionList(id="selected-list")
            with Vertical(id="action-bar"):
                yield Button("Export selected to .npy", id="btn-export", variant="primary")
                yield Button("Plot selected to .png", id="btn-plot", variant="default")
                yield Button("Print stats", id="btn-stats", variant="default")
            yield Static(id="export-status")

        yield Footer()

    def on_mount(self) -> None:
        self.all_vars = _get_var_list()
        self._populate_var_list(self.all_vars)
        self._scan_files(self.start_path)

    # ── File list ──

    def _scan_files(self, path: str) -> None:
        files = _find_wrf_files(path)
        file_list = self.query_one("#file-list", OptionList)
        file_list.clear_options()
        if not files:
            file_list.add_option(Option("[dim]No WRF files found[/dim]", id="__none__"))
            return
        for f in files:
            name = os.path.basename(f)
            file_list.add_option(Option(name, id=f))

    @on(Input.Changed, "#file-filter")
    def _on_file_filter(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        file_list = self.query_one("#file-list", OptionList)
        files = _find_wrf_files(self.start_path)
        file_list.clear_options()
        for f in files:
            name = os.path.basename(f)
            if q and q not in name.lower():
                continue
            file_list.add_option(Option(name, id=f))

    @on(OptionList.OptionSelected, "#file-list")
    def _on_file_select(self, event: OptionList.OptionSelected) -> None:
        if not event.option or event.option.id == "__none__":
            return
        path = str(event.option.id)
        self._load_file(path)

    @work(thread=True)
    def _load_file(self, path: str) -> None:
        try:
            wf = _load_wrf(path)
            self.wf = wf
            self.wf_path = path

            try:
                times = wf.times()
            except Exception:
                times = []

            info_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            info_tbl.add_column("", style="bold cyan", width=8)
            info_tbl.add_column("")
            info_tbl.add_row("Grid", f"{wf.nx} x {wf.ny} x {wf.nz}")
            info_tbl.add_row("Times", str(wf.nt))
            info_tbl.add_row("dx", f"{wf.dx:g} m")
            if times:
                info_tbl.add_row("Start", times[0])
                if len(times) > 1:
                    info_tbl.add_row("End", times[-1])

            panel = Panel(info_tbl, title=os.path.basename(path), border_style="green")
            self.call_from_thread(
                self.query_one("#file-info", Static).update, panel
            )
            self.call_from_thread(self._set_subtitle, os.path.basename(path))

        except Exception as e:
            self.call_from_thread(
                self.query_one("#file-info", Static).update,
                Panel(f"[red]{e}[/red]", title="Error", border_style="red"),
            )

    def _set_subtitle(self, text: str) -> None:
        self.sub_title = text

    # ── Variable list ──

    def _populate_var_list(self, vars_list: list[dict]) -> None:
        var_list = self.query_one("#var-list", OptionList)
        var_list.clear_options()
        for v in vars_list:
            marker = "[green]\u2713[/green] " if v["name"] in self.selected_vars else "  "
            label = f"{marker}[bold]{v['name']}[/bold]  [dim]{v['units']}[/dim]"
            var_list.add_option(Option(label, id=v["name"]))

    @on(Input.Changed, "#var-filter")
    def _on_var_filter(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        if q:
            filtered = [
                v for v in self.all_vars
                if q in v["name"].lower() or q in v["description"].lower()
                or q in v["units"].lower()
            ]
        else:
            filtered = list(self.all_vars)
        self._populate_var_list(filtered)

    @on(OptionList.OptionHighlighted, "#var-list")
    def _on_var_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if not event.option or not event.option.id:
            return
        varname = str(event.option.id)
        info = next((v for v in self.all_vars if v["name"] == varname), None)
        if not info:
            return

        detail = self.query_one("#var-detail", Static)
        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        tbl.add_column("", style="bold cyan", width=12)
        tbl.add_column("")
        tbl.add_row("Name", f"[bold]{info['name']}[/bold]")
        tbl.add_row("Description", info["description"])
        tbl.add_row("Units", info["units"])

        # Show stats if file is loaded
        if self.wf:
            try:
                stats = _compute_stats(self.wf, varname, 0, None)
                if "error" not in stats:
                    tbl.add_row("Shape", str(stats["shape"]))
                    tbl.add_row("Min", f"{stats['min']:.4g}")
                    tbl.add_row("Max", f"{stats['max']:.4g}")
                    tbl.add_row("Mean", f"{stats['mean']:.4g}")
                    tbl.add_row("Std", f"{stats['std']:.4g}")
                else:
                    tbl.add_row("Shape", str(stats["shape"]))
                    tbl.add_row("", f"[red]{stats['error']}[/red]")
            except Exception as e:
                tbl.add_row("", f"[red]{e}[/red]")

        detail.update(Panel(tbl, title=varname, border_style="blue"))

    @on(OptionList.OptionSelected, "#var-list")
    def _on_var_select(self, event: OptionList.OptionSelected) -> None:
        """Toggle variable selection on Enter/click."""
        if not event.option or not event.option.id:
            return
        varname = str(event.option.id)
        if varname in self.selected_vars:
            self.selected_vars.remove(varname)
        else:
            self.selected_vars.append(varname)
        self._refresh_var_marks()
        self._refresh_selected_list()

    def _refresh_var_marks(self) -> None:
        """Refresh the variable list to show selection checkmarks."""
        q = self.query_one("#var-filter", Input).value.lower().strip()
        if q:
            filtered = [
                v for v in self.all_vars
                if q in v["name"].lower() or q in v["description"].lower()
            ]
        else:
            filtered = list(self.all_vars)
        self._populate_var_list(filtered)

    def _refresh_selected_list(self) -> None:
        sel_list = self.query_one("#selected-list", OptionList)
        sel_list.clear_options()
        if not self.selected_vars:
            sel_list.add_option(Option("[dim]None selected[/dim]", id="__none__"))
            return
        for name in self.selected_vars:
            info = next((v for v in self.all_vars if v["name"] == name), None)
            units = info["units"] if info else ""
            sel_list.add_option(Option(f"{name}  [dim]{units}[/dim]", id=name))

    # ── Actions ──

    @on(Button.Pressed, "#btn-export")
    def _on_export(self) -> None:
        self.action_export_selected()

    @on(Button.Pressed, "#btn-plot")
    def _on_plot(self) -> None:
        self.action_plot_selected()

    @on(Button.Pressed, "#btn-stats")
    def _on_stats_btn(self) -> None:
        self.action_show_stats()

    def action_select_all(self) -> None:
        self.selected_vars = [v["name"] for v in self.all_vars]
        self._refresh_var_marks()
        self._refresh_selected_list()
        self.notify(f"Selected all {len(self.selected_vars)} variables")

    def action_clear_selected(self) -> None:
        self.selected_vars.clear()
        self._refresh_var_marks()
        self._refresh_selected_list()
        self.notify("Cleared selection")

    @work(thread=True)
    def action_export_selected(self) -> None:
        if not self.wf or not self.selected_vars:
            self.call_from_thread(self.notify, "Select a file and variables first", severity="warning")
            return

        status = self.query_one("#export-status", Static)
        outdir = os.path.dirname(self.wf_path) if self.wf_path else "."
        messages = []

        for varname in self.selected_vars:
            outpath = os.path.join(outdir, f"{varname}.npy")
            try:
                msg = _export_var(self.wf, varname, 0, None, outpath)
                messages.append(f"[green]\u2713[/green] {msg}")
            except Exception as e:
                messages.append(f"[red]\u2717[/red] {varname}: {e}")

        self.call_from_thread(status.update, "\n".join(messages))
        self.call_from_thread(self.notify, f"Exported {len(self.selected_vars)} variables to {outdir}")

    @work(thread=True)
    def action_plot_selected(self) -> None:
        if not self.wf or not self.selected_vars:
            self.call_from_thread(self.notify, "Select a file and variables first", severity="warning")
            return

        status = self.query_one("#export-status", Static)
        outdir = os.path.dirname(self.wf_path) if self.wf_path else "."
        messages = []

        try:
            from wrf.plot import plot_field
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.call_from_thread(self.notify, "matplotlib not installed", severity="error")
            return

        for varname in self.selected_vars:
            outpath = os.path.join(outdir, f"{varname}.png")
            try:
                fig, _ = plot_field(self.wf, varname, timeidx=0)
                fig.savefig(outpath, dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
                messages.append(f"[green]\u2713[/green] {outpath}")
            except Exception as e:
                messages.append(f"[red]\u2717[/red] {varname}: {e}")

        self.call_from_thread(status.update, "\n".join(messages))
        self.call_from_thread(self.notify, f"Plotted {len(self.selected_vars)} variables to {outdir}")

    @work(thread=True)
    def action_show_stats(self) -> None:
        if not self.wf or not self.selected_vars:
            self.call_from_thread(self.notify, "Select a file and variables first", severity="warning")
            return

        tbl = Table(title="Variable Statistics", box=box.ROUNDED, padding=(0, 1))
        tbl.add_column("Variable", style="bold")
        tbl.add_column("Shape")
        tbl.add_column("Min", justify="right")
        tbl.add_column("Max", justify="right")
        tbl.add_column("Mean", justify="right")
        tbl.add_column("Std", justify="right")
        tbl.add_column("Units", style="dim")

        for varname in self.selected_vars:
            info = next((v for v in self.all_vars if v["name"] == varname), None)
            units = info["units"] if info else ""
            try:
                stats = _compute_stats(self.wf, varname, 0, None)
                if "error" not in stats:
                    tbl.add_row(
                        varname,
                        str(stats["shape"]),
                        f"{stats['min']:.4g}",
                        f"{stats['max']:.4g}",
                        f"{stats['mean']:.4g}",
                        f"{stats['std']:.4g}",
                        units,
                    )
                else:
                    tbl.add_row(varname, str(stats["shape"]), "", "", "", "", f"[red]{stats['error']}[/red]")
            except Exception as e:
                tbl.add_row(varname, "", "", "", "", "", f"[red]{e}[/red]")

        status = self.query_one("#export-status", Static)
        self.call_from_thread(status.update, tbl)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    app = WrfTui(path)
    app.run()


if __name__ == "__main__":
    main()
