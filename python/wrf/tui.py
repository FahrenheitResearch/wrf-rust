"""
wrf-rust Terminal UI.

Browse WRF files (multi-select for timesteps), select variables, then
explicitly compute/export/plot/gif. Nothing computes until you press a button.

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
from textual.containers import Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Static,
    Button,
    ProgressBar,
)
from textual.widgets.option_list import Option

from rich.panel import Panel
from rich.table import Table
from rich import box


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_wrf_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    files = []
    for pattern in ("wrfout*", "wrfout_*", "*.nc", "*.nc4"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(set(os.path.abspath(f) for f in files))


def _load_wrf(path: str):
    from wrf import WrfFile
    return WrfFile(path)


def _get_var_list() -> list[dict]:
    from wrf import list_variables
    return list_variables()


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

    #files-panel { height: 100%; padding: 0 1; }
    #vars-panel  { height: 100%; padding: 0 1; }
    #action-panel { height: 100%; padding: 0 1; }

    #file-list { height: 1fr; }
    #var-list  { height: 1fr; }
    #file-info { height: auto; margin-bottom: 1; }
    #var-detail { height: auto; margin-bottom: 1; }

    #selected-vars-list { height: auto; max-height: 10; margin-bottom: 1; }
    #selected-files-list { height: auto; max-height: 6; margin-bottom: 1; }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
    }

    #progress-bar { height: auto; margin: 1 0; }
    #progress-label { height: auto; color: $text-muted; }
    #output-log { height: auto; max-height: 12; margin-top: 1; }
    .action-btn { margin-bottom: 1; width: 100%; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "select_all_vars", "All vars"),
        Binding("c", "clear_all", "Clear"),
    ]

    TITLE = "wrf-rust"

    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.start_path = start_path or os.getcwd()
        self.all_files: list[str] = []
        self.all_vars: list[dict] = []
        self.selected_files: list[str] = []  # ordered list of selected file paths
        self.selected_vars: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()

        # Left: file browser (multi-select = timesteps)
        with Vertical(id="files-panel"):
            yield Label("[bold]Files[/bold]  [dim]Enter = toggle select (order = timesteps)[/dim]", classes="panel-title")
            yield Input(placeholder="Filter files...", id="file-filter")
            yield OptionList(id="file-list")
            yield Static(id="file-info")

        # Center: variable picker
        with Vertical(id="vars-panel"):
            yield Label("[bold]Variables[/bold]  [dim]Enter = toggle select[/dim]", classes="panel-title")
            yield Input(placeholder="Filter variables...", id="var-filter")
            yield OptionList(id="var-list")
            yield Static(id="var-detail")

        # Right: selections + actions
        with Vertical(id="action-panel"):
            yield Label("[bold]Selected files[/bold]  [dim](= timesteps)[/dim]", classes="panel-title")
            yield OptionList(id="selected-files-list")
            yield Label("[bold]Selected variables[/bold]", classes="panel-title")
            yield OptionList(id="selected-vars-list")
            yield Button("Export to .npy", id="btn-export", variant="primary", classes="action-btn")
            yield Button("Plot to .png", id="btn-plot", variant="default", classes="action-btn")
            yield Button("Animate to .gif", id="btn-gif", variant="default", classes="action-btn")
            yield Button("Compute stats", id="btn-stats", variant="default", classes="action-btn")
            yield Label("", id="progress-label")
            yield ProgressBar(id="progress-bar", total=100, show_eta=False)
            yield Static(id="output-log")

        yield Footer()

    def on_mount(self) -> None:
        self.all_vars = _get_var_list()
        self._populate_var_list(self.all_vars)
        self.all_files = _find_wrf_files(self.start_path)
        self._populate_file_list(self.all_files)
        self._refresh_selected_files()
        self._refresh_selected_vars()

        # Auto-select all files if given a directory
        if os.path.isdir(self.start_path) and self.all_files:
            self.selected_files = list(self.all_files)
            self._populate_file_list(self.all_files)
            self._refresh_selected_files()
            self._load_first_file()

        # Auto-select single file
        if os.path.isfile(self.start_path) and self.all_files:
            self.selected_files = list(self.all_files)
            self._populate_file_list(self.all_files)
            self._refresh_selected_files()
            self._load_first_file()

    # ── File list (multi-select) ──

    def _populate_file_list(self, files: list[str]) -> None:
        fl = self.query_one("#file-list", OptionList)
        fl.clear_options()
        if not files:
            fl.add_option(Option("[dim]No WRF files found[/dim]", id="__none__"))
            return
        for f in files:
            name = os.path.basename(f)
            marker = "[green]\u2713[/green] " if f in self.selected_files else "  "
            fl.add_option(Option(f"{marker}{name}", id=f))

    @on(Input.Changed, "#file-filter")
    def _on_file_filter(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        if q:
            filtered = [f for f in self.all_files if q in os.path.basename(f).lower()]
        else:
            filtered = list(self.all_files)
        self._populate_file_list(filtered)

    @on(OptionList.OptionHighlighted, "#file-list")
    def _on_file_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if not event.option or event.option.id == "__none__":
            return
        path = str(event.option.id)
        self._show_file_info(path)

    @on(OptionList.OptionSelected, "#file-list")
    def _on_file_toggle(self, event: OptionList.OptionSelected) -> None:
        """Toggle file selection."""
        if not event.option or event.option.id == "__none__":
            return
        path = str(event.option.id)
        if path in self.selected_files:
            self.selected_files.remove(path)
        else:
            self.selected_files.append(path)
        # Re-sort to match file order
        self.selected_files.sort(key=lambda f: self.all_files.index(f) if f in self.all_files else 0)
        self._populate_file_list(
            [f for f in self.all_files
             if not self.query_one("#file-filter", Input).value.strip()
             or self.query_one("#file-filter", Input).value.lower() in os.path.basename(f).lower()]
            or self.all_files
        )
        self._refresh_selected_files()

        # Load first selected file to populate grid info
        if self.selected_files:
            self._load_first_file()

    @work(thread=True)
    def _show_file_info(self, path: str) -> None:
        try:
            wf = _load_wrf(path)
            tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            tbl.add_column("", style="bold cyan", width=8)
            tbl.add_column("")
            tbl.add_row("Grid", f"{wf.nx} x {wf.ny} x {wf.nz}")
            tbl.add_row("dx", f"{wf.dx:g} m")
            self.call_from_thread(
                self.query_one("#file-info", Static).update,
                Panel(tbl, title=os.path.basename(path), border_style="cyan"),
            )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#file-info", Static).update,
                f"[red]{e}[/red]",
            )

    def _load_first_file(self) -> None:
        if self.selected_files:
            name = os.path.basename(self.selected_files[0])
            n = len(self.selected_files)
            self.sub_title = f"{name}  [{n} file{'s' if n > 1 else ''}]"

    @on(OptionList.OptionSelected, "#selected-files-list")
    def _on_deselect_file(self, event: OptionList.OptionSelected) -> None:
        if not event.option or event.option.id == "__none__":
            return
        path = str(event.option.id)
        if path in self.selected_files:
            self.selected_files.remove(path)
            self._populate_file_list(self.all_files)
            self._refresh_selected_files()

    def _refresh_selected_files(self) -> None:
        sl = self.query_one("#selected-files-list", OptionList)
        sl.clear_options()
        if not self.selected_files:
            sl.add_option(Option("[dim]None[/dim]", id="__none__"))
            return
        for i, f in enumerate(self.selected_files):
            sl.add_option(Option(f"[bold]t={i}[/bold]  {os.path.basename(f)}", id=f))

    # ── Variable list ──

    def _populate_var_list(self, vars_list: list[dict]) -> None:
        vl = self.query_one("#var-list", OptionList)
        vl.clear_options()
        for v in vars_list:
            marker = "[green]\u2713[/green] " if v["name"] in self.selected_vars else "  "
            label = f"{marker}[bold]{v['name']}[/bold]  [dim]{v['units']}[/dim]"
            vl.add_option(Option(label, id=v["name"]))

    @on(Input.Changed, "#var-filter")
    def _on_var_filter(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        if q:
            filtered = [v for v in self.all_vars
                        if q in v["name"].lower() or q in v["description"].lower()
                        or q in v["units"].lower()]
        else:
            filtered = list(self.all_vars)
        self._populate_var_list(filtered)

    @on(OptionList.OptionHighlighted, "#var-list")
    def _on_var_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if not event.option or not event.option.id:
            return
        varname = str(event.option.id)
        info = next((v for v in self.all_vars if v["name"] == varname), None)
        if info:
            self.query_one("#var-detail", Static).update(Panel(
                f"[bold]{info['name']}[/bold]\n{info['description']}\nUnits: {info['units']}",
                border_style="blue",
            ))

    @on(OptionList.OptionSelected, "#var-list")
    def _on_var_toggle(self, event: OptionList.OptionSelected) -> None:
        if not event.option or not event.option.id:
            return
        varname = str(event.option.id)
        if varname in self.selected_vars:
            self.selected_vars.remove(varname)
        else:
            self.selected_vars.append(varname)
        self._refresh_var_marks()
        self._refresh_selected_vars()

    @on(OptionList.OptionSelected, "#selected-vars-list")
    def _on_deselect_var(self, event: OptionList.OptionSelected) -> None:
        if not event.option or event.option.id == "__none__":
            return
        varname = str(event.option.id)
        if varname in self.selected_vars:
            self.selected_vars.remove(varname)
            self._refresh_var_marks()
            self._refresh_selected_vars()

    def _refresh_var_marks(self) -> None:
        q = self.query_one("#var-filter", Input).value.lower().strip()
        if q:
            filtered = [v for v in self.all_vars
                        if q in v["name"].lower() or q in v["description"].lower()]
        else:
            filtered = list(self.all_vars)
        self._populate_var_list(filtered)

    def _refresh_selected_vars(self) -> None:
        sl = self.query_one("#selected-vars-list", OptionList)
        sl.clear_options()
        if not self.selected_vars:
            sl.add_option(Option("[dim]None[/dim]", id="__none__"))
            return
        for name in self.selected_vars:
            info = next((v for v in self.all_vars if v["name"] == name), None)
            units = info["units"] if info else ""
            sl.add_option(Option(f"{name}  [dim]{units}[/dim]", id=name))

    def action_select_all_vars(self) -> None:
        self.selected_vars = [v["name"] for v in self.all_vars]
        self._refresh_var_marks()
        self._refresh_selected_vars()
        self.notify(f"Selected {len(self.selected_vars)} variables")

    def action_clear_all(self) -> None:
        self.selected_vars.clear()
        self.selected_files.clear()
        self._refresh_var_marks()
        self._refresh_selected_vars()
        self._populate_file_list(self.all_files)
        self._refresh_selected_files()
        self.notify("Cleared all")

    # ── Actions ──

    def _pre_check(self) -> bool:
        if not self.selected_files:
            self.notify("Select files first", severity="warning")
            return False
        if not self.selected_vars:
            self.notify("Select variables first", severity="warning")
            return False
        return True

    @on(Button.Pressed, "#btn-export")
    def _on_export(self) -> None:
        if self._pre_check():
            self._run_export()

    @on(Button.Pressed, "#btn-plot")
    def _on_plot(self) -> None:
        if self._pre_check():
            self._run_plot()

    @on(Button.Pressed, "#btn-gif")
    def _on_gif(self) -> None:
        if not self._pre_check():
            return
        if len(self.selected_files) < 2:
            self.notify("Select 2+ files for GIF animation", severity="warning")
            return
        self._run_gif()

    @on(Button.Pressed, "#btn-stats")
    def _on_stats(self) -> None:
        if self._pre_check():
            self._run_stats()

    @work(thread=True)
    def _run_export(self) -> None:
        from wrf import getvar
        outdir = os.path.dirname(self.selected_files[0])
        jobs = [(f, v) for f in self.selected_files for v in self.selected_vars]
        total = len(jobs)
        log_lines = []

        self.call_from_thread(self._reset_progress, total)

        for i, (fpath, varname) in enumerate(jobs):
            fname = os.path.basename(fpath)
            self.call_from_thread(self._set_progress_label,
                                  f"Exporting {varname} from {fname}  ({i+1}/{total})")
            try:
                wf = _load_wrf(fpath)
                data = getvar(wf, varname, timeidx=0)
                tag = fname.replace("wrfout_d01_", "").replace(":", "")
                outpath = os.path.join(outdir, f"{varname}_{tag}.npy")
                np.save(outpath, data)
                log_lines.append(f"[green]\u2713[/green] {varname} {fname}  {data.shape}")
            except Exception as e:
                log_lines.append(f"[red]\u2717[/red] {varname} {fname}: {e}")
            self.call_from_thread(self._advance_progress, i + 1, total)

        self.call_from_thread(self._set_progress_label, f"Done - {total} exports")
        self.call_from_thread(self._set_log, "\n".join(log_lines))

    @work(thread=True)
    def _run_plot(self) -> None:
        try:
            from wrf.plot import plot_field
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.call_from_thread(self.notify, "matplotlib not installed", severity="error")
            return

        outdir = os.path.dirname(self.selected_files[0])
        jobs = [(f, v) for f in self.selected_files for v in self.selected_vars]
        total = len(jobs)
        log_lines = []

        self.call_from_thread(self._reset_progress, total)

        for i, (fpath, varname) in enumerate(jobs):
            fname = os.path.basename(fpath)
            self.call_from_thread(self._set_progress_label,
                                  f"Plotting {varname} from {fname}  ({i+1}/{total})")
            try:
                wf = _load_wrf(fpath)
                fig, _ = plot_field(wf, varname, timeidx=0)
                tag = fname.replace("wrfout_d01_", "").replace(":", "")
                outpath = os.path.join(outdir, f"{varname}_{tag}.png")
                fig.savefig(outpath, dpi=150, bbox_inches="tight")
                plt.close(fig)
                log_lines.append(f"[green]\u2713[/green] {varname} {fname}")
            except Exception as e:
                log_lines.append(f"[red]\u2717[/red] {varname} {fname}: {e}")
            self.call_from_thread(self._advance_progress, i + 1, total)

        self.call_from_thread(self._set_progress_label, f"Done - {total} plots")
        self.call_from_thread(self._set_log, "\n".join(log_lines))

    @work(thread=True)
    def _run_gif(self) -> None:
        try:
            from wrf.plot import plot_field, _get_style, _auto_levels
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.call_from_thread(self.notify, "matplotlib not installed", severity="error")
            return

        outdir = os.path.dirname(self.selected_files[0])
        total_vars = len(self.selected_vars)
        nfiles = len(self.selected_files)
        log_lines = []

        self.call_from_thread(self._reset_progress, total_vars)

        for vi, varname in enumerate(self.selected_vars):
            self.call_from_thread(self._set_progress_label,
                                  f"GIF: {varname}  scanning {nfiles} files for scale...")

            # Pass 1: find global min/max across all files for consistent scale
            from wrf import getvar
            all_mins, all_maxs = [], []
            for fpath in self.selected_files:
                try:
                    wf = _load_wrf(fpath)
                    data = getvar(wf, varname, timeidx=0)
                    if data.ndim == 3:
                        data = data[0]
                    valid = data[np.isfinite(data)]
                    if len(valid) > 0:
                        all_mins.append(float(np.percentile(valid, 2)))
                        all_maxs.append(float(np.percentile(valid, 98)))
                except Exception:
                    pass

            style = _get_style(varname)
            if "levels" in style and style["levels"] is not None:
                fixed_levels = style["levels"]
            elif all_mins and all_maxs:
                fixed_levels = np.linspace(min(all_mins), max(all_maxs), 20)
            else:
                fixed_levels = None

            # Pass 2: render each file as a frame
            png_paths = []
            for fi, fpath in enumerate(self.selected_files):
                self.call_from_thread(self._set_progress_label,
                                      f"GIF: {varname}  frame {fi+1}/{nfiles}")
                try:
                    wf = _load_wrf(fpath)
                    kwargs = {"levels": fixed_levels} if fixed_levels is not None else {}
                    fig, _ = plot_field(wf, varname, timeidx=0, **kwargs)
                    png_path = os.path.join(outdir, f"_gif_{varname}_{fi:04d}.png")
                    fig.savefig(png_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    png_paths.append(png_path)
                except Exception:
                    pass

            # Assemble GIF
            if len(png_paths) >= 2:
                gif_path = os.path.join(outdir, f"{varname}.gif")
                try:
                    from wrf.plot import _make_gif
                    _make_gif(png_paths, gif_path, fps=4)
                    log_lines.append(f"[green]\u2713[/green] {varname}  -> {os.path.basename(gif_path)}  ({len(png_paths)} frames)")
                except Exception as e:
                    log_lines.append(f"[red]\u2717[/red] {varname} gif: {e}")
                # Clean up temp PNGs
                for p in png_paths:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            else:
                log_lines.append(f"[yellow]![/yellow] {varname}: not enough frames")

            self.call_from_thread(self._advance_progress, vi + 1, total_vars)

        self.call_from_thread(self._set_progress_label, f"Done - {total_vars} GIFs")
        self.call_from_thread(self._set_log, "\n".join(log_lines))

    @work(thread=True)
    def _run_stats(self) -> None:
        from wrf import getvar
        jobs = [(f, v) for f in self.selected_files for v in self.selected_vars]
        total = len(jobs)
        multi_file = len(self.selected_files) > 1

        self.call_from_thread(self._reset_progress, total)

        tbl = Table(box=box.ROUNDED, padding=(0, 1))
        tbl.add_column("Variable", style="bold")
        if multi_file:
            tbl.add_column("File")
        tbl.add_column("Shape")
        tbl.add_column("Min", justify="right")
        tbl.add_column("Max", justify="right")
        tbl.add_column("Mean", justify="right")
        tbl.add_column("Std", justify="right")
        tbl.add_column("Units", style="dim")

        for i, (fpath, varname) in enumerate(jobs):
            fname = os.path.basename(fpath)
            self.call_from_thread(self._set_progress_label,
                                  f"Computing {varname} from {fname}  ({i+1}/{total})")
            info = next((v for v in self.all_vars if v["name"] == varname), None)
            units = info["units"] if info else ""
            try:
                wf = _load_wrf(fpath)
                data = getvar(wf, varname, timeidx=0)
                valid = data[np.isfinite(data)]
                row = [varname]
                if multi_file:
                    row.append(fname[-20:])
                if len(valid) > 0:
                    row.extend([
                        "x".join(str(d) for d in data.shape),
                        f"{valid.min():.4g}", f"{valid.max():.4g}",
                        f"{valid.mean():.4g}", f"{valid.std():.4g}", units,
                    ])
                else:
                    row.extend(["x".join(str(d) for d in data.shape), "", "", "", "", "[dim]no data[/dim]"])
                tbl.add_row(*row)
            except Exception as e:
                row = [varname]
                if multi_file:
                    row.append(fname[-20:])
                row.extend(["", "", "", "", "", f"[red]{e}[/red]"])
                tbl.add_row(*row)
            self.call_from_thread(self._advance_progress, i + 1, total)

        self.call_from_thread(self._set_progress_label, f"Done - {total} computations")
        self.call_from_thread(self._set_log, tbl)

    # ── Progress helpers ──

    def _reset_progress(self, total: int) -> None:
        self.query_one("#progress-bar", ProgressBar).update(total=total, progress=0)
        self.query_one("#output-log", Static).update("")

    def _advance_progress(self, current: int, total: int) -> None:
        self.query_one("#progress-bar", ProgressBar).update(total=total, progress=current)

    def _set_progress_label(self, text: str) -> None:
        self.query_one("#progress-label", Label).update(text)

    def _set_log(self, content) -> None:
        self.query_one("#output-log", Static).update(content)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    app = WrfTui(path)
    app.run()


if __name__ == "__main__":
    main()
