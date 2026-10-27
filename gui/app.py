"""
gui/app.py
----------
Tkinter front-end for the Trading Strategy Arena.

Layout
  * Sidebar, three numbered steps: 1 Market data -> 2 Strategies ->
    3 Capital, costs & risk (collapsed by default). Each strategy's
    settings only appear when it is ticked. The Run button is pinned
    at the bottom so it is always visible.
  * Results notebook with plain-English tabs, each opening with a one-line
    "how to read this": Overview, Equity, Drawdowns, Risk (VaR), Combine,
    Robustness, Metrics, Trades & costs.

Backtests and parameter sweeps run on a background thread and report back
through a queue polled by the Tk main loop.

The window is DPI-aware on Windows: without this, Windows bitmap-stretches
the whole UI on scaled displays (e.g. 150%), which makes text blurry.
"""

from __future__ import annotations
import os
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog

if sys.platform == "win32":
    try:  # must run before the first Tk window is created
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.ticker  # noqa: F401  (FuncFormatter via matplotlib.ticker)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from engine import (  # noqa: E402
    SyntheticMarketData, HistoricCSVDataHandler, YFinanceDataHandler,
    StatArbStrategy, MLFactorStrategy, BuyAndHoldStrategy, MomentumStrategy, MeanReversionStrategy,
    Backtest, FrictionConfig, performance, sensitivity, export_results, export_html_report,
)
from engine.backtest import build_comparison_table, format_metric  # noqa: E402

# ---------------------------------------------------------------- theme

BG = "#0f1117"        # window background
PANEL = "#171b26"     # sidebar / tab background
CARD = "#1f2433"      # cards, inputs
BORDER = "#2f3649"
FG = "#e8eaf0"
MUTED = "#a3acbf"     # secondary text (kept bright enough to read on PANEL)
ACCENT = "#4fd1c5"
ACCENT_DARK = "#0b2e2b"
ACCENT2 = "#f6ad55"
GOOD = "#68d391"
BAD = "#fc8181"

UI_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO_FAMILY = "Consolas" if sys.platform == "win32" else "Courier"

STRATEGY_COLORS = {
    "StatArb": ACCENT,
    "MLFactor": ACCENT2,
    "BuyAndHold": "#63b3ed",
    "Momentum": "#f687b3",
    "MeanReversion": "#b794f4",
}

STRATEGY_INFO = {
    # key: (display name, one-line description)
    "StatArb": ("Pairs trading (StatArb)",
                "Trades the price gap between two related stocks: buys the cheap one, "
                "shorts the expensive one, and exits when the gap closes."),
    "MLFactor": ("Machine-learning factors",
                 "A random forest learns from momentum, reversal, volatility and RSI to "
                 "predict each stock's 5-day direction. Retrained as it goes."),
    "BuyAndHold": ("Buy & hold",
                   "Buys every stock in equal amounts on day one and holds. "
                   "The passive baseline to beat."),
    "Momentum": ("Momentum",
                 "Each month, buys the stocks that rose most over the past year and "
                 "shorts the ones that fell most."),
    "MeanReversion": ("Mean reversion",
                      "Buys a stock after an unusually sharp drop and shorts it after an "
                      "unusually sharp rise (Bollinger bands)."),
}

SHORT_NAMES = {"StatArb": "Pairs trading", "MLFactor": "ML factors", "BuyAndHold": "Buy & hold",
               "Momentum": "Momentum", "MeanReversion": "Mean reversion"}

PRESETS = [
    "Quick demo: pairs trading (offline)",
    "Pairs trading vs machine learning (offline)",
    "All 5 strategies (offline)",
    "US tech stocks via Yahoo Finance (internet)",
]

METRIC_INFO = {
    # column in comparison table: (friendly name, what it means)
    "Total Return": ("Total return", "Gain or loss over the whole period."),
    "CAGR": ("Return per year", "Average yearly growth rate (CAGR)."),
    "Annualized Vol": ("Volatility per year", "How much returns swing around. Lower = smoother."),
    "Sharpe": ("Sharpe ratio", "Return per unit of risk. Above 1 is good; below 0 lost money."),
    "Sortino": ("Sortino ratio", "Like Sharpe, but only counts downside swings as risk."),
    "Max Drawdown": ("Worst drawdown", "Largest fall from a previous peak."),
    "Calmar": ("Calmar ratio", "Return per year divided by the worst drawdown."),
    "Alpha (ann.)": ("Alpha per year", "Return not explained by the benchmark's moves."),
    "Beta": ("Beta", "Sensitivity to the benchmark. 1 = moves with it, 0 = unrelated."),
    "Final Equity": ("Final equity", "Portfolio value at the end."),
    "# Trades": ("Number of trades", "Fills executed, including risk liquidations."),
    "Total Commission": ("Commission paid", "Broker fees across all trades."),
    "Total Slippage Cost": ("Slippage cost", "Cost of filling worse than the opening price."),
    "Total Borrow Cost": ("Borrow cost", "Fees for holding short positions."),
}

TAB_INTROS = {
    "equity": "How each strategy's money grew. The dashed grey line is the benchmark "
              "(all selected stocks bought equally and held). Above it means beating the benchmark.",
    "dd": "How far each strategy fell below its own previous peak. Shallow, short dips mean a "
          "smoother ride. A line that goes flat means the strategy hit its drawdown limit and stopped.",
    "var": "Value at Risk: on a bad day, roughly how much of the portfolio could be lost "
           "(estimated from the last 20 days). Lower is safer.",
    "div": "Run strategies together. Drag the sliders to set each one's share of capital. "
           "A diversification ratio above 1 means combining them reduced risk.",
    "sens": "Re-runs one strategy across a range of values for one setting. Stable results mean "
            "the strategy is robust; a single sharp peak suggests it was over-tuned.",
    "metrics": "Every statistic for every strategy, with a plain-English explanation.",
    "tca": "Every trade and what it cost. Slippage: how much worse the fill was than the "
           "opening price. Shortfall: how far it was from the price when the decision was made.",
}

# sweep: strategy -> {display name: (parameter, start, end, step)}
SWEEP_PARAMS = {
    "StatArb": {"Lookback (days)": ("lookback", 30, 120, 15),
                "Entry z-score": ("entry_z", 1.0, 3.0, 0.25)},
    "MLFactor": {"Training window (days)": ("train_window", 100, 250, 50),
                 "Retrain every (days)": ("refit_every", 10, 40, 10)},
    "MeanReversion": {"Band period (days)": ("bb_period", 10, 40, 5),
                      "Band width (std devs)": ("bb_std", 1.0, 3.0, 0.5)},
    "Momentum": {"Lookback (days)": ("lookback", 63, 252, 63),
                 "Rebalance every (days)": ("rebalance_every", 5, 42, 7)},
}
INT_PARAMS = {"lookback", "train_window", "refit_every", "bb_period", "rebalance_every"}

matplotlib.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": FG,
    "text.color": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#2a3040", "legend.facecolor": CARD, "legend.edgecolor": BORDER,
    "legend.labelcolor": FG, "legend.framealpha": 0.9,
    "font.family": [UI_FAMILY, "DejaVu Sans"], "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 10,
    "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- small widgets

class ToolTip:
    """Borderless popup shown while the pointer is over a widget."""

    def __init__(self, widget, text, wrap=360):
        self.widget, self.text, self.wrap = widget, text, wrap
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip, add="+")
        widget.bind("<Leave>", self.hide_tip, add="+")

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background=CARD, foreground=FG,
                 relief="solid", borderwidth=1, font=(UI_FAMILY, 9), wraplength=self.wrap,
                 padx=8, pady=6).pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None


class Field(ttk.Frame):
    """A label above a full-width entry: readable, and long values aren't cut off."""

    def __init__(self, parent, label, default="", tooltip="", style="Panel.TFrame"):
        super().__init__(parent, style=style)
        lbl = ttk.Label(self, text=label, style="Field.TLabel")
        lbl.pack(anchor="w")
        self.var = tk.StringVar(value=str(default))
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.pack(fill="x", pady=(2, 0))
        if tooltip:
            ToolTip(lbl, tooltip)
            ToolTip(self.entry, tooltip)

    def get(self):
        return self.var.get().strip()

    def get_float(self, default=0.0):
        try:
            return float(self.get().replace(",", "").replace("%", ""))
        except ValueError:
            return default

    def get_int(self, default=0):
        try:
            return int(float(self.get().replace(",", "")))
        except ValueError:
            return default


class ScrollableFrame(ttk.Frame):
    """Vertical scroll container; the mouse wheel scrolls it only while hovered."""

    def __init__(self, container, bg=PANEL, **kwargs):
        super().__init__(container, style="Panel.TFrame", **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_frame, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ---------------------------------------------------------------- app

class TradingArenaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trading Strategy Arena")
        self.configure(bg=BG)

        # pixel sizes are written for a 96-dpi screen and scaled to the real one
        self.S = max(self.winfo_fpixels("1i") / 96.0, 1.0)
        self.fig_dpi = 96  # matplotlib's Tk canvas applies the screen scaling itself
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(self.px(1440), int(sw * 0.94)), min(self.px(900), int(sh * 0.88))
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max((sh - h) // 3, 0)}")
        self.minsize(min(self.px(1100), sw), min(self.px(680), sh))

        self._setup_fonts()
        self._setup_style()

        self.result_queue = queue.Queue()
        self.all_results = {}
        self.benchmark_equity = None
        self.comparison_table = None
        self.var_conf = 0.95
        self.sliders = {}
        self.csv_dir = os.path.join(PROJECT_ROOT, "data", "nse_real")

        self._build_layout()
        self._poll_id = self.after(150, self._poll_queue)

    def px(self, n):
        return int(round(n * self.S))

    def destroy(self):
        # cancel the polling loop so Tcl doesn't fire it on a dead interpreter
        try:
            self.after_cancel(self._poll_id)
        except (AttributeError, tk.TclError):
            pass
        super().destroy()

    # ---------------------------------------------------------------- style

    def _setup_fonts(self):
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=UI_FAMILY, size=10)
        self.F_BASE = (UI_FAMILY, 10)
        self.F_SMALL = (UI_FAMILY, 9)
        self.F_BOLD = (UI_FAMILY, 10, "bold")
        self.F_SECTION = (UI_FAMILY, 12, "bold")
        self.F_TITLE = (UI_FAMILY, 17, "bold")
        self.F_BIG = (UI_FAMILY, 20, "bold")
        self.F_MONO = (MONO_FAMILY, 10)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        p = self.px
        style.configure(".", background=BG, foreground=FG, fieldbackground=CARD,
                        font=self.F_BASE, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=self.F_SMALL)
        style.configure("MutedBG.TLabel", background=BG, foreground=MUTED, font=self.F_SMALL)
        style.configure("Field.TLabel", background=PANEL, foreground=MUTED, font=self.F_SMALL)
        style.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=self.F_SECTION)
        style.configure("Title.TLabel", background=BG, foreground=FG, font=self.F_TITLE)
        style.configure("Intro.TLabel", background=PANEL, foreground=MUTED, font=self.F_BASE)

        style.configure("TEntry", fieldbackground=CARD, foreground=FG, insertcolor=FG,
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(p(6), p(4)))
        style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])

        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG,
                        arrowcolor=FG, bordercolor=BORDER, padding=(p(6), p(4)))
        style.map("TCombobox",
                  fieldbackground=[("readonly", CARD)], foreground=[("readonly", FG)],
                  selectbackground=[("readonly", CARD)], selectforeground=[("readonly", FG)],
                  background=[("active", BORDER)], bordercolor=[("focus", ACCENT)])
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", ACCENT_DARK)
        self.option_add("*TCombobox*Listbox.font", self.F_BASE)

        for cls in ("TCheckbutton", "TRadiobutton"):
            style.configure(cls, background=PANEL, foreground=FG, font=self.F_BASE,
                            indicatorbackground=CARD, indicatorforeground=ACCENT_DARK,
                            indicatormargin=(0, 0, p(8), 0), indicatorsize=p(15), focuscolor=PANEL)
            style.map(cls, background=[("active", PANEL)],
                      indicatorbackground=[("selected", ACCENT), ("active", BORDER)])
        style.configure("Strategy.TCheckbutton", font=self.F_BOLD)

        style.configure("TButton", background=CARD, foreground=FG, bordercolor=BORDER,
                        padding=(p(12), p(6)), font=self.F_BOLD, focuscolor=CARD)
        style.map("TButton", background=[("active", BORDER), ("disabled", PANEL)],
                  foreground=[("disabled", "#5d6578")])
        style.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_DARK,
                        font=(UI_FAMILY, 12, "bold"), padding=(p(12), p(10)), focuscolor=ACCENT)
        style.map("Accent.TButton", background=[("active", "#3bb8ab"), ("disabled", BORDER)],
                  foreground=[("disabled", MUTED)])
        style.configure("Link.TButton", background=PANEL, foreground=ACCENT, bordercolor=PANEL,
                        padding=(0, p(2)), font=self.F_BOLD, focuscolor=PANEL)
        style.map("Link.TButton", background=[("active", PANEL)], foreground=[("active", FG)])

        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(p(14), p(8)),
                        font=self.F_BOLD, borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", PANEL), ("active", CARD)],
                  foreground=[("selected", ACCENT), ("active", FG)], expand=[("selected", (0, 0, 0, 0))])

        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                        rowheight=p(28), font=self.F_BASE, bordercolor=BORDER, borderwidth=0)
        style.map("Treeview", background=[("selected", BORDER)], foreground=[("selected", FG)])
        style.configure("Treeview.Heading", background=CARD, foreground=ACCENT, font=self.F_BOLD,
                        padding=(p(6), p(6)), bordercolor=BORDER, relief="flat")
        style.map("Treeview.Heading", background=[("active", BORDER)])

        for orient in ("Vertical", "Horizontal"):
            style.configure(f"{orient}.TScrollbar", background=CARD, troughcolor=PANEL,
                            bordercolor=PANEL, arrowcolor=MUTED, lightcolor=CARD, darkcolor=CARD,
                            gripcount=0)
            style.map(f"{orient}.TScrollbar", background=[("active", BORDER)])
        style.configure("TProgressbar", background=ACCENT, troughcolor=CARD, bordercolor=BG,
                        lightcolor=ACCENT, darkcolor=ACCENT, thickness=p(8))
        style.configure("Horizontal.TScale", background=ACCENT, troughcolor=CARD)
        style.configure("TSeparator", background=BORDER)

    # ---------------------------------------------------------------- layout

    def _build_layout(self):
        p = self.px
        header = ttk.Frame(self)
        header.pack(fill="x", padx=p(18), pady=(p(12), p(8)))
        title_box = ttk.Frame(header)
        title_box.pack(side="left")
        ttk.Label(title_box, text="Trading Strategy Arena", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="Backtest trading strategies side by side, with realistic trading costs "
                                  "and no peeking at future prices.",
                  style="MutedBG.TLabel", font=self.F_BASE).pack(anchor="w")

        preset_box = ttk.Frame(header)
        preset_box.pack(side="right", anchor="e")
        ttk.Label(preset_box, text="Quick start", style="MutedBG.TLabel").pack(anchor="w")
        self.preset_var = tk.StringVar(value="Choose a ready-made setup…")
        preset_combo = ttk.Combobox(preset_box, textvariable=self.preset_var, state="readonly",
                                    width=44, values=PRESETS)
        preset_combo.pack(anchor="e", pady=(p(2), 0))
        preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)
        ToolTip(preset_combo, "Fills in data, strategies and costs for a typical comparison. "
                              "You can still change anything afterwards.")

        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", padx=p(18), pady=(p(4), p(10)))
        self.status_var = tk.StringVar(value="Pick a quick-start setup (top right) or choose data and "
                                             "strategies on the left, then press Run backtest.")
        ttk.Label(footer, textvariable=self.status_var, style="MutedBG.TLabel",
                  font=self.F_BASE).pack(side="left")
        self.export_btn = ttk.Button(footer, text="Export results…", command=self._export_dialog,
                                     state="disabled")
        self.export_btn.pack(side="right", padx=(p(10), 0))
        ToolTip(self.export_btn, "Save the trade logs and metrics as CSV files plus an HTML report.")
        self.progress = ttk.Progressbar(footer, mode="determinate", length=p(220))
        self.progress.pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=p(18))

        sidebar = ttk.Frame(body, style="Panel.TFrame", width=p(370))
        sidebar.pack(side="left", fill="y", padx=(0, p(14)))
        sidebar.pack_propagate(False)
        run_bar = ttk.Frame(sidebar, style="Panel.TFrame")
        run_bar.pack(side="bottom", fill="x", padx=p(14), pady=p(12))
        self.run_btn = ttk.Button(run_bar, text="▶   Run backtest", style="Accent.TButton",
                                  command=self.run_backtest)
        self.run_btn.pack(fill="x")
        ttk.Separator(sidebar).pack(side="bottom", fill="x")

        self.config_scroll = ScrollableFrame(sidebar)
        self.config_scroll.pack(fill="both", expand=True)
        self._build_config_panel(self.config_scroll.scrollable_frame)

        self.right = ttk.Frame(body)
        self.right.pack(side="left", fill="both", expand=True)
        self._build_results_panel(self.right)

    @staticmethod
    def _autowrap(label, margin=4):
        """Wrap a label's text to whatever width it is given, instead of a fixed width."""
        label.bind("<Configure>", lambda e: label.configure(wraplength=max(e.width - margin, 50)))
        return label

    def _section(self, parent, number, title, subtitle=""):
        p = self.px
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.pack(fill="x", padx=p(14), pady=(p(14), p(2)))
        ttk.Label(box, text=f"{number}   {title}", style="Section.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(box, text=subtitle, style="Muted.TLabel", wraplength=p(320),
                      justify="left").pack(anchor="w", pady=(p(2), 0))
        body = ttk.Frame(parent, style="Panel.TFrame")
        body.pack(fill="x", padx=p(14), pady=(p(6), 0))
        return body

    def _grid_fields(self, parent, specs, cols=2):
        """specs: list of (attr, label, default, tooltip). Returns nothing; sets self.<attr>."""
        p = self.px
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=(p(4), 0))
        for c in range(cols):
            frame.columnconfigure(c, weight=1, uniform="f")
        for i, (attr, label, default, tip) in enumerate(specs):
            f = Field(frame, label, default, tip)
            f.grid(row=i // cols, column=i % cols, sticky="ew",
                   padx=(0 if i % cols == 0 else p(5), p(5) if i % cols < cols - 1 else 0), pady=p(4))
            setattr(self, attr, f)
        return frame

    def _build_config_panel(self, parent):
        p = self.px

        # ---- 1. data
        data = self._section(parent, "1", "Market data")
        self.data_source = tk.StringVar(value="real")
        for value, text, tip in [
            ("real", "Indian large-cap stocks (offline)", "10 NSE stocks, daily prices 2011–2021, split-adjusted. Bundled with the project."),
            ("synthetic", "Synthetic test data (offline)", "Randomly generated prices that include one genuinely cointegrated pair (AAA/BBB)."),
            ("yfinance", "Any ticker from Yahoo Finance", "Downloads daily prices; needs an internet connection."),
        ]:
            rb = ttk.Radiobutton(data, text=text, value=value, variable=self.data_source,
                                 command=self._on_source_change)
            rb.pack(anchor="w", pady=p(2))
            ToolTip(rb, tip)

        self._source_frames = ttk.Frame(data, style="Panel.TFrame")
        self._source_frames.pack(fill="x", pady=(p(6), 0))

        self._real_frame = ttk.Frame(self._source_frames, style="Panel.TFrame")
        self.real_symbols_entry = Field(self._real_frame, "Stocks (comma-separated)",
                                        "ICICIBANK,AXISBANK,SBIN,TCS,INFY,WIPRO",
                                        "Available: ICICIBANK, AXISBANK, SBIN, HDFCBANK, TCS, INFY, WIPRO, "
                                        "RELIANCE, TATAMOTORS, ITC")
        self.real_symbols_entry.pack(fill="x")
        folder_row = ttk.Frame(self._real_frame, style="Panel.TFrame")
        folder_row.pack(fill="x", pady=(p(4), 0))
        self.csv_dir_label = ttk.Label(folder_row, style="Muted.TLabel")
        self.csv_dir_label.pack(side="left")
        ttk.Button(folder_row, text="Change…", style="Link.TButton",
                   command=self._choose_csv_dir).pack(side="right")
        self._update_csv_dir_label()

        self._synth_frame = ttk.Frame(self._source_frames, style="Panel.TFrame")
        self.symbols_entry = Field(self._synth_frame, "Symbols", "AAA,BBB,TREND1,TREND2",
                                   "The StatArb pair's two symbols are generated as a cointegrated pair.")
        self.symbols_entry.pack(fill="x")
        self._grid_fields(self._synth_frame, [
            ("days_entry", "Days of history", 500, "Number of daily bars to generate."),
            ("seed_entry", "Random seed", 7, "Same seed = same prices every run."),
        ])

        self._yf_frame = ttk.Frame(self._source_frames, style="Panel.TFrame")
        self.yf_symbols_entry = Field(self._yf_frame, "Tickers (comma-separated)", "AAPL,MSFT,KO,PEP",
                                      "Yahoo Finance tickers, e.g. AAPL, MSFT, RELIANCE.NS")
        self.yf_symbols_entry.pack(fill="x")
        self._grid_fields(self._yf_frame, [
            ("yf_start_entry", "From (YYYY-MM-DD)", "2019-01-01", "First date to download."),
            ("yf_end_entry", "To (blank = today)", "", "Last date to download."),
        ])

        # ---- 2. strategies
        strat = self._section(parent, "2", "Strategies",
                              "Tick the strategies to compare. Each one trades its own copy of the capital.")
        self.use_statarb, self.statarb_frame = self._strategy_block(strat, "StatArb", True)
        self.pair_entry = Field(self.statarb_frame, "Stock pair (A,B)", "ICICIBANK,AXISBANK",
                                "Two related stocks from your list, e.g. two banks.")
        self.pair_entry.pack(fill="x", pady=(0, p(2)))
        self._grid_fields(self.statarb_frame, [
            ("lookback_entry", "Lookback (days)", 60, "Window used to measure the normal gap between the two stocks."),
            ("entryz_entry", "Entry z-score", 2.0, "How unusual the gap must be before trading. Higher = fewer, more selective trades."),
        ])

        self.use_ml, self.ml_frame = self._strategy_block(strat, "MLFactor", True)
        self.ml_symbols_entry = Field(self.ml_frame, "Stocks to trade", "TCS,INFY", "Must be in your stock list.")
        self.ml_symbols_entry.pack(fill="x", pady=(0, p(2)))
        self._grid_fields(self.ml_frame, [
            ("train_window_entry", "Training window (days)", 150, "How much recent history the model learns from."),
            ("refit_entry", "Retrain every (days)", 20, "How often the model is retrained on new data."),
        ])

        self.use_bah, _ = self._strategy_block(strat, "BuyAndHold", False, has_params=False)

        self.use_mom, self.mom_frame = self._strategy_block(strat, "Momentum", False)
        self._grid_fields(self.mom_frame, [
            ("mom_lookback", "Lookback (days)", 252, "Past performance window used to rank stocks (252 ≈ 1 year)."),
            ("mom_top_n", "Stocks per side", 2, "How many winners to buy and losers to short. Needs at least 2× this many stocks."),
        ])

        self.use_mr, self.mr_frame = self._strategy_block(strat, "MeanReversion", False)
        self._grid_fields(self.mr_frame, [
            ("mr_period", "Band period (days)", 20, "Moving-average window for the bands."),
            ("mr_std", "Band width (std devs)", 2.0, "How far price must stretch before trading. Higher = fewer trades."),
        ])

        # ---- 3. capital, costs & risk (collapsed)
        adv = self._section(parent, "3", "Capital, costs & risk",
                            "Realistic defaults are already set. Expand to change them.")
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_btn = ttk.Button(adv, text="Show settings  ▾", style="Link.TButton",
                                   command=self._toggle_advanced)
        self._adv_btn.pack(anchor="w")
        self._adv_frame = ttk.Frame(adv, style="Panel.TFrame")

        def sub(title):
            ttk.Label(self._adv_frame, text=title, style="Panel.TLabel",
                      font=self.F_BOLD).pack(anchor="w", pady=(p(10), 0))

        sub("Capital")
        self._grid_fields(self._adv_frame, [
            ("capital_entry", "Starting capital", "1,000,000", "Money each strategy starts with."),
            ("risk_entry", "Risk per position (%)", "1", "Each position is sized so its typical daily swing is this % of capital."),
        ])
        sub("Trading costs")
        self._grid_fields(self._adv_frame, [
            ("commission_entry", "Commission (bps)", "1.0", "Broker fee per trade. 1 bp = 0.01% of the trade value."),
            ("spread_entry", "Bid/ask spread (bps)", "2.0", "Cost of crossing the bid/ask spread on every trade."),
            ("impact_entry", "Market impact", "0.10", "Extra slippage for large orders relative to normal trading volume."),
            ("latency_entry", "Order delay (days)", "1", "Days between the signal and the fill. 1 = next day's open "
                                                       "(realistic). 0 = same-day close (optimistic)."),
            ("borrow_cost_entry", "Short borrow fee (bps/yr)", "0", "Yearly fee for borrowing shares to short."),
        ])
        sub("Risk limits")
        self._grid_fields(self._adv_frame, [
            ("max_dd_entry", "Stop at drawdown (%)", "", "Optional. Sell everything and stop trading after falling this % from the peak, e.g. 25."),
            ("max_pos_entry", "Max per position (%)", "", "Optional. Largest share of capital in one stock, e.g. 20."),
            ("max_lev_entry", "Max leverage (×)", "1.0", "Total long + short exposure as a multiple of capital. 1 = no borrowing. Blank = no limit."),
            ("var_conf_entry", "VaR confidence (%)", "95", "Confidence level for Value at Risk, e.g. 95 or 99."),
        ])

        ttk.Frame(parent, style="Panel.TFrame", height=p(10)).pack()
        self._on_source_change()

    def _strategy_block(self, parent, key, default_on, has_params=True):
        p = self.px
        name, desc = STRATEGY_INFO[key]
        var = tk.BooleanVar(value=default_on)
        block = ttk.Frame(parent, style="Panel.TFrame")
        block.pack(fill="x", pady=(p(8), 0))
        params = ttk.Frame(block, style="Panel.TFrame") if has_params else None

        def toggle():
            if params is not None:
                if var.get():
                    params.pack(fill="x", padx=(p(26), 0), pady=(p(4), 0))
                else:
                    params.pack_forget()

        chk = ttk.Checkbutton(block, text=name, variable=var, style="Strategy.TCheckbutton", command=toggle)
        chk.pack(anchor="w")
        ttk.Label(block, text=desc, style="Muted.TLabel", wraplength=p(300),
                  justify="left").pack(anchor="w", padx=(p(26), 0))
        var._toggle = toggle  # so presets can refresh visibility
        toggle()
        return var, params

    def _refresh_strategy_blocks(self):
        for var in (self.use_statarb, self.use_ml, self.use_bah, self.use_mom, self.use_mr):
            var._toggle()

    def _toggle_advanced(self):
        if self._adv_open.get():
            self._adv_frame.pack_forget()
            self._adv_btn.config(text="Show settings  ▾")
        else:
            self._adv_frame.pack(fill="x")
            self._adv_btn.config(text="Hide settings  ▴")
            self.update_idletasks()
            self.config_scroll.canvas.configure(scrollregion=self.config_scroll.canvas.bbox("all"))
            self.config_scroll.canvas.yview_moveto(1.0)
        self._adv_open.set(not self._adv_open.get())

    def _choose_csv_dir(self):
        d = filedialog.askdirectory(title="Folder containing <SYMBOL>.csv files", initialdir=self.csv_dir)
        if d:
            self.csv_dir = d
            self._update_csv_dir_label()

    def _update_csv_dir_label(self):
        rel = os.path.relpath(self.csv_dir, PROJECT_ROOT) if self.csv_dir.startswith(PROJECT_ROOT) else self.csv_dir
        if len(rel) > 38:
            rel = "…" + rel[-37:]
        self.csv_dir_label.config(text=f"Data folder: {rel}")

    def _on_source_change(self, event=None):
        source = self.data_source.get()
        for frame in (self._real_frame, self._synth_frame, self._yf_frame):
            frame.pack_forget()
        if source == "real":
            self._real_frame.pack(fill="x")
            self.pair_entry.var.set("ICICIBANK,AXISBANK")
            self.ml_symbols_entry.var.set("TCS,INFY")
        elif source == "synthetic":
            self._synth_frame.pack(fill="x")
            self.pair_entry.var.set("AAA,BBB")
            self.ml_symbols_entry.var.set("TREND1,TREND2")
        else:
            self._yf_frame.pack(fill="x")
            self.pair_entry.var.set("KO,PEP")
            self.ml_symbols_entry.var.set("AAPL,MSFT")

    def _apply_preset(self, event=None):
        preset = self.preset_var.get()
        if preset not in PRESETS:
            return
        # shared defaults
        values = dict(capital_entry="1,000,000", risk_entry="1", max_dd_entry="", max_pos_entry="",
                      max_lev_entry="1.0", borrow_cost_entry="0", commission_entry="1.0",
                      spread_entry="2.0", impact_entry="0.10", latency_entry="1")
        on = dict(use_statarb=False, use_ml=False, use_bah=False, use_mom=False, use_mr=False)
        if preset == PRESETS[0]:
            self.data_source.set("real"); self._on_source_change()
            self.real_symbols_entry.var.set("ICICIBANK,AXISBANK")
            on.update(use_statarb=True)
        elif preset == PRESETS[1]:
            self.data_source.set("real"); self._on_source_change()
            self.real_symbols_entry.var.set("ICICIBANK,AXISBANK,TCS,INFY")
            values.update(borrow_cost_entry="50")
            on.update(use_statarb=True, use_ml=True)
        elif preset == PRESETS[2]:
            self.data_source.set("real"); self._on_source_change()
            self.real_symbols_entry.var.set("ICICIBANK,AXISBANK,SBIN,TCS,INFY,WIPRO")
            values.update(max_dd_entry="25", max_pos_entry="30", borrow_cost_entry="100")
            on = {k: True for k in on}
        else:
            self.data_source.set("yfinance"); self._on_source_change()
            self.yf_symbols_entry.var.set("AAPL,MSFT,GOOGL,AMZN")
            self.yf_start_entry.var.set("2021-01-01")
            self.yf_end_entry.var.set("")
            values.update(risk_entry="1.5", borrow_cost_entry="200", commission_entry="1.5",
                          spread_entry="3.0", impact_entry="0.15")
            on.update(use_bah=True, use_mom=True)
        for attr, v in values.items():
            getattr(self, attr).var.set(v)
        for attr, v in on.items():
            getattr(self, attr).set(v)
        self._refresh_strategy_blocks()
        self.status_var.set(f"Loaded “{preset}”. Press Run backtest.")

    # ---------------------------------------------------------------- results panel

    def _tab(self, key_text, intro_key=None):
        p = self.px
        tab = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(tab, text=key_text)
        if intro_key:
            self._autowrap(ttk.Label(tab, text=TAB_INTROS[intro_key], style="Intro.TLabel",
                                     justify="left")).pack(fill="x", padx=p(16), pady=(p(12), p(4)))
        return tab

    def _figure(self, parent, size=(8, 5)):
        fig = Figure(figsize=size, dpi=self.fig_dpi, layout="constrained")
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        return fig, ax, canvas

    def _build_results_panel(self, parent):
        p = self.px
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.dash_tab = self._tab("Overview")
        self.equity_tab = self._tab("Equity", "equity")
        self.dd_tab = self._tab("Drawdowns", "dd")
        self.var_tab = self._tab("Risk (VaR)", "var")
        self.div_tab = self._tab("Combine", "div")
        self.sens_tab = self._tab("Robustness", "sens")
        self.metrics_tab = self._tab("Metrics", "metrics")
        self.tca_tab = self._tab("Trades & costs", "tca")

        self._build_dashboard_layout(self.dash_tab)

        # equity
        ctrl = ttk.Frame(self.equity_tab, style="Panel.TFrame")
        ctrl.pack(fill="x", padx=p(16))
        self.eq_toggle_var = tk.StringVar(value="pct")
        ttk.Label(ctrl, text="Show as:", style="Panel.TLabel").pack(side="left")
        ttk.Radiobutton(ctrl, text="Return (%)", variable=self.eq_toggle_var, value="pct",
                        command=self._replotted_equity).pack(side="left", padx=p(10))
        ttk.Radiobutton(ctrl, text="Portfolio value", variable=self.eq_toggle_var, value="val",
                        command=self._replotted_equity).pack(side="left")
        self.equity_fig, self.equity_ax, self.equity_canvas = self._figure(self.equity_tab)
        self.equity_canvas.get_tk_widget().pack(fill="both", expand=True, padx=p(8), pady=p(8))

        self.dd_fig, self.dd_ax, self.dd_canvas = self._figure(self.dd_tab)
        self.dd_canvas.get_tk_widget().pack(fill="both", expand=True, padx=p(8), pady=p(8))

        self.var_fig, self.var_ax, self.var_canvas = self._figure(self.var_tab)
        self.var_canvas.get_tk_widget().pack(fill="both", expand=True, padx=p(8), pady=p(8))

        self._build_diversification_layout(self.div_tab)
        self._build_sensitivity_layout(self.sens_tab)

        # metrics: one row per metric, one column per strategy; hover a row for its meaning
        self.metric_help = tk.Label(self.metrics_tab, text="Hover over a metric to see what it means.",
                                    bg=CARD, fg=FG, font=self.F_BASE, anchor="w", padx=p(12), pady=p(8))
        self.metric_help.pack(side="bottom", fill="x", padx=p(16), pady=(p(4), p(12)))
        self.metrics_tree = self._tree(self.metrics_tab, height=None)
        self.metrics_tree.bind("<Motion>", self._on_metric_hover)
        self.metrics_tree.bind("<Leave>", lambda e: self.metric_help.config(
            text="Hover over a metric to see what it means."))
        self._metric_meanings = {}

        # trades & costs: headline tiles, cost per strategy, then the trade log
        self.tca_tiles = ttk.Frame(self.tca_tab, style="Panel.TFrame")
        self.tca_tiles.pack(fill="x", padx=p(16), pady=(p(4), p(10)))
        self.tca_summary_var = tk.StringVar(value="")
        ttk.Label(self.tca_tab, text="Cost by strategy", style="Panel.TLabel",
                  font=self.F_BOLD).pack(anchor="w", padx=p(16))
        self.cost_tree = self._tree(self.tca_tab, height=5, scroll=False, pady=(p(4), p(12)))
        ttk.Label(self.tca_tab, text="Trade log (most recent 300)", style="Panel.TLabel",
                  font=self.F_BOLD).pack(anchor="w", padx=p(16))
        self.tca_tree = self._tree(self.tca_tab, height=None, pady=(p(4), p(12)))

        self._blank_axes()

    def _tree(self, parent, height=None, scroll=True, pady=None):
        """A Treeview in its own frame; height=None fills the remaining space."""
        p = self.px
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.pack(fill="both" if height is None else "x", expand=height is None,
                 padx=p(16), pady=pady or (p(4), p(4)))
        tree = ttk.Treeview(box, show="headings", height=height or 10)
        tree.tag_configure("odd", background=CARD)
        if scroll:
            sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        return tree

    def _set_columns(self, tree, cols):
        """cols: list of (key, heading, width, anchor, stretch)."""
        p = self.px
        tree.delete(*tree.get_children())
        tree["columns"] = [c[0] for c in cols]
        for key, text, width, anchor, stretch in cols:
            tree.heading(key, text=text, anchor=anchor)
            tree.column(key, width=p(width), minwidth=p(40), anchor=anchor, stretch=stretch)

    def _on_metric_hover(self, event):
        row = self.metrics_tree.identify_row(event.y)
        if row and row in self._metric_meanings:
            name, meaning = self._metric_meanings[row]
            self.metric_help.config(text=f"{name}: {meaning}")

    def _build_dashboard_layout(self, parent):
        p = self.px
        # empty state, replaced by results after the first run
        self.dash_empty = ttk.Frame(parent, style="Panel.TFrame")
        self.dash_empty.pack(fill="both", expand=True, padx=p(40), pady=p(40))
        ttk.Label(self.dash_empty, text="Compare trading strategies in three steps",
                  style="Panel.TLabel", font=self.F_SECTION).pack(anchor="w", pady=(0, p(14)))
        for n, text in [
            ("1", "Pick a Quick start setup at the top right, or choose market data and strategies "
                  "in the panel on the left."),
            ("2", "Press  ▶ Run backtest. Each strategy trades through the same historical prices, "
                  "paying commission, spread and slippage, with orders filled the next day."),
            ("3", "Results appear here. The tabs above show growth, losses, risk, costs, and how the "
                  "strategies behave when combined."),
        ]:
            row = ttk.Frame(self.dash_empty, style="Panel.TFrame")
            row.pack(anchor="w", fill="x", pady=p(6))
            tk.Label(row, text=n, bg=ACCENT, fg=ACCENT_DARK, font=self.F_BOLD,
                     width=2).pack(side="left", anchor="n", padx=(0, p(12)))
            self._autowrap(ttk.Label(row, text=text, style="Panel.TLabel", justify="left")).pack(
                side="left", fill="x", expand=True)

        self.dash_results = ttk.Frame(parent, style="Panel.TFrame")
        self.win_lbl = tk.Label(self.dash_results, text="", bg="#163633", fg=FG, font=self.F_SECTION,
                                anchor="w", padx=p(14), pady=p(10))
        self.win_lbl.pack(fill="x", padx=p(16), pady=(p(14), p(10)))
        self.cards_frame = ttk.Frame(self.dash_results, style="Panel.TFrame")
        self.cards_frame.pack(fill="x", padx=p(16))

        lower = ttk.Frame(self.dash_results, style="Panel.TFrame")
        lower.pack(fill="both", expand=True, padx=p(16), pady=(p(14), p(10)))

        self.heat_frame = ttk.Frame(lower, style="Panel.TFrame", width=p(380))
        self.heat_frame.pack(side="right", fill="y", padx=(p(14), 0))
        self.heat_frame.pack_propagate(False)
        ttk.Label(self.heat_frame, text="Do the strategies move together?", style="Panel.TLabel",
                  font=self.F_BOLD).pack(anchor="w")
        ttk.Label(self.heat_frame, text="Correlation of daily returns: near 0 means they behave "
                                        "differently, which is good for combining them.",
                  style="Muted.TLabel", wraplength=p(370), justify="left").pack(anchor="w")
        self.heat_fig, self.heat_ax, self.heat_canvas = self._figure(self.heat_frame, size=(3.8, 3.4))
        self.heat_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(p(6), 0))

        txt_box = ttk.Frame(lower, style="Panel.TFrame")
        txt_box.pack(side="left", fill="both", expand=True)
        ttk.Label(txt_box, text="What happened", style="Panel.TLabel",
                  font=self.F_BOLD).pack(anchor="w", pady=(0, p(4)))
        self.dash_txt = tk.Text(txt_box, wrap="word", bg=PANEL, fg=FG, relief="flat", bd=0,
                                font=self.F_BASE, spacing1=p(2), spacing3=p(6), padx=0,
                                highlightthickness=0, cursor="arrow")
        dsb = ttk.Scrollbar(txt_box, orient="vertical", command=self.dash_txt.yview)
        self.dash_txt.configure(yscrollcommand=dsb.set)
        dsb.pack(side="right", fill="y")
        self.dash_txt.pack(side="left", fill="both", expand=True)
        self.dash_txt.tag_configure("muted", foreground=MUTED)
        self.dash_txt.tag_configure("warn", foreground=ACCENT2)
        for name, color in STRATEGY_COLORS.items():
            self.dash_txt.tag_configure(name, foreground=color, font=self.F_BOLD)
        self.dash_txt.config(state="disabled")

    def _build_diversification_layout(self, parent):
        p = self.px
        self.sliders = {}
        left = ttk.Frame(parent, style="Panel.TFrame", width=p(340))
        left.pack(side="left", fill="y", padx=p(16), pady=p(8))
        left.pack_propagate(False)
        self.sliders_frame = ttk.Frame(left, style="Panel.TFrame")
        self.sliders_frame.pack(fill="x")
        ttk.Label(self.sliders_frame, text="Run a backtest with two or more strategies to combine them.",
                  style="Muted.TLabel", wraplength=p(320)).pack(anchor="w", pady=p(10))
        self.div_stats_lbl = ttk.Label(left, text="", style="Panel.TLabel", justify="left",
                                       wraplength=p(320), font=self.F_BOLD)
        self.div_stats_lbl.pack(anchor="w", pady=(p(14), 0))
        self.div_stats_grid = ttk.Frame(left, style="Panel.TFrame")
        self.div_stats_grid.pack(fill="x", pady=(p(8), 0))
        right = ttk.Frame(parent, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(0, p(8)), pady=p(8))
        self.div_fig, self.div_ax, self.div_canvas = self._figure(right)
        self.div_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_sensitivity_layout(self, parent):
        p = self.px
        ctrl = ttk.Frame(parent, style="Panel.TFrame", width=p(320))
        ctrl.pack(side="left", fill="y", padx=p(16), pady=p(8))
        ctrl.pack_propagate(False)

        ttk.Label(ctrl, text="Strategy", style="Field.TLabel").pack(anchor="w")
        self.sens_strat = tk.StringVar(value="StatArb")
        self._sens_strat_names = {STRATEGY_INFO[k][0]: k for k in SWEEP_PARAMS}
        self._sens_strat_display = tk.StringVar(value=STRATEGY_INFO["StatArb"][0])
        strat_combo = ttk.Combobox(ctrl, textvariable=self._sens_strat_display, state="readonly",
                                   values=list(self._sens_strat_names))
        strat_combo.pack(fill="x", pady=(p(2), p(10)))
        strat_combo.bind("<<ComboboxSelected>>", self._on_sens_strat_display)

        ttk.Label(ctrl, text="Setting to vary", style="Field.TLabel").pack(anchor="w")
        self.sens_param = tk.StringVar()
        self.sens_param_combo = ttk.Combobox(ctrl, textvariable=self.sens_param, state="readonly")
        self.sens_param_combo.pack(fill="x", pady=(p(2), p(10)))
        self.sens_param_combo.bind("<<ComboboxSelected>>", self._on_sens_param_change)

        rng = ttk.Frame(ctrl, style="Panel.TFrame")
        rng.pack(fill="x")
        for c in range(3):
            rng.columnconfigure(c, weight=1, uniform="r")
        self.sens_start = Field(rng, "From", "")
        self.sens_end = Field(rng, "To", "")
        self.sens_step = Field(rng, "Step", "")
        for c, f in enumerate((self.sens_start, self.sens_end, self.sens_step)):
            f.grid(row=0, column=c, sticky="ew", padx=(0 if c == 0 else p(4), 0))

        ttk.Label(ctrl, text="Uses the data, capital, cost and risk settings from the left panel. "
                             "Each value is a full backtest, so this can take a minute.",
                  style="Muted.TLabel", wraplength=p(300), justify="left").pack(anchor="w", pady=p(10))
        self.sens_run_btn = ttk.Button(ctrl, text="Run robustness check", command=self.run_sensitivity)
        self.sens_run_btn.pack(fill="x")
        self.sens_verdict = ttk.Label(ctrl, text="", style="Panel.TLabel", wraplength=p(300),
                                      justify="left", font=self.F_BOLD)
        self.sens_verdict.pack(anchor="w", pady=(p(14), p(6)))
        self.sens_txt = tk.Text(ctrl, wrap="none", bg=PANEL, fg=FG, relief="flat", bd=0,
                                font=self.F_MONO, height=12, highlightthickness=0)
        self.sens_txt.pack(fill="both", expand=True)
        self.sens_txt.config(state="disabled")

        right = ttk.Frame(parent, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(0, p(8)), pady=p(8))
        self.sens_fig, self.sens_ax, self.sens_canvas = self._figure(right)
        self.sens_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._on_sens_strat_change()

    def _on_sens_strat_display(self, event=None):
        self.sens_strat.set(self._sens_strat_names[self._sens_strat_display.get()])
        self._on_sens_strat_change()

    def _on_sens_strat_change(self, event=None):
        key = self.sens_strat.get()
        self._sens_strat_display.set(STRATEGY_INFO[key][0])
        params = list(SWEEP_PARAMS[key])
        self.sens_param_combo["values"] = params
        self.sens_param.set(params[0])
        self._on_sens_param_change()

    def _on_sens_param_change(self, event=None):
        _, start, end, step = SWEEP_PARAMS[self.sens_strat.get()][self.sens_param.get()]
        self.sens_start.var.set(str(start))
        self.sens_end.var.set(str(end))
        self.sens_step.var.set(str(step))

    def _placeholder(self, ax, canvas, title, text):
        ax.clear()
        ax.set_title(title, color=FG)
        ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        canvas.draw()

    def _blank_axes(self):
        msg = "Run a backtest to see results"
        self._placeholder(self.equity_ax, self.equity_canvas, "Growth of each strategy", msg)
        self._placeholder(self.dd_ax, self.dd_canvas, "Drawdowns", msg)
        self._placeholder(self.var_ax, self.var_canvas, "Value at Risk", msg)
        self._placeholder(self.div_ax, self.div_canvas, "Combined portfolio", msg)
        self._placeholder(self.sens_ax, self.sens_canvas, "Robustness check",
                          "Choose a strategy and setting, then press Run robustness check")
        self._placeholder(self.heat_ax, self.heat_canvas, "", "")

    def _style_time_axis(self, ax):
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.grid(alpha=0.35)

    # ---------------------------------------------------------------- run

    def run_backtest(self):
        try:
            cfg = self._read_config()
        except Exception as e:
            messagebox.showerror("Check your settings", str(e))
            return
        if not cfg["agents"]:
            messagebox.showwarning("No strategy selected", "Tick at least one strategy in step 2.")
            return
        self.run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Running…")
        threading.Thread(target=self._run_thread, args=(cfg,), daemon=True).start()

    @staticmethod
    def _split(text, upper=False):
        items = [s.strip() for s in text.split(",") if s.strip()]
        return [s.upper() for s in items] if upper else items

    @staticmethod
    def _pct(field, default=None):
        """Read a field typed as a percentage ('25' -> 0.25). Blank -> default."""
        v = field.get_float(float("nan"))
        if np.isnan(v):
            return default
        return v / 100.0

    def _read_config(self):
        source = self.data_source.get()
        if source == "real":
            symbols = self._split(self.real_symbols_entry.get())
        elif source == "synthetic":
            symbols = self._split(self.symbols_entry.get())
        else:
            symbols = self._split(self.yf_symbols_entry.get(), upper=True)
        if not symbols:
            raise ValueError("Enter at least one stock in step 1.")

        capital = self.capital_entry.get_float(1_000_000)
        if capital <= 0:
            raise ValueError("Starting capital must be positive.")
        risk = self._pct(self.risk_entry, 0.01)
        max_dd = self._pct(self.max_dd_entry)
        max_pos = self._pct(self.max_pos_entry)
        max_lev = self.max_lev_entry.get_float(-1.0)
        var_conf = self.var_conf_entry.get_float(95.0)
        var_conf = var_conf / 100.0 if var_conf > 1 else var_conf
        if not 0.5 <= var_conf < 1:
            raise ValueError("VaR confidence should be between 50 and 99.9 (%).")

        friction = FrictionConfig(
            commission_bps=self.commission_entry.get_float(1.0),
            spread_bps=self.spread_entry.get_float(2.0),
            impact_coefficient=self.impact_entry.get_float(0.10),
            latency_bars=max(self.latency_entry.get_int(1), 0),
            borrow_cost_bps=self.borrow_cost_entry.get_float(0.0),
        )
        self.var_conf = var_conf

        agents = []
        if self.use_statarb.get():
            pair = tuple(self._split(self.pair_entry.get(), upper=(source == "yfinance")))
            if len(pair) != 2:
                raise ValueError("Pairs trading needs exactly two stocks, e.g. ICICIBANK,AXISBANK.")
            for s in pair:
                if s not in symbols:
                    raise ValueError(f"Pairs trading: '{s}' is not in your stock list (step 1).")
            agents.append(("StatArb", dict(pair=pair, lookback=self.lookback_entry.get_int(60),
                                           entry_z=self.entryz_entry.get_float(2.0))))
        if self.use_ml.get():
            ml_syms = self._split(self.ml_symbols_entry.get(), upper=(source == "yfinance"))
            if not ml_syms:
                raise ValueError("Machine-learning strategy: enter at least one stock to trade.")
            for s in ml_syms:
                if s not in symbols:
                    raise ValueError(f"Machine-learning strategy: '{s}' is not in your stock list (step 1).")
            agents.append(("MLFactor", dict(symbols=ml_syms, train_window=self.train_window_entry.get_int(150),
                                            refit_every=self.refit_entry.get_int(20))))
        if self.use_bah.get():
            agents.append(("BuyAndHold", dict(symbols=symbols)))
        if self.use_mom.get():
            top_n = self.mom_top_n.get_int(2)
            if len(symbols) < 2 * top_n:
                raise ValueError(f"Momentum with {top_n} stocks per side needs at least {2 * top_n} "
                                 f"stocks in your list (you have {len(symbols)}).")
            agents.append(("Momentum", dict(symbols=list(symbols), lookback=self.mom_lookback.get_int(252),
                                            top_n=top_n, rebalance_every=21)))
        if self.use_mr.get():
            agents.append(("MeanReversion", dict(symbols=list(symbols), bb_period=self.mr_period.get_int(20),
                                                 bb_std=self.mr_std.get_float(2.0))))

        cfg = dict(symbols=symbols, capital=capital, risk=risk, friction=friction, agents=agents,
                   var_conf=var_conf, source=source, max_drawdown_pct=max_dd, max_position_pct=max_pos,
                   max_gross_leverage=max_lev if max_lev > 0 else None)
        if source == "real":
            cfg["csv_dir"] = self.csv_dir
            missing = [s for s in symbols if not os.path.exists(os.path.join(self.csv_dir, f"{s}.csv"))]
            if missing:
                raise ValueError(f"No data file for {', '.join(missing)} in the data folder.")
        elif source == "synthetic":
            cfg["n_days"] = self.days_entry.get_int(500)
            cfg["seed"] = self.seed_entry.get_int(7)
            cfg["pairs_for_data"] = [tuple(self._split(self.pair_entry.get()))] if self.use_statarb.get() else []
        else:
            cfg["start"] = self.yf_start_entry.get() or None
            cfg["end"] = self.yf_end_entry.get() or None
        return cfg

    def _make_data_handler_factory(self, cfg):
        source = cfg["source"]
        if source == "real":
            return lambda q: HistoricCSVDataHandler(q, csv_dir=cfg["csv_dir"], symbol_list=cfg["symbols"])
        if source == "synthetic":
            return lambda q: SyntheticMarketData(q, cfg["symbols"], pairs=cfg["pairs_for_data"],
                                                 n_days=cfg["n_days"], seed=cfg["seed"])
        cache_dir = os.path.join(PROJECT_ROOT, "data", "_yf_cache")
        return lambda q: YFinanceDataHandler(q, symbol_list=cfg["symbols"], start=cfg["start"],
                                             end=cfg["end"], cache_dir=cache_dir)

    def _build_benchmark_equity(self, cfg):
        dh = self._make_data_handler_factory(cfg)(queue.Queue())
        closes = pd.DataFrame({s: dh.symbol_data[s]["Close"] for s in cfg["symbols"]})
        bench = (closes / closes.iloc[0]).mean(axis=1) * cfg["capital"]
        bench.name = "equity"
        return bench

    @staticmethod
    def _strategy_factory(kind, params):
        p = params
        if kind == "StatArb":
            return lambda dh, q: StatArbStrategy(dh, q, pair=p["pair"], lookback=p["lookback"], entry_z=p["entry_z"])
        if kind == "MLFactor":
            return lambda dh, q: MLFactorStrategy(dh, q, symbols=p["symbols"], train_window=p["train_window"],
                                                  refit_every=p["refit_every"])
        if kind == "BuyAndHold":
            return lambda dh, q: BuyAndHoldStrategy(dh, q, symbols=p["symbols"])
        if kind == "Momentum":
            return lambda dh, q: MomentumStrategy(dh, q, symbols=p["symbols"], lookback=p["lookback"],
                                                  top_n=p["top_n"], rebalance_every=p["rebalance_every"])
        return lambda dh, q: MeanReversionStrategy(dh, q, symbols=p["symbols"], bb_period=p["bb_period"],
                                                   bb_std=p["bb_std"])

    def _run_thread(self, cfg):
        try:
            results = {}
            n_agents = len(cfg["agents"])
            data_factory = self._make_data_handler_factory(cfg)
            for i, (kind, params) in enumerate(cfg["agents"]):
                bt = Backtest(
                    data_handler_factory=data_factory, strategy_factory=self._strategy_factory(kind, params),
                    symbols=cfg["symbols"], initial_capital=cfg["capital"], friction=cfg["friction"],
                    risk_per_trade=cfg["risk"], label=kind, max_drawdown_pct=cfg["max_drawdown_pct"],
                    max_position_pct=cfg["max_position_pct"], max_gross_leverage=cfg["max_gross_leverage"],
                )
                name = STRATEGY_INFO[kind][0]

                def progress_cb(cur, total, i=i, n=n_agents, name=name):
                    if cur % 25 == 0 or cur == total:
                        self.result_queue.put(("progress", (i + cur / max(total, 1)) / n,
                                               f"Running {name} ({i + 1} of {n}): day {cur:,} of {total:,}"))

                results[kind] = bt.run(progress_callback=progress_cb)
            bench_equity = self._build_benchmark_equity(cfg)
            self.result_queue.put(("done", results, bench_equity, cfg["var_conf"]))
        except Exception as e:
            import traceback
            self.result_queue.put(("error", str(e), traceback.format_exc(), None))

    def _poll_queue(self):
        try:
            while True:
                item = self.result_queue.get_nowait()
                if item[0] == "progress":
                    _, frac, msg = item
                    self.progress["value"] = frac * 100
                    self.status_var.set(msg)
                elif item[0] == "sens_done":
                    _, df, interpret, param_name = item
                    self._render_sensitivity(df, interpret, param_name)
                    self.status_var.set("Robustness check finished.")
                    self.progress["value"] = 100
                    self.sens_run_btn.config(state="normal")
                elif item[0] == "done":
                    _, results, bench_equity, var_conf = item
                    self.all_results = results
                    self.benchmark_equity = bench_equity
                    self._render_results(var_conf)
                    n = len(results)
                    self.status_var.set(f"Done: {n} strateg{'y' if n == 1 else 'ies'} compared. "
                                        f"Use the tabs above for details.")
                    self.progress["value"] = 100
                    self.run_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                elif item[0] == "error":
                    _, msg, tb, _ = item
                    self.status_var.set("Something went wrong. See the error message.")
                    self.progress["value"] = 0
                    self.run_btn.config(state="normal")
                    self.sens_run_btn.config(state="normal")
                    print(tb)
                    messagebox.showerror("Run failed", msg)
        except queue.Empty:
            pass
        self._poll_id = self.after(150, self._poll_queue)

    # ---------------------------------------------------------------- render

    def _render_results(self, var_conf):
        table = build_comparison_table(self.all_results, benchmark_equity=self.benchmark_equity,
                                       var_confidence=var_conf)
        self.comparison_table = table
        self._replotted_equity()
        self._replotted_drawdown()
        self._replotted_var(var_conf)
        self._fill_metrics(table)
        self._replotted_tca()
        self._replotted_dashboard(table)
        self._init_diversification_sliders()
        self.notebook.select(0)

    def _color(self, label):
        return STRATEGY_COLORS.get(label, ACCENT)

    def _name(self, label):
        return SHORT_NAMES.get(label, label)

    def _replotted_equity(self):
        ax = self.equity_ax
        ax.clear()
        if not self.all_results:
            return
        pct = self.eq_toggle_var.get() == "pct"
        for label, res in self.all_results.items():
            eq = res["equity"]
            if len(eq):
                y = (eq / eq.iloc[0] - 1) * 100 if pct else eq
                ax.plot(y.index, y.values, label=self._name(label), color=self._color(label), linewidth=1.8)
        if self.benchmark_equity is not None:
            first_eq = next(iter(self.all_results.values()))["equity"]
            be = self.benchmark_equity.reindex(first_eq.index).ffill()
            y = (be / be.iloc[0] - 1) * 100 if pct else be
            ax.plot(y.index, y.values, label="Benchmark (buy & hold all)", color=MUTED,
                    linestyle="--", linewidth=1.3)
        if pct:
            ax.axhline(0, color=BORDER, linewidth=1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title("Growth of each strategy" + (" (% return)" if pct else " (portfolio value)"))
        ax.legend(loc="upper left")
        self._style_time_axis(ax)
        self.equity_canvas.draw()

    def _replotted_drawdown(self):
        ax = self.dd_ax
        ax.clear()
        for label, res in self.all_results.items():
            eq = res["equity"]
            if len(eq):
                dd = performance.drawdown_series(eq) * 100
                ax.fill_between(dd.index, dd.values, 0, alpha=0.12, color=self._color(label))
                ax.plot(dd.index, dd.values, color=self._color(label), label=self._name(label), linewidth=1.4)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title("Distance below previous peak")
        ax.legend(loc="lower left")
        self._style_time_axis(ax)
        self.dd_canvas.draw()

    def _replotted_var(self, var_conf):
        ax = self.var_ax
        ax.clear()
        for label, res in self.all_results.items():
            eq = res["equity"]
            if len(eq) > 25:
                rets = performance.returns_from_equity(eq)
                var = performance.rolling_var(rets, window=20, confidence=var_conf) * 100
                ax.plot(var.index, var.values, label=self._name(label), color=self._color(label), linewidth=1.4)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set_title(f"1-day loss not exceeded on {var_conf * 100:.0f}% of days (rolling 20-day VaR)")
        ax.legend(loc="upper left")
        self._style_time_axis(ax)
        self.var_canvas.draw()

    def _fill_metrics(self, table):
        strategies = list(table.index)
        self._set_columns(self.metrics_tree, [("metric", "Metric", 170, "w", False)] +
                          [(s, self._name(s), 100, "center", True) for s in strategies])
        self._metric_meanings = {}
        for i, col in enumerate(table.columns):
            if col.startswith("VaR"):
                conf = col.split("%")[0].split()[-1]
                which = "latest" if "latest" in col else "average"
                name = f"1-day VaR {conf}% ({which})"
                meaning = (f"the 1-day loss that was not exceeded on {conf}% of days "
                           f"({'over the last 20 days' if which == 'latest' else 'averaged over the whole run'}).")
            else:
                name, meaning = METRIC_INFO.get(col, (col, ""))
            vals = [format_metric(col, table.loc[s, col]) for s in strategies]
            iid = self.metrics_tree.insert("", "end", values=[name] + vals, tags=("odd",) if i % 2 else ())
            self._metric_meanings[iid] = (name, meaning[0].lower() + meaning[1:] if meaning else "")

    def _tile(self, parent, title, value, color=FG):
        p = self.px
        t = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        tk.Label(t, text=title, bg=CARD, fg=MUTED, font=self.F_SMALL).pack(anchor="w", padx=p(12), pady=(p(8), 0))
        tk.Label(t, text=value, bg=CARD, fg=color, font=(UI_FAMILY, 15, "bold")).pack(anchor="w", padx=p(12), pady=(0, p(8)))
        return t

    def _replotted_tca(self):
        p = self.px
        for w in self.tca_tiles.winfo_children():
            w.destroy()
        frames = []
        for label, res in self.all_results.items():
            if len(res["tca"]):
                df = res["tca"].copy()
                df.insert(0, "strategy", label)
                frames.append(df)
        self.cost_tree.delete(*self.cost_tree.get_children())
        self.tca_tree.delete(*self.tca_tree.get_children())
        if not frames:
            self.tca_summary_var.set("No trades were made.")
            ttk.Label(self.tca_tiles, text="No trades were made.", style="Panel.TLabel").pack(anchor="w")
            return

        tca = pd.concat(frames, ignore_index=True)
        total_comm, total_slip = tca["commission"].sum(), tca["slippage_cost"].sum()
        avg_slip, avg_short = tca["slip_bps"].mean(), tca["shortfall_bps"].mean()
        self.tca_summary_var.set(
            f"Commission {total_comm:,.0f} | Slippage {total_slip:,.0f} | Avg slippage {avg_slip:.2f} bps | "
            f"Avg shortfall {avg_short:.2f} bps | Trades {len(tca):,}")
        tiles = [("Trades", f"{len(tca):,}", FG), ("Commission paid", f"{total_comm:,.0f}", FG),
                 ("Slippage cost", f"{total_slip:,.0f}", FG),
                 ("Avg slippage", f"{avg_slip:.2f} bps", FG),
                 ("Avg shortfall", f"{avg_short:+.2f} bps", GOOD if avg_short <= 0 else ACCENT2)]
        for c, (title, value, color) in enumerate(tiles):
            self.tca_tiles.columnconfigure(c, weight=1, uniform="t")
            self._tile(self.tca_tiles, title, value, color).grid(
                row=0, column=c, sticky="ew", padx=(0 if c == 0 else p(8), 0))

        # cost by strategy
        self._set_columns(self.cost_tree, [
            ("s", "Strategy", 125, "w", True), ("n", "Trades", 75, "e", True),
            ("c", "Commission", 95, "e", True), ("sl", "Slippage", 90, "e", True),
            ("b", "Borrow", 80, "e", True), ("t", "Total cost", 95, "e", True),
            ("pct", "% of capital", 100, "e", True)])
        for i, (label, res) in enumerate(self.all_results.items()):
            tr = res["trades"]
            comm = tr["commission"].sum() if len(tr) else 0.0
            slip = tr["slippage_cost"].sum() if len(tr) else 0.0
            borrow = res.get("borrow_cost", 0.0)
            total = comm + slip + borrow
            start = res["equity"].iloc[0] if len(res["equity"]) else 1.0
            self.cost_tree.insert("", "end", tags=("odd",) if i % 2 else (), values=[
                self._name(label), f"{len(tr):,}", f"{comm:,.0f}", f"{slip:,.0f}", f"{borrow:,.0f}",
                f"{total:,.0f}", f"{total / start * 100:.2f}%"])
        self.cost_tree.configure(height=len(self.all_results))

        # trade log
        self._set_columns(self.tca_tree, [
            ("strategy", "Strategy", 115, "w", True), ("date", "Date", 90, "center", False),
            ("symbol", "Stock", 90, "center", False), ("side", "Side", 45, "center", False),
            ("qty", "Shares", 65, "e", False), ("fill", "Price", 75, "e", False),
            ("slip", "Slip (bps)", 75, "e", False), ("short", "Shortfall (bps)", 105, "e", False),
            ("comm", "Fee", 65, "e", False)])
        recent = tca.sort_values("timestamp").tail(300).iloc[::-1]
        for i, (_, r) in enumerate(recent.iterrows()):
            self.tca_tree.insert("", "end", tags=("odd",) if i % 2 else (), values=[
                self._name(r["strategy"]), pd.Timestamp(r["timestamp"]).strftime("%Y-%m-%d"), r["symbol"],
                "Buy" if r["direction"] == "BUY" else "Sell", f"{r['quantity']:,.0f}",
                f"{r['fill_price']:,.2f}", f"{r['slip_bps']:.2f}", f"{r['shortfall_bps']:+.1f}",
                f"{r['commission']:,.2f}"])

    def _replotted_dashboard(self, table):
        p = self.px
        self.dash_empty.pack_forget()
        self.dash_results.pack(fill="both", expand=True)

        winner = None
        if not table.empty and "Sharpe" in table.columns:
            winner = table["Sharpe"].idxmax()
            self.win_lbl.config(text=f"★  Best risk-adjusted result: {self._name(winner)}   "
                                     f"(Sharpe ratio {table.loc[winner, 'Sharpe']:.2f})")
        else:
            self.win_lbl.config(text="No results to compare.")

        for w in self.cards_frame.winfo_children():
            w.destroy()
        for c, (strat, row) in enumerate(table.iterrows()):
            self.cards_frame.columnconfigure(c, weight=1, uniform="cards")
            is_win = strat == winner
            card = tk.Frame(self.cards_frame, bg=CARD, highlightthickness=p(2) if is_win else 1,
                            highlightbackground=ACCENT if is_win else BORDER)
            card.grid(row=0, column=c, sticky="nsew", padx=(0 if c == 0 else p(10), 0))
            tk.Label(card, text=self._name(strat), bg=CARD, fg=self._color(strat), font=self.F_BOLD,
                     anchor="w").pack(fill="x", padx=p(12), pady=(p(10), 0))
            sharpe = row.get("Sharpe", np.nan)
            tk.Label(card, text=f"{sharpe:.2f}", bg=CARD, fg=GOOD if sharpe > 0 else BAD,
                     font=self.F_BIG, anchor="w").pack(fill="x", padx=p(12))
            tk.Label(card, text="Sharpe ratio", bg=CARD, fg=MUTED, font=self.F_SMALL,
                     anchor="w").pack(fill="x", padx=p(12))
            stats = tk.Frame(card, bg=CARD)
            stats.pack(fill="x", padx=p(12), pady=(p(8), p(10)))
            stats.columnconfigure(1, weight=1)
            cagr = row.get("CAGR", np.nan)
            for r, (k, v, color) in enumerate([
                ("Per year", f"{cagr * 100:+.1f}%", GOOD if cagr > 0 else BAD),
                ("Worst fall", f"{row.get('Max Drawdown', np.nan) * 100:.1f}%", FG),
                ("Trades", f"{int(row.get('# Trades', 0)):,}", FG),
            ]):
                tk.Label(stats, text=k, bg=CARD, fg=MUTED, font=self.F_SMALL).grid(row=r, column=0, sticky="w")
                tk.Label(stats, text=v, bg=CARD, fg=color, font=self.F_BOLD).grid(row=r, column=1, sticky="e")

        # plain-English summary
        t = self.dash_txt
        t.config(state="normal")
        t.delete("1.0", tk.END)
        for strat, row in table.iterrows():
            t.insert(tk.END, self._name(strat), strat)
            t.insert(tk.END, f"  {row['Total Return'] * 100:+.1f}% in total ({row['CAGR'] * 100:+.1f}% a year), "
                             f"worst fall {row['Max Drawdown'] * 100:.1f}%, {int(row['# Trades']):,} trades.")
            beta = row.get("Beta", np.nan)
            if not pd.isna(beta):
                link = ("moved closely with the market" if abs(beta) > 0.7 else
                        "partly followed the market" if abs(beta) > 0.3 else
                        "was largely independent of the market")
                t.insert(tk.END, f" It {link} (beta {beta:.2f}).", "muted")
            halted = self.all_results.get(strat, {}).get("halted_at")
            if halted is not None:
                t.insert(tk.END, f" Hit the drawdown limit on {pd.Timestamp(halted):%d %b %Y} and stopped trading.", "warn")
            t.insert(tk.END, "\n")
        if self.benchmark_equity is not None and len(self.benchmark_equity) > 1:
            be = self.benchmark_equity
            t.insert(tk.END, f"For comparison, simply buying all the selected stocks equally and holding "
                             f"returned {(be.iloc[-1] / be.iloc[0] - 1) * 100:+.1f}% ({performance.cagr(be) * 100:+.1f}% a year).",
                     "muted")
        t.config(state="disabled")

        # correlation heatmap (only meaningful with 2+ strategies)
        corr = performance.strategy_correlation_matrix({k: v["equity"] for k, v in self.all_results.items()})
        if corr.empty:
            self.heat_frame.pack_forget()
            return
        self.heat_frame.pack(side="right", fill="y", padx=(p(14), 0), before=t.master)
        ax = self.heat_ax
        ax.clear()
        ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        short = [self._name(c) for c in corr.columns]
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=40, ha="right", rotation_mode="anchor", fontsize=9)
        ax.set_yticklabels(short, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        for r in range(corr.shape[0]):
            for c in range(corr.shape[1]):
                ax.text(c, r, f"{corr.values[r, c]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", color="#111", fontsize=8.5)
        self.heat_canvas.draw()

    # ---------------------------------------------------------------- combine

    def _init_diversification_sliders(self):
        p = self.px
        for w in self.sliders_frame.winfo_children():
            w.destroy()
        self.sliders = {}
        strats = list(self.all_results.keys())
        if len(strats) < 2:
            ttk.Label(self.sliders_frame, text="Run a backtest with two or more strategies to combine them.",
                      style="Muted.TLabel", wraplength=p(320)).pack(anchor="w", pady=p(10))
            self.div_stats_lbl.config(text="")
            self._placeholder(self.div_ax, self.div_canvas, "Combined portfolio", "Needs two or more strategies")
            return
        ttk.Label(self.sliders_frame, text="Share of capital", style="Panel.TLabel",
                  font=self.F_BOLD).pack(anchor="w")
        ttk.Label(self.sliders_frame, text="Shares are rescaled to add up to 100%.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, p(8)))
        default = 1.0 / len(strats)
        for s in strats:
            row = ttk.Frame(self.sliders_frame, style="Panel.TFrame")
            row.pack(fill="x", pady=p(3))
            tk.Label(row, text=self._name(s), bg=PANEL, fg=self._color(s), font=self.F_BOLD,
                     anchor="w").pack(anchor="w")
            line = ttk.Frame(row, style="Panel.TFrame")
            line.pack(fill="x")
            val = ttk.Label(line, text=f"{default * 100:.0f}%", width=5, style="Panel.TLabel", anchor="e")
            scale = tk.Scale(line, from_=0, to=1.0, resolution=0.01, orient="horizontal", showvalue=0,
                             bg=ACCENT, troughcolor=CARD, activebackground="#7ee3d8", highlightthickness=0,
                             bd=0, sliderrelief="flat", sliderlength=self.px(22), width=self.px(12),
                             command=lambda v, vl=val: vl.config(text=f"{float(v) * 100:.0f}%"))
            scale.set(default)
            scale.pack(side="left", fill="x", expand=True, padx=(0, self.px(8)))
            val.pack(side="right")
            scale.bind("<ButtonRelease-1>", lambda e: self.recalculate_diversification())
            self.sliders[s] = scale
        self.recalculate_diversification()

    def recalculate_diversification(self):
        raw = {s: sc.get() for s, sc in self.sliders.items()}
        total = sum(raw.values())
        weights = {s: (w / total if total > 0 else 1.0 / len(raw)) for s, w in raw.items()}
        eq_dict = {lbl: res["equity"] for lbl, res in self.all_results.items()}
        combined = performance.combined_portfolio_equity(eq_dict, weights)
        if combined.empty:
            self.div_stats_lbl.config(text="Could not combine these strategies.", foreground=BAD)
            return
        ratio = performance.diversification_ratio(eq_dict, weights)
        stats = performance.summarize(combined)

        ax = self.div_ax
        ax.clear()
        for s, res in self.all_results.items():
            eq = res["equity"]
            y = (eq / eq.iloc[0] - 1) * 100
            ax.plot(y.index, y.values, label=f"{self._name(s)} ({weights[s] * 100:.0f}%)",
                    color=self._color(s), alpha=0.4, linewidth=1.1)
        y = (combined / combined.iloc[0] - 1) * 100
        ax.plot(y.index, y.values, label="Combined portfolio", color="#ffffff", linewidth=2.4)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax.set_title("Combined portfolio vs. each strategy")
        ax.legend(loc="upper left")
        self._style_time_axis(ax)
        self.div_canvas.draw()

        g = self.div_stats_grid
        for w in g.winfo_children():
            w.destroy()
        g.columnconfigure(1, weight=1)
        ttk.Label(g, text="Combined portfolio", style="Panel.TLabel", font=self.F_BOLD).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, self.px(4)))
        for r, (k, v) in enumerate([("Sharpe ratio", f"{stats['Sharpe']:.2f}"),
                                    ("Return per year", f"{stats['CAGR'] * 100:+.2f}%"),
                                    ("Worst drawdown", f"{stats['Max Drawdown'] * 100:.2f}%"),
                                    ("Diversification ratio", f"{ratio:.2f}")], start=1):
            ttk.Label(g, text=k, style="Muted.TLabel", font=self.F_BASE).grid(row=r, column=0, sticky="w")
            ttk.Label(g, text=v, style="Panel.TLabel", font=self.F_BOLD).grid(row=r, column=1, sticky="e")
        if sum(w > 0.001 for w in weights.values()) < 2:
            verdict, color = "All capital is in one strategy, so there is nothing to diversify.", MUTED
        elif ratio > 1.05:
            verdict, color = f"Combining them reduced risk (diversification ratio {ratio:.2f}).", GOOD
        else:
            verdict, color = "Little diversification benefit: these strategies tend to move together.", ACCENT2
        self.div_stats_lbl.config(foreground=color, text=verdict)

    # ---------------------------------------------------------------- robustness

    def run_sensitivity(self):
        try:
            cfg = self._read_config()
        except Exception as e:
            messagebox.showerror("Check your settings", str(e))
            return

        strat = self.sens_strat.get()
        display = self.sens_param.get()
        param = SWEEP_PARAMS[strat][display][0]
        start, end, step = self.sens_start.get_float(), self.sens_end.get_float(), self.sens_step.get_float()
        if step <= 0 or start >= end:
            messagebox.showerror("Check the range", "'To' must be greater than 'From', and 'Step' must be positive.")
            return
        values = [round(float(v), 6) for v in np.arange(start, end + step / 2, step)]
        if param in INT_PARAMS:
            values = sorted({int(round(v)) for v in values if v >= 1})
        if not values:
            messagebox.showerror("Check the range", "The range contains no valid values.")
            return

        base_cfg = dict(
            data_handler_factory=self._make_data_handler_factory(cfg), symbols=cfg["symbols"],
            initial_capital=cfg["capital"], friction=cfg["friction"], risk_per_trade=cfg["risk"],
            max_drawdown_pct=cfg["max_drawdown_pct"], max_position_pct=cfg["max_position_pct"],
            max_gross_leverage=cfg["max_gross_leverage"], label=f"Sweep_{strat}",
        )
        upper = cfg["source"] == "yfinance"
        if strat == "StatArb":
            pair = tuple(self._split(self.pair_entry.get(), upper))
            if len(pair) != 2 or any(s not in cfg["symbols"] for s in pair):
                messagebox.showerror("Check your settings", "Pairs trading needs two stocks from your stock list.")
                return
            base = dict(pair=pair, lookback=self.lookback_entry.get_int(60), entry_z=self.entryz_entry.get_float(2.0))
        elif strat == "MLFactor":
            syms = self._split(self.ml_symbols_entry.get(), upper)
            if not syms or any(s not in cfg["symbols"] for s in syms):
                messagebox.showerror("Check your settings", "Machine-learning stocks must all be in your stock list.")
                return
            base = dict(symbols=syms, train_window=self.train_window_entry.get_int(150),
                        refit_every=self.refit_entry.get_int(20))
        elif strat == "MeanReversion":
            base = dict(symbols=list(cfg["symbols"]), bb_period=self.mr_period.get_int(20),
                        bb_std=self.mr_std.get_float(2.0))
        else:
            base = dict(symbols=list(cfg["symbols"]), lookback=self.mom_lookback.get_int(252),
                        top_n=self.mom_top_n.get_int(2), rebalance_every=21)

        def builder(val, base=base, param=param, strat=strat):
            params = dict(base)
            params[param] = int(val) if param in INT_PARAMS else float(val)
            return self._strategy_factory(strat, params)

        base_cfg["strategy_factory_builder"] = builder
        self.sens_run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set(f"Robustness check: {self._name(strat)}, {display}…")
        threading.Thread(target=self._run_sweep_thread, args=(base_cfg, cfg, display, values), daemon=True).start()

    def _run_sweep_thread(self, base_cfg, cfg, param_name, values):
        try:
            bench = self._build_benchmark_equity(cfg)

            def prog_cb(i, total):
                self.result_queue.put(("progress", (i + 1) / max(total, 1),
                                       f"Robustness check: {param_name} = {values[i]} done ({i + 1} of {total})"))

            df = sensitivity.run_sensitivity_sweep(base_cfg, param_name, values, benchmark_equity=bench,
                                                   progress_callback=prog_cb)
            self.result_queue.put(("sens_done", df, sensitivity.interpret_sensitivity(df, param_name), param_name))
        except Exception as e:
            import traceback
            self.result_queue.put(("error", f"Robustness check failed: {e}", traceback.format_exc(), None))

    def _render_sensitivity(self, df, interpret, param_name):
        # clf() also removes the twin axis from any previous run
        self.sens_fig.clf()
        ax = self.sens_ax = self.sens_fig.add_subplot(111)
        ax.plot(df.index, df["Sharpe"], marker="o", color=ACCENT, linewidth=2, label="Sharpe ratio")
        ax.set_ylabel("Sharpe ratio", color=ACCENT)
        ax.tick_params(axis="y", labelcolor=ACCENT)
        ax.set_xlabel(param_name)
        ax.axhline(0, color=BORDER, linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(df.index, df["CAGR"] * 100, marker="s", color=ACCENT2, linestyle="--", linewidth=1.6)
        ax2.set_ylabel("Return per year (%)", color=ACCENT2)
        ax2.tick_params(axis="y", labelcolor=ACCENT2)
        ax2.spines["right"].set_visible(True)
        ax.set_title(f"How results change with {param_name.lower()}")
        ax.grid(alpha=0.35)
        self.sens_canvas.draw()

        self.sens_verdict.config(text=interpret)
        table = pd.DataFrame({
            "Sharpe": df["Sharpe"].map(lambda v: f"{v:.2f}"),
            "Ret/yr": df["CAGR"].map(lambda v: f"{v * 100:+.1f}%"),
            "Worst DD": df["Max Drawdown"].map(lambda v: f"{v * 100:.1f}%"),
            "Trades": df["# Trades"].map(lambda v: f"{int(v)}"),
        }, index=df.index)
        table.index.name = "Value"
        self.sens_txt.config(state="normal")
        self.sens_txt.delete("1.0", tk.END)
        self.sens_txt.insert("1.0", table.to_string())
        self.sens_txt.config(state="disabled")

    # ---------------------------------------------------------------- export

    def _export_dialog(self):
        if not self.all_results:
            messagebox.showwarning("Nothing to export", "Run a backtest first.")
            return
        out_dir = filedialog.askdirectory(title="Choose a folder for the results")
        if not out_dir:
            return
        try:
            export_results(self.all_results, self.comparison_table, out_dir)
            export_html_report(self.all_results, self.comparison_table, benchmark_equity=self.benchmark_equity,
                               output_path=os.path.join(out_dir, "backtest_report.html"))
            messagebox.showinfo("Exported", f"CSV files and backtest_report.html saved to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


def main():
    app = TradingArenaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
