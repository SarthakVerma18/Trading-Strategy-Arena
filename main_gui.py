import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import yfinance as yf
import pandas as pd
import importlib.util
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tournament import run_tournament, plot_equity_curves
from utils.data_loader import load_strategy_from_file
from backtester.backtest import run_backtest
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion import MeanReversionAgent
from agents.rsi import RSIAgent
from agents.SMA import ReverseMomentumAgent


class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Trading Arena")
        root.geometry("700x650")
        root.configure(bg="#f0f0f0")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.single_strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.single_strategy_frame, text="Single Strategy")

        self.tournament_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tournament_frame, text="Compare Strategies")

        self.setup_single_strategy_tab()
        self.setup_tournament_tab()

    def setup_single_strategy_tab(self):
        frame = self.single_strategy_frame

        tk.Label(frame, text="Enter Stock Ticker:").grid(row=0, column=0)
        self.s_ticker_entry = tk.Entry(frame)
        self.s_ticker_entry.grid(row=0, column=1)

        tk.Label(frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0)
        self.s_start_entry = tk.Entry(frame)
        self.s_start_entry.grid(row=1, column=1)

        tk.Label(frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0)
        self.s_end_entry = tk.Entry(frame)
        self.s_end_entry.grid(row=2, column=1)

        tk.Label(frame, text="Select Strategy:").grid(row=3, column=0)
        self.strategy_var = tk.StringVar()
        self.strategy_menu = ttk.Combobox(frame, textvariable=self.strategy_var)
        self.strategy_menu['values'] = ("Momentum", "Mean Reversion", "Relative Strength Index", "Simple Moving Average Crossover", "Upload Custom Strategy")
        self.strategy_menu.current(0)
        self.strategy_menu.grid(row=3, column=1)

        self.run_button = tk.Button(frame, text="Run Backtest", command=self.run_backtest_single)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.output_text = tk.Text(frame, height=10, width=60)
        self.output_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        self.canvas = None

    def run_backtest_single(self):
        ticker = self.s_ticker_entry.get()
        start_date = self.s_start_entry.get()
        end_date = self.s_end_entry.get()
        strategy_choice = self.strategy_var.get()

        if not ticker or not start_date or not end_date:
            messagebox.showerror("Input Error", "Please enter all fields.")
            return

        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df['Close'] = df.get("Adj Close", df.get("Close"))
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to load data: {str(e)}")
            return

        try:
            if strategy_choice == "Momentum":
                agent = MomentumAgent()
            elif strategy_choice == "Mean Reversion":
                agent = MeanReversionAgent()
            elif strategy_choice == "Relative Strength Index":
                agent = RSIAgent()
            elif strategy_choice == "Simple Moving Average Crossover":
                agent = ReverseMomentumAgent()
            else:
                file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
                if not file_path:
                    return
                spec = importlib.util.spec_from_file_location("custom_strategy", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                agent = module.CustomAgent()
        except Exception as e:
            messagebox.showerror("Strategy Error", f"Error loading strategy: {str(e)}")
            return

        try:
            results = run_backtest(df, agent)
        except Exception as e:
            messagebox.showerror("Backtest Error", f"Error during backtest: {str(e)}")
            return

        self.output_text.delete("1.0", tk.END)
        for k, v in results.items():
            if k != "equity_curve":
                self.output_text.insert(tk.END, f"{k}: {v:.4f}\n")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 3))
        results["equity_curve"].plot(ax=ax)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Portfolio Value")

        self.canvas = FigureCanvasTkAgg(fig, master=self.single_strategy_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

    def setup_tournament_tab(self):
        frame = self.tournament_frame

        ttk.Label(frame, text="Select Built-in Strategies to Compare:").pack(anchor='w', padx=10, pady=(10, 0))
        self.builtin_agents = {
            "MomentumAgent": MomentumAgent,
            "MeanReversionAgent": MeanReversionAgent,
            "RSIAgent": RSIAgent
        }
        self.selected_builtins = {}
        for name in self.builtin_agents:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(frame, text=name, variable=var)
            cb.pack(anchor='w', padx=20)
            self.selected_builtins[name] = var

        ttk.Label(frame, text="Or upload custom strategy files (.py):").pack(anchor='w', padx=10, pady=(15, 0))
        upload_btn = ttk.Button(frame, text="Upload Strategies", command=self.upload_strategy_files)
        upload_btn.pack(anchor='w', padx=20)

        self.uploaded_files_listbox = tk.Listbox(frame, height=5)
        self.uploaded_files_listbox.pack(fill='x', padx=20, pady=(5, 15))

        ttk.Label(frame, text="Stock Ticker (e.g. AAPL):").pack(anchor='w', padx=10)
        self.ticker_entry = ttk.Entry(frame)
        self.ticker_entry.pack(fill='x', padx=20, pady=(0, 10))
        self.ticker_entry.insert(0, "AAPL")

        ttk.Label(frame, text="Start Date (YYYY-MM-DD):").pack(anchor='w', padx=10)
        self.start_date_entry = ttk.Entry(frame)
        self.start_date_entry.pack(fill='x', padx=20)

        ttk.Label(frame, text="End Date (YYYY-MM-DD):").pack(anchor='w', padx=10)
        self.end_date_entry = ttk.Entry(frame)
        self.end_date_entry.pack(fill='x', padx=20)

        run_btn = ttk.Button(frame, text="Run Tournament", command=self.run_tournament_clicked)
        run_btn.pack(pady=10)

        self.result_text = tk.Text(frame, height=15)
        self.result_text.pack(fill='both', padx=10, pady=(0, 10))

    def upload_strategy_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Python files", "*.py")])
        for f in files:
            if f not in self.uploaded_files_listbox.get(0, tk.END):
                self.uploaded_files_listbox.insert(tk.END, f)

    def run_tournament_clicked(self):
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get() or "2020-01-01"
        end_date = self.end_date_entry.get() or "2023-01-01"

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Downloading data...\n")

        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("No data downloaded. Check ticker or dates.")
        except Exception as e:
            messagebox.showerror("Data Download Error", str(e))
            return

        agents = []
        for name, var in self.selected_builtins.items():
            if var.get():
                agent_class = self.builtin_agents[name]
                agents.append(agent_class())

        for idx in range(self.uploaded_files_listbox.size()):
            filepath = self.uploaded_files_listbox.get(idx)
            try:
                agent_instance = load_strategy_from_file(filepath)
                agents.append(agent_instance)
            except Exception as e:
                self.result_text.insert(tk.END, f"Failed to load {filepath}: {e}\n")

        if not agents:
            messagebox.showwarning("No Strategies Selected", "Please select or upload at least one strategy.")
            return

        self.result_text.insert(tk.END, f"Running tournament with {len(agents)} strategies...\n")
        self.root.update()

        results = run_tournament(data, agents, run_backtest)
        
        ranking = sorted(results.items(), key=lambda x: x[1]['metrics'].get('sharpe',0), reverse=True)
        self.result_text.insert(tk.END, "Strategy Ranking by Sharpe Ratio:\n")
        for rank, (name, res) in enumerate(ranking, 1):
            sharpe = res['metrics'].get('sharpe', 'N/A')
            self.result_text.insert(tk.END, f"{rank}. {name} — Sharpe Ratio: {sharpe}\n")

        plot_equity_curves(results)


if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()