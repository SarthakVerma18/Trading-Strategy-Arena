# tournament.py
import matplotlib.pyplot as plt

def run_tournament(data, agents, backtest_func):
    results = {}
    for agent in agents:
        result = backtest_func(data, agent)
        results[agent.name] = {
            'equity_curve': result["equity_curve"],
            'metrics': {
                'Sharpe Ratio': result["Sharpe Ratio"],
                'Max Drawdown': result["Max Drawdown"],
                'Alpha': result["Alpha"],
                'Beta': result["Beta"],
            }
        }
    return results


def print_ranking(results, metric='sharpe_ratio'):

    ranking = sorted(results.items(), key=lambda x: x[1]['metrics'].get(metric, 0), reverse=True)
    for rank, (name, res) in enumerate(ranking, 1):
        print(f"{rank}. {name} — {metric}: {res['metrics'].get(metric, 'N/A'):.4f}")

def plot_equity_curves(results):
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        res['equity_curve'].plot(label=name)
    plt.title("Strategy Tournament: Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()
