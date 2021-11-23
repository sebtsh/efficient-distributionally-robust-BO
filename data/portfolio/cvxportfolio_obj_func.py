import pandas as pd
import cvxportfolio as cp
import quandl
import numpy as np
import pickle
from scipy.stats import qmc
from pathlib import Path
from tqdm import trange


def extract_results(result):
    portfolio_return = result.returns.mean() * 100 * result.PPY
    excess_return = result.excess_returns.mean() * 100 * result.PPY
    excess_risk = result.excess_returns.std() * 100 * np.sqrt(result.PPY)
    sharpe_ratio = result.sharpe_ratio
    max_drawdown = result.max_drawdown
    return [portfolio_return, excess_return, excess_risk, sharpe_ratio, max_drawdown]


quandl.ApiConfig.api_key = 'XB2H55DxdNiJjBbLvCzu'
Path("").mkdir(parents=True, exist_ok=True)
tickers = ['AMZN', 'GOOGL', 'TSLA', 'NKE']
start_date = '2012-01-01'
end_date = '2016-12-31'
returns = pd.DataFrame(dict([(ticker, quandl.get('WIKI/' + ticker,
                                                 start_date=start_date,
                                                 end_date=end_date)['Adj. Close'].pct_change())
                             for ticker in tickers]))
returns[["USDOLLAR"]] = quandl.get('FRED/DTB3', start_date=start_date, end_date=end_date) / (250 * 100)
returns = returns.fillna(method='ffill').iloc[1:]

r_hat = returns.rolling(window=250, min_periods=250).mean().shift(1).dropna()
Sigma_hat = returns.rolling(window=250, min_periods=250, closed='neither').cov().dropna().droplevel(1)
risk_model = cp.FullSigma(Sigma_hat)
leverage_limit = cp.LeverageLimit(5)

num_samples = 2048

# Generate Sobol sequence
sampler = qmc.Sobol(d=4, scramble=False)
sample = sampler.random(num_samples)
lowers = np.log([0.1, 5.5, 1e-04, 1e-04])  # Work in log scale
uppers = np.log([1000, 8, 1e-02, 1e-03])
scaled_samples = qmc.scale(sample, lowers, uppers)
pickle.dump(scaled_samples, open("data/portfolio/scaled_samples.p", "wb"))

all_results = []
for i in trange(num_samples):
    # Dimensions are [risk_aversion, trade_aversion, holding_cost, bid_ask_spread, borrow_cost]
    params = scaled_samples[i]
    risk_aversion = np.exp(params[0])  # Params are in log scale, so we take the exp now
    trade_aversion = np.exp(params[1])
    holding_cost = 50  # fixed
    bid_ask_spread = np.exp(params[2])
    borrow_cost = np.exp(params[3])

    # Context parameters go here
    tcost_model = cp.TcostModel(half_spread=0.5 * bid_ask_spread)
    hcost_model = cp.HcostModel(borrow_costs=borrow_cost)

    # Action parameters go here
    spo_policy = cp.SinglePeriodOpt(return_forecast=r_hat,
                                    costs=[risk_aversion * risk_model, trade_aversion * tcost_model,
                                           holding_cost * hcost_model],
                                    constraints=[leverage_limit])

    market_sim = cp.MarketSimulator(returns, [tcost_model, hcost_model], cash_key='USDOLLAR')
    init_portfolio = pd.Series(index=returns.columns, data=250000.)
    init_portfolio.USDOLLAR = 0
    results = market_sim.run_multiple_backtest(init_portfolio,
                                               start_time='2013-01-03', end_time='2016-12-31',
                                               policies=[spo_policy])
    extracted_results = extract_results(results[0])
    all_results.append(extracted_results)
    # if (i+1) % 100 == 0:
    #     pickle.dump(np.array(all_results), open("data/portfolio/partial_results_{}.p".format(i), "wb"))

print("All simulations completed successfully")
all_results = np.array(all_results)
scaled_samples = np.array(scaled_samples)

pickle.dump(all_results, open("data/portfolio/all_results.p", "wb"))
pickle.dump(scaled_samples, open("data/portfolio/scaled_samples.p", "wb"))

standardized_returns = (all_results[:, 0] - np.mean(all_results[:, 0])) / np.std(all_results[:, 0])
normalized_samples = (scaled_samples - lowers) / (uppers - lowers)
pickle.dump(standardized_returns, open("data/portfolio/standardized_returns.p", "wb"))
pickle.dump(normalized_samples, open("data/portfolio/normalized_samples.p", "wb"))
