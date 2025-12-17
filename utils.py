import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns.
    """
    return price_df.pct_change().dropna()


def compute_cumulative_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns from daily returns.
    """
    return (1 + daily_returns).cumprod()


def annualized_return(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate annualized return for each asset.
    """
    return daily_returns.mean() * TRADING_DAYS


def annualized_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate annualized volatility for each asset.
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(
    daily_returns: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Calculate Sharpe Ratio (risk-free rate default = 0).
    """
    ann_return = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)

    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(cumulative_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate maximum drawdown for each asset.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1

    return drawdown.min()


def summary_statistics(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a full summary table with
    return, volatility, Sharpe ratio, and max drawdown.
    """
    daily_ret = compute_daily_returns(price_df)
    cum_ret = compute_cumulative_returns(daily_ret)

    summary = pd.DataFrame({
        "Annual Return (%)": annualized_return(daily_ret) * 100,
        "Annual Volatility (%)": annualized_volatility(daily_ret) * 100,
        "Sharpe Ratio": sharpe_ratio(daily_ret),
        "Max Drawdown (%)": max_drawdown(cum_ret) * 100
    })

    return summary.round(2)
