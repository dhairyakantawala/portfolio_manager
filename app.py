import streamlit as st
import pandas as pd
from main_functions import api_calling
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ticker_names = pd.read_csv('/Users/dhairya/cs projects/ideas 2025/main_functions/temp_data.csv')
list_tickers = ticker_names.columns.drop('date')



options = st.multiselect(
    "What stocks you want in your portfolio?",
    list_tickers,
)

next_button = False

if options:
    st.write("Selected stocks:", options)
    next_button = st.button(
        "Next",
        type="primary",
        use_container_width=True,
        key="next_button"
    )
    if next_button:
        st.session_state.stocks = options

if next_button:
    stock_data = {}
    for ticker in options:
        stock_data[ticker] = ticker_names[ticker]

    returns = pd.DataFrame({ticker: data.pct_change() for ticker, data in stock_data.items()})
    returns = returns.dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    risk_free_rate = 0.06299
    num_portfolios = 10000
    results = []

    for i in range(num_portfolios):
        weights = np.random.random(len(options))
        weights = weights/np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 12
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 12, weights)))
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        results.append({
            'Return': portfolio_return,
            'Risk': portfolio_std, 
            'Sharpe': sharpe_ratio,
            'Weights': weights
        })

    results_df = pd.DataFrame(results)
    max_sharpe_port = results_df.iloc[results_df['Sharpe'].idxmax()]
    min_var_port = results_df.iloc[results_df['Risk'].idxmin()]

    st.session_state.optimal_weights = {
        ticker: weight for ticker, weight in zip(options, max_sharpe_port['Weights'])
    }

    fig_frontier = plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Risk'], results_df['Return'], c=results_df['Sharpe'], 
                cmap='viridis', alpha=0.5)
    
    x_cml = np.linspace(0, max(results_df['Risk']), 100)
    y_cml = risk_free_rate + (max_sharpe_port['Return'] - risk_free_rate) * x_cml / max_sharpe_port['Risk']
    
    plt.plot(x_cml, y_cml, 'r--', label='Capital Market Line')
    plt.scatter(max_sharpe_port['Risk'], max_sharpe_port['Return'], color='red', 
                marker='*', s=200, label='Optimal Portfolio')
    plt.scatter(min_var_port['Risk'], min_var_port['Return'], color='green', 
                marker='*', s=200, label='Minimum Variance')
    plt.plot(0, risk_free_rate, 'k*', markersize=15, label='Risk-Free Rate')
    
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier and Capital Market Line')
    plt.legend()
    
    st.pyplot(fig_frontier)

    st.subheader("Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{max_sharpe_port['Return']:.2%}")
    col2.metric("Risk", f"{max_sharpe_port['Risk']:.2%}")
    col3.metric("Sharpe Ratio", f"{max_sharpe_port['Sharpe']:.2f}")

    fig_weights = plt.figure(figsize=(10, 6))
    plt.pie(max_sharpe_port['Weights'], labels=options, autopct='%1.1f%%')
    plt.title('Optimal Portfolio Weights')
    st.pyplot(fig_weights)

    st.subheader("Optimal Portfolio Weights")
    weights_df = pd.DataFrame({
        'Stock': options,
        'Weight': [f"{w:.2%}" for w in max_sharpe_port['Weights']]
    })
    st.table(weights_df)
if st.button("Show Historical Performance with Optimal Weights"):
    monthly_data = {}
    for ticker in options:
        try:
            monthly_data[ticker] = ticker_names[ticker]
        except ValueError as e:
            st.error(f"Error getting data for {ticker}: {str(e)}")
            st.stop()

    nifty_data = pd.read_csv('/Users/dhairya/cs projects/ideas 2025/main_functions/Nifty 50 Historical Data.csv')
    nifty_data['Date'] = pd.to_datetime(nifty_data['Date'], format='%m/%d/%Y')
    nifty_data = nifty_data.sort_values('Date')
    nifty_data['Price'] = nifty_data['Price'].str.replace(',', '').astype(float)
    nifty_returns = nifty_data['Price'].pct_change()
    nifty_returns = nifty_returns.iloc[-120:]

    portfolio_returns = pd.DataFrame()
    for ticker, weight in st.session_state.optimal_weights.items():
        monthly_returns = monthly_data[ticker].pct_change()
        portfolio_returns[ticker] = monthly_returns * weight

    portfolio_returns['Total'] = portfolio_returns.sum(axis=1)
    portfolio_returns = portfolio_returns.iloc[-120:]
    
    cumulative_returns = (1 + portfolio_returns['Total']).cumprod()
    
    risk_free_rate = 0.03  # Assuming 3% risk-free rate
    excess_portfolio_returns = portfolio_returns['Total'] - risk_free_rate/12
    excess_market_returns = nifty_returns - risk_free_rate/12
    
    beta = np.cov(excess_portfolio_returns, excess_market_returns)[0,1] / np.var(excess_market_returns)
    expected_returns_capm = risk_free_rate/12 + beta * excess_market_returns
    cumulative_returns_capm = (1 + expected_returns_capm).cumprod()
    
    fig_historical = plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns.values, label='Actual Returns')
    plt.plot(cumulative_returns.index, cumulative_returns_capm.values, label='CAPM Expected Returns')
    plt.title('10-Year Historical Performance: Actual vs CAPM Model')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(fig_historical)

    final_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + final_return) ** (1/10) - 1
    
    st.metric("10-Year Cumulative Return", f"{final_return:.2%}")
    st.metric("Annualized Return", f"{annual_return:.2%}")
    st.metric("Portfolio Beta", f"{beta:.2f}")


# to do's tommorow:
# risk tolorance defining - eg, willing to take risk of 12% +- 2%
# ideal portfolio in this condition, markovix frontier, captial allocation line show
# tagent line go up or down by considering your risk tolorance, get waitages of given asset
# only CAPM model running, know the betas, (calculate betas), realised vs normal returns, monthly data can be used, last 8 year data
# expected return vs actual return
# plot 1 - realised return, plot 2 - CAPM model, plot 3 - Arbitrag pricing model (matrix show and returns)
# every year efficency calculate, take best efficency, and then predict how will your model look - something similar to broinger band