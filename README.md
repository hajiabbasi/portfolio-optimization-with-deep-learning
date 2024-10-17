# portfolio-optimization-with-deep-learning
In this project, we implemented simple financial concepts such as risk and return, progressing to more complex financial ideas. Based on this foundation, a module named "mahdi" (after myself) was created to facilitate the implementation of these concepts in future projects. Additionally, portfolio optimization was performed in the Iranian stock market. 

Given the annual depreciation of the national currency in Iran, the US dollar was chosen as an investment option. In the next project, titled "Tehran Stock Exchange with USD," portfolio optimization was conducted with the dollar considered as an investment factor. This file includes explanations about Conditional Value at Risk (CVaR), the Sharpe Ratio, and Modern Portfolio Theory.
In this section, we delve into the implementation of the Markowitz model for portfolio optimization. The Markowitz model, introduced by Harry Markowitz in 1952, is a foundational concept in Modern Portfolio Theory (MPT). It focuses on maximizing returns while minimizing risk through optimal asset allocation. The model is based on the idea of efficient portfolios, where the expected return is maximized for a given level of risk, or conversely, where risk is minimized for a given level of expected return.

The optimization problem can be mathematically represented as:

\[
\text{Minimize} \quad \sigma_p^2 = \mathbf{w}^T \mathbf{C} \mathbf{w}
\]

subject to 

\[
\sum_{i=1}^n w_i = 1
\]

where:
- \( \sigma_p^2 \) is the variance of the portfolio's returns,
- \( \mathbf{w} \) is the vector of asset weights,
- \( \mathbf{C} \) is the covariance matrix of asset returns.

Following this theoretical foundation, we utilized very simple machine learning models to predict prices and subsequently constructed a portfolio based on these predictions.
