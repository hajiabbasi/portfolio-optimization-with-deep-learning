# portfolio-optimization-with-deep-learning
This section focuses on implementing the Markowitz model for portfolio optimization. The Markowitz model, also known as Modern Portfolio Theory (MPT), was introduced by Harry Markowitz in 1952. It aims to maximize returns for a given level of risk by carefully selecting the proportions of various assets in a portfolio. The key formula used in Markowitz's optimization is:

\[
\text{Minimize} \quad \sigma_p^2 = \mathbf{w}^T \mathbf{C} \mathbf{w}
\]

subject to 

\[
\sum_{i=1}^n w_i = 1
\]

where \( \sigma_p^2 \) is the variance of the portfolio's returns, \( \mathbf{w} \) is the vector of asset weights, and \( \mathbf{C} \) is the covariance matrix of asset returns.

Following this, we used very simple machine learning models to predict prices and subsequently formed the portfolio.
