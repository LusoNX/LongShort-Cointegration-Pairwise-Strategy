# LongShort-Cointegration-Pairwise-Strategy

# Guide 
	Run the file DataCreator.py in Database, to create the Database (Adjust if necessary).
	Adjust the directory to your own database directory in the database_directory.py. 
	Run price_data.py to extract the prices
	Run label_assets.py to calculate the raw inputs for the strategy
	Run TRADING_STRATEGY_v2.py to run the strategy. 
Ensure you have installed the necessary dependencies. 
You must populate the AssetIndex in the database, so it can import the price data.

# The Strategy 
The strategy seeks to profit from the mean conversion of two cointegrated stocks. Cointegration is chosen because it reflects the long-term co-movement among two series, as opposed to the correlation, which is more short term oriented. The rationale (assumption) is that if two stocks are cointegrated over time, it suggests that their fundamental characteristics are similar and such similarities are embedded through the price action of the market participants. To further enhance the rationale, pairs are considered only within their industry/sector group. 
Thus, if they tend to co-move over time, any short-term deviations from their common central tendency (explained below) can be exploited. Of course, even if two stocks share similarities and co-move over time, discrepancies may arise from idiosyncrasies in their businesses. That is, common specific factors that marginally affects, either positively or negatively, on company relative to its peer.
To partially cope with this issue, the strategy should be employed on a vast diversified investment universe. Furthermore, the strategy alleviates part of this unsystematic risk, by segregating winners from losers, based on their price momentum, relatively to other industry/sector peers. The rationale (again assumption) is the same as above, since market participants are expected to embed any positive(negative) news flows into stock prices, the labelling allows us the strategy to safeguard itself in catching falling knifes (buying losers) or from selling stars (selling winners).

#Note on Momentum 
The common standard to define momentum is based on the relative returns among several stocks in a peer group, and a classification of winners and losers, given a certain threshold. Stocks outperforming are winners and stocks underperforming are losers. However, performance is a byproduct of two types of risks – systematic and unsystematic risk – and just because a given stock is performing better than the other, it may be because it’s a winner, but rather because it’s riding the on the right sector. Thus, ranking based on industry/sector may alleviate this issue. But again, this may not suffice, and an extension of this strategy could account the consideration of additional risk premiums (besides industry or sector) to better extract. By extracting the alfa of the stock, momentum can be enhanced, by better reflecting the outperformance of a stock – this is commonly referred to as Residual Momentum.

#Note on factor premiums
Besides Momentum, there are other premium factors that can be employed to further enhance the trading strategy or filter out the investment universe. Premiums, in a nutshell are the exposure to a systematic factor and is up to the analyst to define the criteria used to characterize the factor. For example, the growth premium, is usually derived from the ranking of stock, based on their P/E ratio, or other combination of growth indicators. 
In this sense, an extension of the model could consider the implementation of these additional premiums. One example, would be the labelling of stocks based on quality metrics, such as, growth of earnings/cash flow (CF), quality of earnings/CF, earnings/CF persistence and a ruling on the strategy that would limit long positions only to quality stocks, further minimizing the idiosyncratic risk. 


# Back to the Strategy…
The strategy starts by: 
	Calculate the residuals from the cointegration regression among two pairs. 
Log(Stock_A )=β*Log(Stock_B )+ ∈
	Check for stationary of the residuals. 
	Perform an ADF test (Augmented Dickey Fuller Test).
	If stationary, then residuals are expected to mean revert, and the two stocks are cointegrated. 
	Cointegration checks should be regularly made, since cointegration may disappear (even if it’s a long-term tendency).
	Standardize the residuals from the cointegrated pairs. 
	Define the signal thresholds for standardized residuals.

Positions are opened every time the zpread crosses the “Open Signal” (red line) threshold and closed subsequently after crossing the “Close Signal” threshold (blue line), as shown below.

![image](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/assets/84282116/9d89dbdf-455b-4cf9-b669-125b7a539f85)


The rationale for using the spread is that, if two series are cointegrated, then we expect that any short term deviations are short lived and eventually converge back to itsmean relationship. By standardizing the results, the mean is 0 and our deviations are X standard deviations.




# Weigh Scheme 
For back testing purposes, the strategy seeks a beta neutrality, with a weighting scheme accounted for that. Because the strategy is employed on a diversified universe, it uses the SP500 the benchmark for the calculation of the hedge ratio.  The hedge ratio is calculated has follow.
Wx= Weigth_X=β_Y/(β_X  +β_Y )

Wy=Weigth_Y=1-Weigth_X

Furthermore, the strategy is assumed to be always fully invested, meaning that, the weights for each stock are given by the Wx and Wy , divided by the total number of trades occurred at each point in time. 

## Backtest Limitation
The backtest results are merely demonstrative and are not curated for a better fit. Some of the limitations include. 
	Selection bias – Only 40-50 stock were selected, which represent a very small segment of the market, not truly reflecting the strategy generalizing capacity. Additionally, the timeframes themselves suffer from this.
	No trading costs and transaction fees assumed. 
	No rebalancing considerations for the weights 
	The dataset considers close prices, which may not be “accessible prices”, since close price is registered at the end of the session, but the trading occurs during the trading session. 
## Strategy Results
(Note: All the inputs are derived on a rolling window basis, including standardization, to avoid look-ahead biases). 
The accumulated performance of the strategies by industry and ALL industries.
![image](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/assets/84282116/13e4ef2f-a269-4c8a-8a52-f6abb2dcadfe)


The rolling betas of the strategies by industry and ALL industries. 
![image](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/assets/84282116/a68feb3e-37f0-471e-af91-ff45b5c472ec)






