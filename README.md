# LongShort-Cointegration-Pairwise-Strategy

# Guide 
	Run the file DataCreator.py in Database, to create the Database (Adjust if necessary).
	Adjust the directory to your own database directory in the database_directory.py. 
	Run price_data.py to extract the prices
	Run label_assets.py to calculate the raw inputs for the strategy
	Run TRADING_STRATEGY_v2.py to run the strategy. 
 (Note: When running label_assets.py, the code will take some time, if you have a lot of assets. This is because you are performing rolling cointegration checks to verify the two series are cointegrated)
Ensure you have installed the necessary dependencies. 
You must populate the AssetIndex in the database, so it can import the price data.
You must also populate the BenchmarkIndex accordingly to the corresponding benchmark they have, if you wish to perform the residual momentum analysis (as explained better below). All new Stocks should have the corresponding market benchmark and the respective industry benchmark. 


# Motivation
For the last decade, financial markets traded within a monetary expansionary regime, up to the point where you actualy ended up paying for depositing/loaning money. This translated into larger volumes of money within the economy, a greater facility to credit and inevitably, higher stock prices. Even bad business models could finance their expansion at the cost of cash (as was the case with oil companies after the bust of the shale revolution and the recovery of the negative performance in begining of 2021 - the so called "cash burners"). Short trading in such market is extremely risky, making this strategies unappealing to hold (in fact, not only shorting, but active investing aswell since the benefit of selecting winners is also jeoperdize by the fact that we are all winners).
However, in 2022, a shift in the regime occured. More precisely, the shift in to a tigher monetary policy, characterized by a higher interest rates. This puts pressure in the economy and consequently on the companies. In such a regime, markets are more fragmented, in the sense that bad companies no longer have access to the same ammount of liquidity and is up to the winners to keep the economy functioning. In such environment, manifestations of winners and losers are more pronounced, solidifying the argument of active managment. That is, the stock picking becomes more important, since systematic risk is already discounted as high. Good managers, are the ones able to identify good companies. While bad managers, end up with the losers. And when losers fall, they fall hard, since there is no longer a "life support"(liquidity). This promotes the perfect hunting ground for short investors, which pop out in this environment. Adding a short approach to the investment case becomes more interesting, in such environment.


# The Strategy 
The strategy seeks to profit from the mean conversion of two cointegrated stocks. Cointegration is chosen because it reflects the long-term co-movement among two series, as opposed to the correlation, which is more short term oriented. The rationale (assumption) is that if two stocks are cointegrated over time, it suggests that their fundamental characteristics are similar and such similarities are embedded through the price action of the market participants. To further enhance the rationale, pairs are considered only within their industry/sector group. 
Thus, if they tend to co-move over time, any short-term deviations from their common central tendency (explained below) can be exploited. Of course, even if two stocks share similarities and co-move over time, discrepancies may arise from idiosyncrasies in their businesses. That is, common specific factors that marginally affects, either positively or negatively, on company relative to its peer.
To partially cope with this issue, the strategy should be employed on a vast diversified investment universe. Furthermore, the strategy alleviates part of this unsystematic risk, by segregating winners from losers, based on their price momentum, relatively to other industry/sector peers. The rationale (again assumption) is the same as above, since market participants are expected to embed any positive(negative) news flows into stock prices, the labelling allows us the strategy to safeguard itself in catching falling knifes (buying losers) or from selling stars (selling winners).

## Note on Momentum 
The common standard to define momentum is based on the relative returns among several stocks in a peer group, and a classification of winners and losers, given a certain threshold. Stocks outperforming are winners and stocks underperforming are losers. However, performance is a byproduct of two types of risks – systematic and unsystematic risk – and just because a given stock is performing better than the other, it may be because it’s a winner, but rather because it’s riding the on the right sector. Thus, ranking based on industry/sector may alleviate this issue. But again, this may not suffice, and an extension of this strategy could account the consideration of additional risk premiums (besides industry or sector) to better extract. By extracting the alfa of the stock, momentum can be enhanced, by better reflecting the outperformance of a stock – this is commonly referred to as Residual Momentum.

The code automatically runs the residual premium factors derived from the regression between the stock returns and theindustry and size betas. To activate it, just run the commented line in the below figure instead of the above. YOu can active it by switchin the functions on the TRADING_STRATEGY_v2.py in here:
![image](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/assets/84282116/85e75f36-19b4-4c64-8f32-d30526059293)


## Note on factor premiums
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
[RET_PERFOMANCES_ACC.docx](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/files/11961056/RET_PERFOMANCES_ACC.docx)



The rolling betas of the strategies by industry and ALL industries. 
[BETA_PERFORMANCES.docx](https://github.com/LusoNX/LongShort-Cointegration-Pairwise-Strategy/files/11961066/BETA_PERFORMANCES.docx)


Again, the backtest results are not fully curated. However, the results do paint the generally idea behind the motivation of the project. The strategy captures majority of its performance in between 2020-2022, suggesting that indeed, markets are more fragmented and we are able to capture short term deviations by long and short positioning, more frequently. 






