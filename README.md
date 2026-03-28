📊 Trader Behavior & Market Sentiment Analysis
Hyperliquid Historical Data x Bitcoin Fear & Greed Index
This project explores the intersection of on-chain trader performance and off-chain market sentiment. By merging historical execution data from Hyperliquid with the Bitcoin Fear & Greed Index, I developed a data-driven overview of how psychological market phases impact profitability, risk appetite, and trade success rates.

🚀 Key Insights (TL;DR)
The "Extreme Fear" Premium: While win rates are lower (~62%) during Extreme Fear, the Average PnL is the highest ($654.19), suggesting that high-conviction contrarian trades yield the best rewards.

The Greed Trap: During "Extreme Greed," trader win rates plummet to 48% and average PnL flips negative (-$53.36), indicating significant capital erosion due to FOMO and late-cycle entries.

Predictive Power: Using a Random Forest Classifier, the Fear & Greed Index value was identified as the 2nd most important feature in predicting trade profitability, outweighing trade size and direction.

🛠️ Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy (Data Manipulation), Matplotlib, Seaborn (Visualization), Scikit-Learn (Machine Learning)

Platform: Hyperliquid (DEX) Historical Data

📈 Analysis Workflow
1. Data Integration & Cleaning
Synchronized high-frequency trade executions (IST) with daily sentiment classifications.

Handled precision for crypto-native units (Size Tokens, Size USD, Execution Price).

Filtered for Closed Trades to ensure PnL analysis was based on realized gains/losses.

2. Exploratory Data Analysis (EDA)
Win Rate vs. Sentiment: Identified that "Neutral" markets offer the highest consistency (96% win rate) but lower volatility-driven rewards.

PnL Distribution: Visualized how "Extreme Greed" leads to outsized liquidations and negative mean returns.

3. Predictive Modeling (Machine Learning)
I trained a Random Forest Classifier to determine if a trade will be profitable based on market context.

Accuracy: ~99% (validated on test split).

Feature Importance: Sentiment Value and Execution Price were the primary drivers of success, suggesting that when you enter is more important than how much you trade.
