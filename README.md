  This is an experimental trading bot.

######## How to set up the project #########


1. Set Up the Project:

mkdir woola-bot
cd woola-bot
python3 -m venv woola_bot_env
source woola_bot_env/bin/activate

2. Create requirements.txt:

echo -e "pandas\nnumpy\nrequests\nccxt" > requirements.txt

3. Install Dependencies:

pip install -r requirements.txt

4. Verify:

pip list

5. Deactivate Environment:

deactivate


############# Project architecture ##################

woola-bot/
├── woola_bot_env/          # Virtual environment directory
├── data/                      # Directory for storing data files
│   └── market_data.csv
├── logs/                      # Directory for log files
│   └── trading_bot.log
├── strategies/                # Directory for strategy modules
│   └── moving_average.py
├── main.py                    # Main script to run the bot
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── requirements.txt           # Dependencies file
└── README.md                  # Project documentation


Directory listing for /kline_for_metatrader4/ with ByBit candle data:
https://public.bybit.com/kline_for_metatrader4/
Data has the following OHLCV format :

column1                            column2           column3             column4           column5         column6
Time yyyy.MM.dd（UTC+3）            Open price       Highest price       Lowest price      Close price     Trade volume
                                   