import pandas as pd
import csv

nasdaq_list = pd.read_csv("/Users/reetikasingh/Desktop/trading/venv/nasdaq.csv")
symbol_list = nasdaq_list['Symbol'].tolist()