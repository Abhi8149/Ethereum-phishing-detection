import pandas as pd
from tqdm import tqdm
from utils.etherscan_api import get_transactions

phishing_df = pd.read_csv("data/raw/phishingaddress.csv")
addresses = phishing_df

all_transactions = []

for addr in tqdm(addresses):
    txs = get_transactions(addr)
    all_transactions.extend(txs)

df = pd.DataFrame(all_transactions)
df.to_csv("data/raw/transactions_raw.csv", index=False)

print("Raw transactions downloaded")
