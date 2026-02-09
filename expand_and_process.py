import pandas as pd
from collections import deque
from utils.etherscan_api import get_transactions
import time
from tqdm import tqdm


phishing = pd.read_csv("data/raw/phishingaddress.csv", header=None, names=["address"])
start_nodes = phishing["address"].str.lower().unique()[1] 
MAX_DEPTH = 2
visited = set()
queue = deque([(addr, 0) for addr in start_nodes])
transactions = []


pbar = tqdm(total=len(start_nodes), desc="Processing addresses", unit="addr")

while queue:
    addr, depth = queue.popleft()

    if addr in visited or depth > MAX_DEPTH:
        continue

    visited.add(addr)
    

    pbar.set_description(f"Depth {depth} | Visited: {len(visited)} | Queue: {len(queue)} | Txs: {len(transactions)}")
    pbar.update(1)
    
    print(f"Processing {addr} at depth {depth}")
    
    try:
        txs = get_transactions(addr)
        

        if not isinstance(txs, list):
            print(f"  Skipping {addr}: API returned non-list response")
            continue
        
        for tx in txs:
     
            if not isinstance(tx, dict) or "from" not in tx or "to" not in tx:
                continue
                
            frm = tx["from"].lower()
            to = tx["to"].lower() if tx["to"] else "" 
            transactions.append(tx)

            if depth < MAX_DEPTH:
                queue.append((frm, depth + 1))
                if to: 
                    queue.append((to, depth + 1))
        
        time.sleep(0.2)  
        
    except Exception as e:
        print(f"  Error processing {addr}: {e}")
        continue

pbar.close()

if transactions:
    pd.DataFrame(transactions).to_csv(
        "data/raw/transactions_expanded.csv", index=False
    )
    print(f"Graph expansion completed - {len(transactions)} transactions, {len(visited)} addresses")
else:
    print("No transactions collected")