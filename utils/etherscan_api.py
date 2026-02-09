import requests

API_KEY = "BRI6G5PQGWG5VX1R8CIKP1MW1HJIXZE8GC"

def get_transactions(address):
    url = "https://api.etherscan.io/v2/api"
    
    params = {
        "chainid": 1,                 # Ethereum mainnet
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": API_KEY
    }

    response = requests.get(url, params=params).json()

    # safety check
    if response.get("status") != "1":
        print(f"{address} is not having any transection record")
        return []

    return response["result"]
