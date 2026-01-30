import pandas as pd
from psi import calculate_psi

baseline = pd.read_csv("data/baseline.csv")

for i in range(1, 6):
    inf = pd.read_csv(f"data/inference_{i}.csv")

    psi_amount = calculate_psi(baseline["amount"], inf["amount"])
    psi_txn = calculate_psi(baseline["txn_count"], inf["txn_count"])

    print(f"Inference {i}:")
    print(f"  PSI Amount    : {psi_amount:.3f}")
    print(f"  PSI Txn Count : {psi_txn:.3f}")
