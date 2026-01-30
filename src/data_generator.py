import numpy as np
import pandas as pd

def generate_data(n, drift=False):
    np.random.seed(42)
    
    amount = np.random.normal(200, 50, n)
    txn_count = np.random.poisson(3, n)
    age = np.random.randint(21, 70, n)

    if drift:
        amount = np.random.normal(350, 80, n)  # drifted
        txn_count = np.random.poisson(6, n)

    fraud = (amount > 300).astype(int)

    return pd.DataFrame({
        "amount": amount,
        "txn_count": txn_count,
        "age": age,
        "fraud": fraud
    })

# baseline
baseline = generate_data(100)
baseline.to_csv("data/baseline.csv", index=False)

# inference datasets
for i in range(1, 6):
    inf = generate_data(100, drift=True)
    inf.to_csv(f"data/inference_{i}.csv", index=False)
