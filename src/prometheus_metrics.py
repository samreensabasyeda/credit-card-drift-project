from prometheus_client import Gauge, start_http_server
import time
import pandas as pd
from psi import calculate_psi

psi_metric = Gauge("creditcard_data_drift_psi", "PSI for credit card drift", ["feature", "dataset"])

start_http_server(8000)

baseline = pd.read_csv("data/baseline.csv")

while True:
    for i in range(1, 6):
        inf = pd.read_csv(f"data/inference_{i}.csv")

        psi_amt = calculate_psi(baseline["amount"], inf["amount"])
        psi_metric.labels("amount", f"inference_{i}").set(psi_amt)

    time.sleep(30)
