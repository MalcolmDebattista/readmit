from prometheus_client import Counter, Histogram

# Same names as the Session 4 shop / Gozo services, so the provided Grafana dashboard works unchanged.
PREDICTIONS = Counter("predictions", "Number of discharges scored")
LATENCY = Histogram("prediction_latency_seconds", "Time taken to answer a /risk request",
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
INPUTS = Histogram("prediction_input", "Input feature: inpatient stays in the prior year (the top feature by gain)",
                   buckets=[0, 1, 2, 3, 5, 8, 13, 21])
SCORES = Histogram("prediction_score", "Risk scores returned", buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
NOT_ELIGIBLE = Counter("not_eligible", "Discharges to death or hospice: never scored (Trap 4)")
