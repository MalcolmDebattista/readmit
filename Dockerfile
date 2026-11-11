FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir --no-deps .

# The raw data (from `readmit-fetch`) is copied in and the model trained at build time, as in Session 3.
# The build fails if the package does not reproduce the notebook's numbers.
COPY data/raw ./data/raw
RUN readmit-train && rm -rf data

# The version comes from pyproject.toml through `make build`; the API reports it and logs it with every score.
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

# Run as an ordinary user, not root
RUN useradd --system --uid 10001 readmit
USER readmit

# --limit-concurrency: beyond 6 requests in progress, answer 503 at once instead of queueing work nobody waits for
# (without it, 2 requests/s built a queue the server was still working through minutes later)
EXPOSE 8888
CMD ["uvicorn", "readmit.api:app", "--host", "0.0.0.0", "--port", "8888", "--limit-concurrency", "6"]
