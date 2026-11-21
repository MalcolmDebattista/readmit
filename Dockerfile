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

# Session 06: the image holds code only. The model comes from the registry when the API starts
# (MODEL_URI=models:/readmit@champion in compose.yaml), so a new model needs no new image and no patient data is inside.

# The code's version comes from pyproject.toml through `make build`; /ready reports it.
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

# Run as an ordinary user, not root
RUN useradd --system --uid 10001 readmit
USER readmit

# --limit-concurrency: beyond 32 requests in progress (a whole day of discharges at once still fits), answer 503 at
# once instead of queueing work nobody waits for; at ~0.4 s each the backlog is capped at ~13 s
# (without it, 2 requests/s built a queue the server was still working through minutes later)
EXPOSE 8888
CMD ["uvicorn", "readmit.api:app", "--host", "0.0.0.0", "--port", "8888", "--limit-concurrency", "32"]
