# ClaimLens demo: API + UI in one container, built and trained on LinuxONE (s390x).
# PyPI has no s390x wheels for xgboost / scikit-learn, so those come from Ubuntu's
# own s390x packages (same approach as the team's Jupyter image). Pure-Python
# packages come from pip.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-numpy python3-pandas python3-sklearn python3-xgboost \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "pypdf>=4.0" "python-multipart>=0.0.9"

WORKDIR /app
COPY src/ src/
COPY ui/ ui/
COPY samples/*.pdf samples/
# Use the committed models instead of retraining on every build. The final model
# (src/explain.py) is located and checksummed via reports/explain_runtime/manifest.json.
COPY models/ models/
COPY reports/explain_runtime/ reports/explain_runtime/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python3", "-m", "uvicorn", "demo_server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
