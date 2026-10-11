FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Uses models/bank_model.joblib if present (the Kaggle-trained one after `python -m scripts.sync_kaggle`);
# otherwise trains the synthetic demo model.
RUN [ -f models/bank_model.joblib ] || python -m fraud.model
ENV HTTPS_ONLY=1
EXPOSE 8000
# --proxy-headers makes the rate limiter see the real client IP behind Render's proxy
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
