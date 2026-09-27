FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 LEADHARBOUR_MODEL_PATH=/app/artifacts/model.joblib
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY artifacts/model.joblib /app/artifacts/model.joblib
RUN useradd --system --uid 10001 --home-dir /nonexistent leadharbour && \
    chown -R leadharbour:leadharbour /app
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn", "leadharbour.api:app_from_environment", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
