FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY app.py .

EXPOSE 3000

CMD ["uv", "run", "app.py"]
