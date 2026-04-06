FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv pip install --system -e .

EXPOSE 8000

# Run uvicorn pointing to server.app:app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
