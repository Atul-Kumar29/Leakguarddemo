FROM public.ecr.aws/docker/library/python:3.12-slim

# Hugging Face runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv in the user's local path
RUN pip install --no-cache-dir uv

# Copy files and ensure the 'user' owns them
COPY --chown=user . $HOME/app

# Install dependencies using uv
RUN uv pip install --system --no-cache -e .

# Hugging Face MUST use port 7860
EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]