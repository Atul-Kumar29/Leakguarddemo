FROM public.ecr.aws/docker/library/python:3.12-slim

# Set up a new user 'user' with a home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv in the user's local directory
RUN pip install --no-cache-dir --user uv

# Copy the rest of your files and ensure 'user' owns them
COPY --chown=user . $HOME/app

# Install dependencies into the user's local site-packages (NO --system flag)
# We use --no-cache to keep the image small
RUN uv pip install --no-cache -e .

# Hugging Face Spaces listen on port 7860
EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]