FROM public.ecr.aws/docker/library/python:3.12-slim

# Set up a new user 'user' with a home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install uv locally
RUN pip install --no-cache-dir --user uv

# Copy files
COPY --chown=user . $HOME/app

# Add the --system flag. 
# Because we are 'user', uv will now safely install to /home/user/.local/lib/
RUN uv pip install --system --no-cache -e .

# Port for Hugging Face
EXPOSE 7860

# Run uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]