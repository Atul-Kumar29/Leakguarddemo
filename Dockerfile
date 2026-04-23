FROM public.ecr.aws/docker/library/python:3.12-slim

# Set up a new user 'user' with a home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy files first and ensure 'user' owns them
COPY --chown=user . $HOME/app

# Install dependencies using standard pip with the --user flag
# This forces installation into /home/user/.local/ where permissions are guaranteed
RUN pip install --no-cache-dir --user -e .

# Hugging Face Spaces listen on port 7860
EXPOSE 7860

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]