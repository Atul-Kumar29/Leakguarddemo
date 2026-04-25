import os
import sys

# Detect environment
IS_COLAB = "google.colab" in sys.modules

def setup_env():
    if IS_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Running on Colab: Drive Mounted.")
    else:
        token = os.getenv("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token)
            print("Running on Hugging Face: Auth Completed.")
        else:
            print("Warning: HF_TOKEN not found. Push to hub will fail.")

    os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    setup_env()
    print("Starting LeakGuard RL Training...")
    import train