# server/app.py
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from server.environment import LeakGuardEnvironment
from server.models import LeakGuardAction

app = FastAPI(title="LeakGuard AI Environment")
env = LeakGuardEnvironment()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step_env(action: LeakGuardAction):
    obs, reward, done, info = env.step(action.model_dump(exclude_none=True))
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state.model_dump()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7864)

if __name__ == "__main__":
    main()