from fastapi import FastAPI
from server.environment import LeakGuardEnvironment
from server.models import LeakGuardAction

app = FastAPI(title="LeakGuard AI Environment")
env = LeakGuardEnvironment()

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step_env(action: LeakGuardAction):
    obs, reward, done, info = env.step(action.model_dump())
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state.model_dump()
