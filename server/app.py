from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from server.db import pool
from server.routes.evals import router as evals_router
from server.routes.judges import router as judges_router
from server.routes.traces import router as traces_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="warden", lifespan=lifespan)
app.include_router(traces_router)
app.include_router(evals_router)
app.include_router(judges_router)


@app.get("/health")
def health():
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")
