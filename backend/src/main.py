from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import settings
from .config.database import startDB
from fastapi.staticfiles import StaticFiles
# from src.routes import auth, user
# from .routes import auth, scan
from .routes import scan

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
origins = [
    settings.CLIENT_ORIGIN,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def start_dependencies():
    await startDB()
    # await startMinio()


# app.include_router(auth.router, tags=["Auth"], prefix="/api/auth")
# app.include_router(user.router, tags=["User"], prefix="/api/users")
app.include_router(scan.router, tags=["DocumentScan"], prefix="/api/scan")
