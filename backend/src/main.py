from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.database import startDB
from src.config.settings import settings
from src.routes import sentiment

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.CLIENT_ORIGIN] if settings.CLIENT_ORIGIN else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize dependencies on startup."""
    await startDB()

app.include_router(sentiment.router, tags=[
                   "Sentiment"], prefix="/api/sentiment")
