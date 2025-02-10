from fastapi import APIRouter
from src.utils.get_sentiment import analyze_sentiment
from ..models.sentiment import Sentiment


router = APIRouter()


@router.post("/analyze")
async def analyze(request: Sentiment):
    return analyze_sentiment(request.text)
