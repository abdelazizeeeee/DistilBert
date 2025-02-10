from .settings import settings
from ..models.user import User
from .settings import settings

from motor.motor_asyncio import AsyncIOMotorClient

from beanie import init_beanie


async def startDB():
    client = AsyncIOMotorClient(settings.DATABASE_URL)

    await init_beanie(database=client.dbname, document_models=[User])


#     await init_beanie(database=client.db_name, document_models=[User])
