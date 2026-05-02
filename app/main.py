from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api import health, qa, document, websocket

app = FastAPI(title="Guideon Voice QA")

app.include_router(health.router)
app.include_router(qa.router)
app.include_router(document.router)
app.include_router(websocket.router)
