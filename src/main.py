from fastapi import FastAPI
from src.db.route import router

app=FastAPI()

app.include_router(router, prefix=f"/prediction")