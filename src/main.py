from fastapi import FastAPI
from src.db.route import router
import uvicorn
app=FastAPI(host="0.0.0.0")

app.include_router(router, prefix=f"/prediction")

if __name__=="__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=5000)