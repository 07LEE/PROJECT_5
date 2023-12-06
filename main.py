from fastapi import FastAPI
import app.utils.findspeaker

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
