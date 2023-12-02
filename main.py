from fastapi import FastAPI, Depends, Form, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse

from app.data_management import add_namelist, print_namelist, display_namelist
from app.data_management import json2db, db2json, show_table_info
import app.utils.findspeaker

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}
