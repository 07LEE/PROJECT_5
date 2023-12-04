from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()
