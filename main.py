"""

"""
from datetime import datetime
import os

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .module.input_text import user_input

app = FastAPI()

# 정적 파일(HTML, CSS 등)을 제공하기 위한 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# HTML 렌더링을 위한 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# /put 엔드포인트 추가
@app.get("/put.html", response_class=HTMLResponse)
async def read_put(request: Request):
    return templates.TemplateResponse("put.html", {"request": request})

# 
@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    # 파일을 저장할 폴더 경로
    upload_folder = "uploads/"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # 현재 날짜와 시간을 이용하여 새로운 파일 이름 생성
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    _, file_extension = os.path.splitext(file.filename)
    new_filename = f"{current_datetime}_{file_extension}"

    # 파일을 저장할 경로
    file_path = os.path.join(upload_folder, new_filename)

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # 파일 내용 읽기
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # 가공된 결과 얻기
    output = user_input(file_content)

    # 파일 경로를 반환
    return templates.TemplateResponse("result.html",
                                      {"request": request, "filename": new_filename,
                                       "file_content": file_content,
                                       "output":output})
