"""
This is main.py
"""
from datetime import datetime
import os

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from module.load_model import load_fs, load_ner
from module.input_process import make_ner_input
from module.ner_utils import make_name_list, show_name_list


# 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class AppData:
    def __init__(self):
        self.file_content = ""
        self.name_list = []

app_data = AppData()

name_list = []
file_content =''

# 모델 불러오기
fs_model = load_fs()
ner_model, checkpoint = load_ner()

# 파일을 저장할 폴더 경로
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 현재 날짜와 시간을 이용하여 새로운 파일 이름 생성
current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
new_filename = f"{current_datetime}.txt"
text_file_path = os.path.join(UPLOAD_FOLDER, new_filename)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """INDEX.HTML 화면"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/put.html", response_class=HTMLResponse)
async def read_put(request: Request):
    """PUT.HTML 화면"""
    return templates.TemplateResponse("put.html", {"request": request})


@app.post("/upload-and-redirect", response_class=HTMLResponse)
async def upload_and_show(request: Request, file: UploadFile = File(...)):
    """파일을 업로드하고 confirm.html로 결과를 보여줍니다."""
    # 파일 저장
    with open(text_file_path, "wb") as f:
        f.write(file.file.read())

    # 파일 내용을 가져와서 file_content 변수에 할당
    with open(text_file_path, "r", encoding="utf-8") as f:
        app_data.file_content = f.read()

    content = make_ner_input(app_data.file_content)
    app_data.name_list = make_name_list(content, checkpoint)
    show = show_name_list(app_data.name_list)

    return templates.TemplateResponse("./confirm.html", {
        "request": request, "name_list": show,
        "file_content": app_data.file_content})


@app.get("./confirm.html", response_class=HTMLResponse)
async def read_confirm(request: Request):
    """confirm.HTML 화면"""
    # with open(text_file_path, "r", encoding="utf-8") as f:
    #     file_content = f.read()
    return templates.TemplateResponse("confirm.html", {
        "request": request, "file_content": app_data.file_content
    })


@app.post("/find-speaker")
async def find_speaker(request: Request):
    """발화자를 찾고 싶다."""
    return templates.TemplateResponse("show.html", {
        "request": request, "name_list": app_data.name_list})

# @app.get("/success", response_class=HTMLResponse)
# async def success_page(request: Request):
#     return templates.TemplateResponse("success.html", {"request": request})
