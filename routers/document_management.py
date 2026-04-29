# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# import os
# import shutil
# import uuid

# from fastapi import APIRouter

# router = APIRouter()

# DOCS_DIR = "documents"
# os.makedirs(DOCS_DIR, exist_ok=True)


# # ==============================
# # 🔹 CREATE (Upload PDF)
# # ==============================
# @router.post("/")
# async def upload_doc(file: UploadFile = File(...)):

#     filename = file.filename  # ✅ keep original name
#     file_path = os.path.join(DOCS_DIR, filename)

#     # 🔥 handle duplicate names
#     if os.path.exists(file_path):
#         name, ext = os.path.splitext(filename)
#         counter = 1

#         while os.path.exists(file_path):
#             filename = f"{name}_{counter}{ext}"
#             file_path = os.path.join(DOCS_DIR, filename)
#             counter += 1

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     return {
#         "message": "File uploaded",
#         "filename": filename
#     }

# # ==============================
# # 🔹 READ (List all docs)
# # ==============================
# @router.get("/")
# def list_docs():
#     files = os.listdir(DOCS_DIR)
#     return {"documents": files}


# # ==============================
# # 🔹 READ (Get specific doc)
# # ==============================
# @router.get("/{file_name}")
# def get_doc(file_name: str):
#     file_path = os.path.join(DOCS_DIR, file_name)

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     return FileResponse(file_path)


# # ==============================
# # 🔹 UPDATE (Replace doc)
# # ==============================
# @router.put("/{file_name}")
# async def update_doc(file_name: str, file: UploadFile = File(...)):
#     file_path = os.path.join(DOCS_DIR, file_name)

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     return {"message": "File updated"}


# # ==============================
# # 🔹 DELETE
# # ==============================
# @router.delete("/{file_name}")
# def delete_doc(file_name: str):
#     file_path = os.path.join(DOCS_DIR, file_name)

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     os.remove(file_path)
#     return {"message": "File deleted"}
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import uuid
import fitz  # PyMuPDF

router = APIRouter()

DOCS_DIR = "documents"
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(DOCS_DIR, exist_ok=True)


# ==============================
# 🔹 UTILS
# ==============================
def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF files are allowed")

    # 🔥 reliable size check
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 10MB)")


def safe_path(filename: str):
    filename = os.path.basename(filename)
    filename = filename.replace(" ", "_")
    return os.path.join(DOCS_DIR, filename)


def is_valid_pdf(file_path: str):
    try:
        fitz.open(file_path)
        return True
    except:
        return False


# ==============================
# 🔹 CREATE (Upload PDF)
# ==============================
@router.post("/")
async def upload_doc(file: UploadFile = File(...)):

    validate_file(file)

    # 🔥 unique filename (avoid collisions)
    filename = f"{uuid.uuid4().hex}.pdf"
    file_path = safe_path(filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 validate actual PDF content
    if not is_valid_pdf(file_path):
        os.remove(file_path)
        raise HTTPException(400, "Invalid PDF file")

    return {
        "message": "File uploaded successfully 🚀",
        "filename": filename
    }


# ==============================
# 🔹 READ (List docs)
# ==============================
@router.get("/")
def list_docs():
    files = sorted(os.listdir(DOCS_DIR))
    return {"documents": files}


# ==============================
# 🔹 READ (Get doc)
# ==============================
@router.get("/{file_name}")
def get_doc(file_name: str):

    file_path = safe_path(file_name)

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    return FileResponse(file_path, filename=file_name)


# ==============================
# 🔹 UPDATE
# ==============================
@router.put("/{file_name}")
async def update_doc(file_name: str, file: UploadFile = File(...)):

    validate_file(file)

    file_path = safe_path(file_name)

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 validate after update
    if not is_valid_pdf(file_path):
        os.remove(file_path)
        raise HTTPException(400, "Invalid PDF file")

    return {"message": "File updated successfully"}


# ==============================
# 🔹 DELETE
# ==============================
@router.delete("/{file_name}")
def delete_doc(file_name: str):

    file_path = safe_path(file_name)

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    os.remove(file_path)

    # 🔥 OPTIONAL: delete embeddings too
    safe_name = os.path.splitext(file_name)[0].replace(" ", "_")
    shutil.rmtree(f"embeddings/{safe_name}", ignore_errors=True)

    return {"message": "File deleted successfully"}