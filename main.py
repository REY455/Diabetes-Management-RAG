from fastapi import FastAPI
from routers.document_management import router as docs_router
from routers.embedding import router as embed_router
from routers.chat import router as chat_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.doc_chat import router as doc_chat_router
app = FastAPI()


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    docs_router,
    prefix="/api/docs",
    tags=["Document Management"]
)

app.include_router(
    embed_router,
    prefix="/api/embed",
    tags=["Embedding"]
)

app.include_router(
   chat_router,
    prefix="/api/chat",
    tags=["Chat"]
)

app.include_router(
   doc_chat_router,
    prefix="/api/doc-chat",
    tags=["Doc Chat"]
)


@app.get("/")
def root():
    return {"message": "API is running 🚀"}