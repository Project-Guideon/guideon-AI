from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/")
def root():
    return {"ok": True, "msg": "Go to /docs"}
