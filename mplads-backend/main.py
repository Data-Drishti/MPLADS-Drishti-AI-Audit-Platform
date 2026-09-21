from engines.financialAndAnomalyCore import AuditingEngine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MPLADS-Drishti API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
  financial_engine = AuditingEngine()
  print("[+] Financial Anomaly Engine initialized successfully.")
except Exception as e:
  financial_engine = None
  print(f"[!] Warning: Could not initialize engine: {e}")


class AuditRequest(BaseModel):
  amount: float | str
  category: str = "Normal/Others"


@app.post("/api/audit/financial")
def audit_financial_endpoint(payload: AuditRequest):
  if financial_engine is None:
    raise HTTPException(
        status_code=503,
        detail="Financial model artifact is not loaded. Run train_iforest.py.",
    )

  result = financial_engine.audit_amount(payload.amount, payload.category)
  return {"status": "success", "data": result}


@app.get("/api/health")
def health():
  return {"status": "healthy", "service": "MPLADS-Drishti Backend"}