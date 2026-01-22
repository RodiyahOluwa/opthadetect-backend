# main.py
from typing import List, Optional

import datetime
import io
import os
import secrets
import sqlite3
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



# ---------- CONFIG ----------
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "opthadetect.db")

os.makedirs(IMG_DIR, exist_ok=True)

# ---------- DB SETUP ----------
def get_conn() -> sqlite3.Connection:
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn


def init_db() -> None:
  conn = get_conn()
  cur = conn.cursor()
  cur.execute(
      """
      CREATE TABLE IF NOT EXISTS scans (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          patient_name TEXT,
          patient_id TEXT,
          patient_age INTEGER,
          eye TEXT,
          original_path TEXT NOT NULL,
          gradcam_path TEXT NOT NULL,
          label TEXT NOT NULL,
          confidence REAL NOT NULL
      )
      """
  )
  conn.commit()
  conn.close()


init_db()

# ---------- AUTH (DEMO ONLY) ----------
DEMO_USER = "doctor@example.com"
DEMO_PASS = "optha123"
DEMO_TOKEN = secrets.token_hex(16)


class LoginRequest(BaseModel):
  email: str
  password: str


class LoginResponse(BaseModel):
  access_token: str


# ---------- RESPONSE MODELS ----------
class PredictionResponse(BaseModel):
  label: str
  confidence: float
  original_url: str
  gradcam_url: str
  timestamp: str
  scan_id: int
  patient_name: Optional[str] = None
  patient_id: Optional[str] = None
  patient_age: Optional[int] = None
  eye: Optional[str] = None


class ScanRecord(BaseModel):
  id: int
  timestamp: str
  label: str
  confidence: float
  original_url: str
  gradcam_url: str
  patient_name: Optional[str] = None
  patient_id: Optional[str] = None
  patient_age: Optional[int] = None
  eye: Optional[str] = None


def get_current_user(token: str) -> str:
  if token != DEMO_TOKEN:
    raise HTTPException(status_code=401, detail="Invalid token")
  return DEMO_USER


# ---------- MODEL SETUP ----------
device = torch.device("cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(
    torch.load("resnet50_dr_classifier.pth", map_location=device)
)
model.to(device)
model.eval()

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

# ---------- FASTAPI APP ----------
app = FastAPI(title="OpthaDetect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=IMG_DIR), name="static")
FRONTEND_DIR = "Static"  # contains index.html, assets/, vite.svg

# Serve React build (Vite)
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

@app.get("/vite.svg", include_in_schema=False)
def vite_svg():
    return FileResponse(os.path.join(FRONTEND_DIR, "vite.svg"))

@app.get("/", include_in_schema=False)
def frontend_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---------- ROUTES ----------


@app.post("/auth/login", response_model=LoginResponse)
def login_endpoint(body: LoginRequest) -> LoginResponse:
  if body.email == DEMO_USER and body.password == DEMO_PASS:
    return LoginResponse(access_token=DEMO_TOKEN)
  raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/predict", response_model=PredictionResponse)
async def predict_retinopathy_api(
    file: UploadFile = File(...),
    patient_name: str = Form(""),
    patient_id: str = Form(""),
    patient_age: Optional[int] = Form(None),
    eye: str = Form(""),
    token: str = Depends(get_current_user),
) -> PredictionResponse:
  # Read image
  contents = await file.read()
  try:
    image = Image.open(io.BytesIO(contents)).convert("RGB")
  except Exception:
    raise HTTPException(status_code=400, detail="Invalid image file")

  timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  img_resized = image.resize((224, 224))
  img_tensor = transform(img_resized).unsqueeze(0).to(device)

  with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred].item())

  label = "DR" if pred == 0 else "NoDR"

  # Grad-CAM
  rgb_img_np = np.array(img_resized).astype(np.float32) / 255.0
  rgb_img_np = np.ascontiguousarray(rgb_img_np)
  grayscale_cam = cam(
      input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred)]
  )[0]
  cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
  cam_pil = Image.fromarray(cam_image)

  # Save images
  base_name = f"{timestamp}_{label}_{confidence:.2f}"
  orig_filename = f"{base_name}_orig.png"
  grad_filename = f"{base_name}_gradcam.png"

  orig_path = os.path.join(IMG_DIR, orig_filename)
  grad_path = os.path.join(IMG_DIR, grad_filename)

  image.save(orig_path)
  cam_pil.save(grad_path)

  # Store record in DB
  conn = get_conn()
  cur = conn.cursor()
  cur.execute(
      """
      INSERT INTO scans (
          user_id, timestamp,
          patient_name, patient_id, patient_age, eye,
          original_path, gradcam_path,
          label, confidence
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      (
          DEMO_USER,
          timestamp,
          patient_name or None,
          patient_id or None,
          patient_age,
          eye or None,
          orig_filename,
          grad_filename,
          label,
          confidence,
      ),
  )
  scan_id = cur.lastrowid
  conn.commit()
  conn.close()

  return PredictionResponse(
      label=label,
      confidence=confidence,
      original_url=f"/static/{orig_filename}",
      gradcam_url=f"/static/{grad_filename}",
      timestamp=timestamp,
      scan_id=scan_id,
      patient_name=patient_name or None,
      patient_id=patient_id or None,
      patient_age=patient_age,
      eye=eye or None,
  )


@app.get("/records", response_model=List[ScanRecord])
def list_records(token: str = Depends(get_current_user)) -> List[ScanRecord]:
  conn = get_conn()
  cur = conn.cursor()
  cur.execute(
      """
      SELECT
          id, timestamp, label, confidence,
          original_path, gradcam_path,
          patient_name, patient_id, patient_age, eye
      FROM scans
      WHERE user_id = ?
      ORDER BY id DESC
      """,
      (DEMO_USER,),
  )
  rows = cur.fetchall()
  conn.close()

  return [
      ScanRecord(
          id=row["id"],
          timestamp=row["timestamp"],
          label=row["label"],
          confidence=row["confidence"],
          original_url=f"/static/{row['original_path']}",
          gradcam_url=f"/static/{row['gradcam_path']}",
          patient_name=row["patient_name"],
          patient_id=row["patient_id"],
          patient_age=row["patient_age"],
          eye=row["eye"],
      )
      for row in rows
  ]


@app.get("/report/{scan_id}")
def generate_report(scan_id: int, token: str = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM scans WHERE id = ? AND user_id = ?",
        (scan_id, DEMO_USER),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    orig_path = os.path.join(IMG_DIR, row["original_path"])
    grad_path = os.path.join(IMG_DIR, row["gradcam_path"])

    if not os.path.exists(orig_path) or not os.path.exists(grad_path):
        raise HTTPException(status_code=404, detail="Images not found")

    # Load images
    orig = Image.open(orig_path).convert("RGB")
    grad = Image.open(grad_path).convert("RGB")

    # Resize for layout
    target_height = 550
    margin = 50
    gap = 40

    def resize_img(im: Image.Image) -> Image.Image:
        w, h = im.size
        scale = target_height / h
        return im.resize((int(w * scale), target_height))

    orig_r = resize_img(orig)
    grad_r = resize_img(grad)

    # Canvas size (a bit taller so there is room for footer)
    canvas_width = orig_r.width + grad_r.width + gap + margin * 2
    canvas_height = target_height + 480  # was 350 – now taller for footer

    report = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(report)

    # Load nicer fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 42)
        font_sub = ImageFont.truetype("arial.ttf", 28)
        font_normal = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()

    y = margin

    # Title
    draw.text(
        (margin, y),
        "OpthaDetect · Diabetic Retinopathy Report",
        fill="black",
        font=font_title,
    )
    y += 60

    # Patient details
    draw.text(
        (margin, y),
        f"Patient: {row['patient_name'] or '-'}   ID: {row['patient_id'] or '-'}",
        fill="black",
        font=font_sub,
    )
    y += 35

    draw.text(
        (margin, y),
        f"Age: {row['patient_age'] or '-'}   Eye: {row['eye'] or '-'}",
        fill="black",
        font=font_sub,
    )
    y += 35

    draw.text(
        (margin, y),
        f"Timestamp: {row['timestamp']}",
        fill="black",
        font=font_sub,
    )
    y += 50

    # Insert images
    report.paste(orig_r, (margin, y))
    report.paste(grad_r, (margin + orig_r.width + gap, y))
    y += target_height + 40

    # Prediction details
    draw.text(
        (margin, y),
        f"Result: {row['label']}",
        fill="black",
        font=font_sub,
    )
    y += 35

    draw.text(
        (margin, y),
        f"Confidence: {row['confidence']:.2f}",
        fill="black",
        font=font_sub,
    )
    y += 40  # extra gap before disclaimer

    # Disclaimer – now clearly below the results section
    disclaimer = "Prototype tool. Not approved for independent clinical use."
    draw.text(
        (margin, y),
        disclaimer,
        fill="gray",
        font=font_small,
    )

    # Save the PDF
    pdf_name = f"report_{scan_id}.pdf"
    pdf_path = os.path.join(DATA_DIR, pdf_name)
    report.save(pdf_path, "PDF", resolution=300)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"OpthaDetect_Report_{scan_id}.pdf",
    )

#@app.get("/{path:path}", include_in_schema=False)
#def frontend_fallback(path: str):
    # Let API and files behave normally
    if path.startswith(("auth", "predict", "records", "report", "docs", "openapi.json", "static", "assets", "vite.svg")):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
