# main.py
from typing import List, Optional
import datetime
import hashlib
import io
import os
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
    Query,
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

# ------------------------------------------------------------------ #
#  ADMIN CONFIG                                                        #
#  Set ADMIN_EMAIL to your own email. Admin can see ALL scans and     #
#  has a dedicated /admin/scans endpoint.                             #
# ------------------------------------------------------------------ #
ADMIN_EMAIL = "admin@opthadetect.com"   # ← change to your email
ADMIN_PASS  = "NewSecurePassword123"        # ← change to a strong password

os.makedirs(IMG_DIR, exist_ok=True)


# ---------- DB SETUP ----------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'clinician'  -- 'clinician' | 'admin'
        )
        """
    )

    # Tokens table (simple token → user mapping)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            token   TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )


    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Scans table — now linked to user_id (integer FK) instead of freeform string
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            patient_name TEXT,
            patient_id TEXT,
            patient_age INTEGER,
            eye TEXT,
            original_path TEXT NOT NULL,
            gradcam_path TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            deleted_by_user INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Migration: add deleted_by_user column to existing databases that lack it
    try:
        cur.execute("ALTER TABLE scans ADD COLUMN deleted_by_user INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass  # Column already exists

    # Seed the admin account
    admin_hash = _hash_password(ADMIN_PASS)
    cur.execute(
        """
        INSERT INTO users (email, password_hash, role)
        VALUES (?, ?, 'admin')
        ON CONFLICT(email) DO UPDATE SET
            password_hash = excluded.password_hash,
            role = 'admin'
        """,
        (ADMIN_EMAIL, admin_hash),
    )

    conn.commit()
    conn.close()


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


init_db()


# ---------- AUTH ----------
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    role: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


def _get_user_from_token(token: str) -> sqlite3.Row:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.email, u.role
        FROM tokens t
        JOIN users u ON t.user_id = u.id
        WHERE t.token = ?
        """,
        (token,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def get_current_user(token: str = Query(None)) -> sqlite3.Row:
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


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
    uploaded_by: Optional[str] = None   # only visible to admin
    deleted_by_user: Optional[bool] = None  # only visible to admin


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=IMG_DIR), name="static")
FRONTEND_DIR = "Static"

app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
    name="assets",
)


@app.get("/vite.svg", include_in_schema=False)
def vite_svg():
    return FileResponse(os.path.join(FRONTEND_DIR, "vite.svg"))


@app.get("/", include_in_schema=False)
def frontend_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---------- AUTH ROUTES ----------

@app.post("/auth/register", response_model=LoginResponse)
def register(body: RegisterRequest) -> LoginResponse:
    """Register a new clinician account."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (body.email,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")

    pw_hash = _hash_password(body.password)
    cur.execute(
        "INSERT INTO users (email, password_hash, role) VALUES (?, ?, 'clinician')",
        (body.email, pw_hash),
    )
    user_id = cur.lastrowid

    token = str(uuid.uuid4())
    cur.execute("INSERT INTO tokens (token, user_id) VALUES (?, ?)", (token, user_id))
    conn.commit()
    conn.close()

    return LoginResponse(access_token=token, role="clinician")


@app.post("/auth/login", response_model=LoginResponse)
def login_endpoint(body: LoginRequest) -> LoginResponse:
    conn = get_conn()
    cur = conn.cursor()
    pw_hash = _hash_password(body.password)
    cur.execute(
        "SELECT id, role FROM users WHERE email = ? AND password_hash = ?",
        (body.email, pw_hash),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    cur.execute("INSERT INTO tokens (token, user_id) VALUES (?, ?)", (token, row["id"]))
    conn.commit()
    conn.close()

    return LoginResponse(access_token=token, role=row["role"])


@app.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (body.email,))
    row = cur.fetchone()

    if row:
        token = str(uuid.uuid4())
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat()
        cur.execute(
            "INSERT INTO password_resets (token, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
            (token, row["id"], expires_at),
        )
        conn.commit()

        # later: send email here
        print(f"RESET LINK: https://opthadetect.com/reset-password?token={token}")

    conn.close()
    return {"detail": "If that email exists, we’ve sent a reset link."}

@app.post("/auth/reset-password")
def reset_password(body: ResetPasswordRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, expires_at, used
        FROM password_resets
        WHERE token = ?
        """,
        (body.token,),
    )
    row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if row["used"] == 1:
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    expires_at = datetime.datetime.fromisoformat(row["expires_at"])
    if datetime.datetime.utcnow() > expires_at:
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    new_hash = _hash_password(body.new_password)
    cur.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_hash, row["user_id"]),
    )
    cur.execute(
        "UPDATE password_resets SET used = 1 WHERE token = ?",
        (body.token,),
    )
    conn.commit()
    conn.close()

    return {"detail": "Password reset successful"}

@app.post("/auth/logout")
def logout(token: str = Query(None)):
    if token:
        conn = get_conn()
        conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return {"detail": "Logged out"}


# ---------- PREDICT ----------

@app.post("/predict", response_model=PredictionResponse)
async def predict_retinopathy_api(
    file: UploadFile = File(...),
    patient_name: str = Form(""),
    patient_id: str = Form(""),
    patient_age: Optional[int] = Form(None),
    eye: str = Form(""),
    user=Depends(get_current_user),
) -> PredictionResponse:
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

    rgb_img_np = np.array(img_resized).astype(np.float32) / 255.0
    rgb_img_np = np.ascontiguousarray(rgb_img_np)
    grayscale_cam = cam(
        input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred)]
    )[0]
    cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)

    base_name = f"{timestamp}_{label}_{confidence:.2f}"
    orig_filename = f"{base_name}_orig.png"
    grad_filename = f"{base_name}_gradcam.png"

    image.save(os.path.join(IMG_DIR, orig_filename))
    cam_pil.save(os.path.join(IMG_DIR, grad_filename))

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
            user["id"],
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


# ---------- RECORDS ----------

@app.get("/records", response_model=List[ScanRecord])
def list_records(user=Depends(get_current_user)) -> List[ScanRecord]:
    """Return only the scans uploaded by the current user (clinicians).
    Admins also see all scans via /admin/scans."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, timestamp, label, confidence,
               original_path, gradcam_path,
               patient_name, patient_id, patient_age, eye
        FROM scans
        WHERE user_id = ? AND deleted_by_user = 0
        ORDER BY id DESC
        """,
        (user["id"],),
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


# ---------- DELETE SCAN (GDPR right to erasure) ----------

@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: int, user=Depends(get_current_user)):
    """
    Clinicians: soft-delete — scan hidden from their view, but images and
    record are retained by admin for model training (Art. 9(2)(j) UK GDPR).
    Admins: hard-delete — permanently removes the record and all image files.
    """
    conn = get_conn()
    cur = conn.cursor()

    if user["role"] == "admin":
        cur.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Scan not found")
        for path_key in ("original_path", "gradcam_path"):
            full_path = os.path.join(IMG_DIR, row[path_key])
            if os.path.exists(full_path):
                os.remove(full_path)
        cur.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()
        conn.close()
        return {"detail": "Scan permanently deleted"}
    else:
        cur.execute(
            "SELECT * FROM scans WHERE id = ? AND user_id = ?",
            (scan_id, user["id"]),
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Scan not found")
        cur.execute("UPDATE scans SET deleted_by_user = 1 WHERE id = ?", (scan_id,))
        conn.commit()
        conn.close()
        return {"detail": "Scan removed from your records"}


# ---------- ADMIN ROUTES ----------

@app.get("/admin/scans", response_model=List[ScanRecord])
def admin_list_all_scans(admin=Depends(require_admin)) -> List[ScanRecord]:
    """Admin-only: view every scan uploaded by every user."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id, s.timestamp, s.label, s.confidence,
               s.original_path, s.gradcam_path,
               s.patient_name, s.patient_id, s.patient_age, s.eye,
               s.deleted_by_user,
               u.email AS uploaded_by
        FROM scans s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.id DESC
        """
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
            uploaded_by=row["uploaded_by"],
            deleted_by_user=row["deleted_by_user"] == 1
        )
        for row in rows
    ]


@app.get("/admin/users")
def admin_list_users(admin=Depends(require_admin)):
    """Admin-only: list all registered users."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, role FROM users ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "email": r["email"], "role": r["role"]} for r in rows]


# ---------- REPORT ----------

@app.get("/report/{scan_id}")
def generate_report(scan_id: int, user=Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()

    if user["role"] == "admin":
        cur.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
    else:
        cur.execute(
            "SELECT * FROM scans WHERE id = ? AND user_id = ?",
            (scan_id, user["id"]),
        )

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    orig_path = os.path.join(IMG_DIR, row["original_path"])
    grad_path = os.path.join(IMG_DIR, row["gradcam_path"])

    if not os.path.exists(orig_path) or not os.path.exists(grad_path):
        raise HTTPException(status_code=404, detail="Images not found")

    orig = Image.open(orig_path).convert("RGB")
    grad = Image.open(grad_path).convert("RGB")

    target_height = 550
    margin = 50
    gap = 40

    def resize_img(im: Image.Image) -> Image.Image:
        w, h = im.size
        scale = target_height / h
        return im.resize((int(w * scale), target_height))

    orig_r = resize_img(orig)
    grad_r = resize_img(grad)

    canvas_width = orig_r.width + grad_r.width + gap + margin * 2
    canvas_height = target_height + 480

    report = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(report)

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
    draw.text((margin, y), "OpthaDetect · Diabetic Retinopathy Report", fill="black", font=font_title)
    y += 60
    draw.text((margin, y), f"Patient: {row['patient_name'] or '-'}   ID: {row['patient_id'] or '-'}", fill="black", font=font_sub)
    y += 35
    draw.text((margin, y), f"Age: {row['patient_age'] or '-'}   Eye: {row['eye'] or '-'}", fill="black", font=font_sub)
    y += 35
    draw.text((margin, y), f"Timestamp: {row['timestamp']}", fill="black", font=font_sub)
    y += 50

    report.paste(orig_r, (margin, y))
    report.paste(grad_r, (margin + orig_r.width + gap, y))
    y += target_height + 40

    draw.text((margin, y), f"Result: {row['label']}", fill="black", font=font_sub)
    y += 35
    draw.text((margin, y), f"Confidence: {row['confidence']:.2f}", fill="black", font=font_sub)
    y += 40
    draw.text((margin, y), "Prototype tool. Not approved for independent clinical use.", fill="gray", font=font_small)

    pdf_name = f"report_{scan_id}.pdf"
    pdf_path = os.path.join(DATA_DIR, pdf_name)
    report.save(pdf_path, "PDF", resolution=300)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"OpthaDetect_Report_{scan_id}.pdf",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
