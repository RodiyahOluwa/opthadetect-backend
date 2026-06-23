"""
reverse_migrate.py
------------------
Restores the old path-based schema from the BLOB-based schema.
Reads image BLOBs from DB, saves them as files to data/images/,
then rebuilds the table with original_path and gradcam_path columns.

Run once from backend/ directory:
    python3 reverse_migrate.py
"""

import os
import sqlite3

DB_PATH = os.path.join("data", "opthadetect.db")
IMG_DIR = os.path.join("data", "images")


def already_reverted(conn):
    cur = conn.execute("PRAGMA table_info(scans)")
    columns = {row[1] for row in cur.fetchall()}
    return "original_path" in columns


def reverse_migrate():
    if not os.path.exists(DB_PATH):
        print("No database found at", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if already_reverted(conn):
        print("Already on old schema — nothing to do.")
        conn.close()
        return

    os.makedirs(IMG_DIR, exist_ok=True)
    print("Starting reverse migration...")

    rows = conn.execute(
        "SELECT id, timestamp, label, confidence, original_img, gradcam_img FROM scans"
    ).fetchall()

    ok = 0
    failed = []

    for row in rows:
        orig_filename = f"{row['timestamp']}_{row['label']}_{row['confidence']:.2f}_orig.png"
        grad_filename = f"{row['timestamp']}_{row['label']}_{row['confidence']:.2f}_gradcam.png"

        orig_path = os.path.join(IMG_DIR, orig_filename)
        grad_path = os.path.join(IMG_DIR, grad_filename)

        try:
            with open(orig_path, "wb") as f:
                f.write(bytes(row["original_img"]))
            with open(grad_path, "wb") as f:
                f.write(bytes(row["gradcam_img"]))
            ok += 1
        except Exception as e:
            print(f"  WARNING: Could not save images for scan id={row['id']}: {e}")
            failed.append(row["id"])

    print(f"  Saved image files for {ok} scan(s).")

    # Add path columns
    conn.execute("ALTER TABLE scans ADD COLUMN original_path TEXT")
    conn.execute("ALTER TABLE scans ADD COLUMN gradcam_path TEXT")
    conn.commit()

    # Fill in the path values
    for row in rows:
        if row["id"] in failed:
            continue
        orig_filename = f"{row['timestamp']}_{row['label']}_{row['confidence']:.2f}_orig.png"
        grad_filename = f"{row['timestamp']}_{row['label']}_{row['confidence']:.2f}_gradcam.png"
        conn.execute(
            "UPDATE scans SET original_path = ?, gradcam_path = ? WHERE id = ?",
            (orig_filename, grad_filename, row["id"]),
        )
    conn.commit()

    # Rebuild table without BLOB columns
    print("  Rebuilding table without BLOB columns...")
    conn.execute(
        """
        CREATE TABLE scans_old (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            patient_name    TEXT,
            patient_id      TEXT,
            patient_age     INTEGER,
            eye             TEXT,
            original_path   TEXT    NOT NULL,
            gradcam_path    TEXT    NOT NULL,
            label           TEXT    NOT NULL,
            confidence      REAL    NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO scans_old
            (id, user_id, timestamp, patient_name, patient_id,
             patient_age, eye, original_path, gradcam_path, label, confidence)
        SELECT
             id, user_id, timestamp, patient_name, patient_id,
             patient_age, eye, original_path, gradcam_path, label, confidence
        FROM scans
        WHERE original_path IS NOT NULL
        """
    )
    conn.execute("DROP TABLE scans")
    conn.execute("ALTER TABLE scans_old RENAME TO scans")
    conn.commit()

    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    print("Reverse migration complete.")
    print(f"Images saved to {IMG_DIR}/")


if __name__ == "__main__":
    reverse_migrate()
