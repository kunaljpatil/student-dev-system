#!/bin/sh
set -e

echo "[*] Starting AI Student Development System container..."

# Run database migrations
echo "[*] Applying database migrations via Alembic..."
alembic upgrade head

# Seed database if not seeded
echo "[*] Checking and seeding database records..."
python scripts/seed_db.py || echo "[!] Database already seeded or seed completed."

# Train models if artifacts are missing
if [ ! -f "app/ml/artifacts/regressor.joblib" ]; then
    echo "[*] Serialized ML models not found. Running ML training pipeline..."
    python scripts/train_all.py
else
    echo "[OK] ML model artifacts verified."
fi

echo "[+] Starting Uvicorn production server on port 8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
