#!/usr/bin/env bash
# =============================================================================
# ARGOS-MED: 100% Offline Edge-Native Maritime Surveillance System
# Demonstration Launcher for the Forum Méditerranéen de l'IA (FMIA)
# Palais du Pharo, Marseille — November 5
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "====================================================================="
echo " ARGOS-MED // MARITIME DOMAIN AWARENESS SYSTEM"
echo " Forum Méditerranéen de l'IA (FMIA) - Palais du Pharo, Marseille"
echo " Developed by GDG on Campus USTO-MB (Oran, Algeria)"
echo " Mode: 100% OFFLINE / ZERO-CLOUD AIR-GAPPED VERIFICATION"
echo "====================================================================="

# 1. Environment Verification
export PYTHONUNBUFFERED=1
export ONNXRUNTIME_EXECUTION_PROVIDERS="CPUExecutionProvider"

# 2. Dependency Sanity Check
if ! python3 -c "import fastapi, uvicorn, onnxruntime, scipy, pydantic, psutil" 2>/dev/null; then
    echo "[!] Missing core dependencies. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# 3. Check for Local SLM (Ollama)
if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[+] Local Ollama daemon detected at 127.0.0.1:11434 (LLaMA 3.2 1B Active)."
else
    echo "[i] Local SLM daemon not detected. Automatic fallback to deterministic STANAG 1059 naval engine."
fi

# 4. Launch Local Tactical Server
echo "[+] Initializing Tactical Maritime C2 Server on http://127.0.0.1:8000 ..."
echo "[+] Tactical UI accessible at: http://127.0.0.1:8000"
echo "====================================================================="

exec uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level info
