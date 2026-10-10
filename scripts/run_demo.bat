@echo off
REM =============================================================================
REM ARGOS-MED: 100% Offline Edge-Native Maritime Surveillance System
REM Windows Demonstration Launcher for FMIA (Palais du Pharo, Marseille)
REM =============================================================================

echo =====================================================================
echo  ARGOS-MED // MARITIME DOMAIN AWARENESS SYSTEM
echo  Forum Mediterraneen de l'IA (FMIA) - Palais du Pharo, Marseille
echo  Developed by GDG on Campus USTO-MB (Oran, Algeria)
echo  Mode: 100%% OFFLINE / ZERO-CLOUD AIR-GAPPED VERIFICATION
echo =====================================================================

set PYTHONUNBUFFERED=1

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level info
pause
