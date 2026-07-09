# PyInstaller spec: IUP Student — полное приложение (лаунчер + прокторинг)
# Сборка на Windows: powershell -File scripts\build_student_windows.ps1

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[2]

datas = []
binaries = []
hiddenimports = [
    "agent",
    "agent.paths",
    "agent.ui",
    "agent.launcher",
    "agent.main",
    "agent.student_app",
    "agent.config",
    "agent.precheck",
    "agent.evidence",
    "agent.transport.client",
    "agent.capture.microphone",
    "agent.capture.screen",
    "agent.capture.process_monitor",
    "agent.proctor.engine",
    "agent.proctor.face",
    "agent.proctor.identity",
    "agent.proctor.overlay",
    "agent.security.consent",
    "agent.security.tamper",
    "shared",
    "shared.constants",
    "shared.types",
    "httpx",
    "httpx._transports",
    "httpx._transports.default",
    "certifi",
    "psutil",
    "mss",
    "PIL",
    "PIL.Image",
    "dotenv",
    "numpy",
    "google.protobuf",
]

for package in ("mediapipe", "cv2", "pyaudio"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += collect_submodules("mediapipe")

model_file = ROOT / "models" / "face_landmarker.task"
if model_file.exists():
    datas.append((str(model_file), "models"))

block_cipher = None

a = Analysis(
    [str(ROOT / "agent" / "student_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="IUP Student",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
