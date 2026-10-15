# PyInstaller spec: Atelier AI 백엔드 동결.
# 사용법: cd backend && uv run pyinstaller atelier-backend.spec --noconfirm
# 산출물: dist/atelier-backend/ (onedir) — Tauri가 resources로 묶는다.
#
# 주의: torch/transformers/diffusers는 collect_all로 통째로 묶는다.
# (hidden import 누락이 잦은 패키지라 용량을 감수하고 안전하게 간다.)

from PyInstaller.utils.hooks import collect_all

datas = [
    # 프롬프트 태그 검색용 단보루 태그 CSV (24MB, 카테고리 1·3 제외 후 로드)
    ("atelier/data/danbooru-tags-kr.csv", "atelier/data"),
]
binaries = []
hiddenimports = [
    # uvicorn이 동적으로 고르는 모듈들
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

for pkg in ("torch", "transformers", "diffusers", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["atelier/backend_main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="atelier-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # 백엔드 로그 확인용 (Tauri가 콘솔 숨김은 별도 처리)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="atelier-backend",
)
