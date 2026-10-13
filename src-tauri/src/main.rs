//! Atelier AI Tauri 셸.
//!
//! # 백엔드 실행 전략 (MVP)
//!
//! 앱 시작 시 Python 백엔드(FastAPI/uvicorn)를 자식 프로세스로 띄우고,
//! 앱 종료 시 함께 종료한다.
//!
//! 백엔드 탐색 순서:
//! 1. `ATELIER_BACKEND_CMD` 환경 변수 — 전체 명령을 직접 지정
//!    (예: `ATELIER_BACKEND_CMD="/opt/atelier/run-backend.sh"`)
//! 2. 실행 파일 옆의 `atelier-backend` 사이드카 바이너리
//!    (향후 PyInstaller 등으로 얼린 배포용 바이너리 자리)
//! 3. 개발 레이아웃 — 실행 파일 기준 `../../backend` 디렉토리가 있으면
//!    `python3 -m uvicorn atelier.api.app:app` 실행
//!
//! 어떤 방법으로도 백엔드를 띄우지 못하면 앱은 그대로 실행되고,
//! 프론트엔드는 내장 기본값(폴백) 모드로 동작한다.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: &str = "8079";

/// 실행 중인 백엔드 자식 프로세스. 앱 종료 시 kill 한다.
struct BackendProcess(Mutex<Option<Child>>);

struct SpawnSpec {
    program: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
}

fn find_backend() -> Option<SpawnSpec> {
    // 1. 환경 변수로 직접 지정
    if let Ok(cmd) = std::env::var("ATELIER_BACKEND_CMD") {
        let mut parts = cmd.split_whitespace();
        if let Some(program) = parts.next() {
            return Some(SpawnSpec {
                program: program.to_string(),
                args: parts.map(str::to_string).collect(),
                cwd: std::env::var("ATELIER_BACKEND_DIR").ok().map(PathBuf::from),
            });
        }
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))?;

    // 2. 사이드카 바이너리 (배포용).
    //    얼린 백엔드는 인자 없이 실행하면 127.0.0.1:8079 에서 서빙한다고 가정한다.
    #[cfg(target_os = "windows")]
    let sidecar_name = "atelier-backend.exe";
    #[cfg(not(target_os = "windows"))]
    let sidecar_name = "atelier-backend";
    let sidecar = exe_dir.join(sidecar_name);
    if sidecar.is_file() {
        return Some(SpawnSpec {
            program: sidecar.to_string_lossy().into_owned(),
            args: Vec::new(),
            cwd: None,
        });
    }

    // 3. 개발 레이아웃: <exe>/../../backend (src-tauri/target/{debug,release}/ 기준)
    let dev_backend = exe_dir.join("../../backend");
    if dev_backend.join("pyproject.toml").is_file() {
        return Some(SpawnSpec {
            program: "python3".to_string(),
            args: vec![
                "-m".into(),
                "uvicorn".into(),
                "atelier.api.app:app".into(),
                "--host".into(),
                BACKEND_HOST.into(),
                "--port".into(),
                BACKEND_PORT.into(),
            ],
            cwd: Some(dev_backend),
        });
    }

    None
}

fn spawn_backend() -> Option<Child> {
    let spec = find_backend()?;
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args).env("PYTHONUNBUFFERED", "1");
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    match cmd.spawn() {
        Ok(child) => {
            eprintln!(
                "[atelier] 백엔드 시작: {} (pid {})",
                spec.program,
                child.id()
            );
            Some(child)
        }
        Err(e) => {
            eprintln!(
                "[atelier] 백엔드 시작 실패 ({}): {}. 프론트엔드는 기본값 모드로 동작합니다.",
                spec.program, e
            );
            None
        }
    }
}

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let child = spawn_backend();
            if child.is_none() {
                eprintln!("[atelier] 백엔드 없이 실행합니다.");
            }
            app.manage(BackendProcess(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Tauri 앱 실행 실패");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            let state = app_handle.state::<BackendProcess>();
            let mut guard = state.0.lock().unwrap();
            if let Some(mut child) = guard.take() {
                eprintln!("[atelier] 백엔드 종료 중...");
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}
