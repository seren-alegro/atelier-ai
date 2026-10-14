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
//! 3. 개발 레이아웃 — 실행 파일 기준 `../../../backend` 디렉토리가 있으면
//!    `backend/.venv`의 Python을 우선 사용하고, 없으면 OS에 맞는 Python으로
//!    `uvicorn atelier.api.app:app` 실행.
//!    (Windows: `py -3` → `python` → `python3`, Unix: `python3` → `python` 순으로 탐색.)
//!
//! 어떤 방법으로도 백엔드를 띄우지 못하면 앱은 그대로 실행되고,
//! 프론트엔드는 내장 기본값(폴백) 모드로 동작한다.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: &str = "8079";

/// 실행 중인 백엔드 자식 프로세스. 앱 종료 시 kill 한다.
struct BackendProcess(Mutex<Option<Child>>);

struct SpawnSpec {
    program: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
}

/// Python 인터프리터 탐색.
/// Windows: `py -3` → `python` → `python3`, Unix: `python3` → `python`.
/// (프로그램, 앞에 붙일 인자) 형태로 반환한다. `--version`으로 실제 동작 확인.
fn find_python() -> Option<(String, Vec<String>)> {
    #[cfg(target_os = "windows")]
    let candidates: &[(&str, &[&str])] = &[("py", &["-3"]), ("python", &[]), ("python3", &[])];
    #[cfg(not(target_os = "windows"))]
    let candidates: &[(&str, &[&str])] = &[("python3", &[]), ("python", &[])];

    for (program, prefix) in candidates {
        let mut check = Command::new(program);
        check.args(*prefix).arg("--version");
        if check.output().map(|o| o.status.success()).unwrap_or(false) {
            return Some((
                program.to_string(),
                prefix.iter().map(|s| s.to_string()).collect(),
            ));
        }
    }
    None
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

    // 3. 개발 레이아웃: <exe>/../../../backend (src-tauri/target/{debug,release}/ 기준)
    //    backend/.venv 의 Python을 우선 사용하고, 없으면 OS에 맞는 Python 탐색
    let dev_backend = exe_dir.join("../../../backend");
    if dev_backend.join("pyproject.toml").is_file() {
        #[cfg(target_os = "windows")]
        let venv_python = dev_backend.join(".venv").join("Scripts").join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let venv_python = dev_backend.join(".venv").join("bin").join("python");

        let python_spec: Option<(String, Vec<String>)> = if venv_python.is_file() {
            Some((venv_python.to_string_lossy().into_owned(), Vec::new()))
        } else {
            find_python()
        };

        if let Some((python, prefix)) = python_spec {
            let mut args = prefix;
            args.extend(
                [
                    "-m",
                    "uvicorn",
                    "atelier.api.app:app",
                    "--host",
                    BACKEND_HOST,
                    "--port",
                    BACKEND_PORT,
                ]
                .iter()
                .map(|s| s.to_string()),
            );
            return Some(SpawnSpec {
                program: python,
                args,
                cwd: Some(dev_backend),
            });
        }
        eprintln!("[atelier] Python을 찾지 못했습니다. python.org에서 설치해주세요.");
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
