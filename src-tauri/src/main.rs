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
//! 2. 번들에 포함된 frozen 백엔드 (`resources/backend-dist/atelier-backend`)
//!    — 릴리즈 빌드에서 PyInstaller로 얼린 백엔드.
//! 3. 실행 파일 옆의 `atelier-backend` 사이드카 바이너리
//!    (수동 배치용 자리)
//! 4. 개발 레이아웃 — 실행 파일 기준 `../../../backend` 디렉토리가 있으면
//!    `backend/.venv`의 Python을 우선 사용하고, 없으면 OS에 맞는 Python으로
//!    `uvicorn atelier.api.app:app` 실행.
//!    (Windows: `py -3` → `python` → `python3`, Unix: `python3` → `python` 순으로 탐색.)
//!
//! 어떤 방법으로도 백엔드를 띄우지 못하면 앱은 그대로 실행되고,
//! 프론트엔드는 내장 기본값(폴백) 모드로 동작한다.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: &str = "8079";

/// 실행 중인 백엔드 자식 프로세스. 앱 종료 시 kill 한다.
/// (이미 떠 있던 백엔드를 재사용한 경우에는 None이라 kill하지 않는다.)
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

/// 따옴표를 인식하는 명령줄 분리.
/// `ATELIER_BACKEND_CMD="/opt/my dir/run.sh" --foo` 같은 공백 포함 경로를 다룬다.
fn split_cmd(cmd: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in cmd.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            c if c.is_whitespace() && !in_quotes => {
                if !cur.is_empty() {
                    parts.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// 백엔드가 이미 떠 있는지 확인 (포트 선점 여부).
fn backend_alive() -> bool {
    std::net::TcpStream::connect(format!("{}:{}", BACKEND_HOST, BACKEND_PORT)).is_ok()
}

fn find_backend(resource_dir: Option<&Path>) -> Option<SpawnSpec> {
    // 1. 환경 변수로 직접 지정
    if let Ok(cmd) = std::env::var("ATELIER_BACKEND_CMD") {
        let mut parts = split_cmd(&cmd).into_iter();
        if let Some(program) = parts.next() {
            return Some(SpawnSpec {
                program,
                args: parts.collect(),
                cwd: std::env::var("ATELIER_BACKEND_DIR").ok().map(PathBuf::from),
            });
        }
    }

    // 2. 번들에 포함된 frozen 백엔드 (릴리즈 빌드)
    if let Some(res) = resource_dir {
        #[cfg(target_os = "windows")]
        let frozen = res.join("backend-dist").join("atelier-backend").join("atelier-backend.exe");
        #[cfg(not(target_os = "windows"))]
        let frozen = res.join("backend-dist").join("atelier-backend").join("atelier-backend");
        if frozen.is_file() {
            return Some(SpawnSpec {
                program: frozen.to_string_lossy().into_owned(),
                args: Vec::new(),
                // frozen 백엔드는 데이터 디렉토리를 스스로 찾으므로 cwd 불필요
                cwd: None,
            });
        }
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))?;

    // 3. 실행 파일 옆의 사이드카 바이너리 (수동 배치용).
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

    // 4. 개발 레이아웃: <exe>/../../../backend (src-tauri/target/{debug,release}/ 기준)
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

fn spawn_backend(resource_dir: Option<&Path>) -> Option<Child> {
    let spec = find_backend(resource_dir)?;
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args).env("PYTHONUNBUFFERED", "1");
    // Windows: 백엔드 실행 시 콘솔 창이 팝업되지 않게
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
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

/// 프론트엔드에서 호출: 업데이트 설치 전 백엔드를 먼저 종료한다.
/// (Windows에서 실행 중인 백엔드 파일이 잠겨 업데이트 설치가 실패하는 것을 방지)
#[tauri::command]
fn stop_backend(state: tauri::State<BackendProcess>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        eprintln!("[atelier] 업데이트 전 백엔드 종료 중...");
        child.kill().map_err(|e| e.to_string())?;
        let _ = child.wait();
    }
    Ok(())
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            // 번들 리소스 디렉토리 (frozen 백엔드 탐색용)
            let resource_dir = app.path().resource_dir().ok();
            // 이미 떠 있는 백엔드가 있으면 새로 띄우지 않고 재사용한다.
            // (포트 선점 시 uvicorn이 조용히 죽는 문제를 방지)
            let child = if backend_alive() {
                eprintln!("[atelier] 이미 실행 중인 백엔드를 사용합니다.");
                None
            } else {
                spawn_backend(resource_dir.as_deref())
            };
            if child.is_none() && !backend_alive() {
                eprintln!("[atelier] 백엔드 없이 실행합니다.");
            }
            app.manage(BackendProcess(Mutex::new(child)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![stop_backend])
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
