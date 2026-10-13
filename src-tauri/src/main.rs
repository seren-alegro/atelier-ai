//! Atelier AI Tauri 셸.
//! Python 백엔드(FastAPI)를 사이드카로 띄우고 웹 UI를 표시한다.

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("Tauri 앱 실행 실패");
}
