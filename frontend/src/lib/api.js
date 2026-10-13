// 백엔드 API 클라이언트. API가 없으면 null을 반환하고 폴백을 사용한다.
const API_BASE = 'http://127.0.0.1:8079';
const TIMEOUT_MS = 3000;

async function getJson(path) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(API_BASE + path, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/** GET /api/settings — 실패 시 null (폴백 사용) */
export async function fetchSettingsMeta() {
  try {
    const data = await getJson('/api/settings');
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

/** POST /api/generate — job 생성 후 폴링으로 완료 대기, 결과 URL 목록 반환 */
export async function requestGenerate(params, mode = 'txt2img', image = null, mask = null) {
  const res = await fetch(API_BASE + '/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, params, image, mask }),
  });
  if (res.status === 422) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '설정값이 올바르지 않아요.');
  }
  if (res.status === 503) {
    throw new Error('모델이 로드되지 않았어요. 백엔드에서 모델 경로를 설정해주세요.');
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const { job_id } = await res.json();

  // 완료까지 폴링
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const job = await getJson(`/api/jobs/${job_id}`);
    if (job.status === 'done') {
      return job.results.map((u) => API_BASE + u);
    }
    if (job.status === 'error') {
      throw new Error(job.error || '생성 중 오류가 발생했어요.');
    }
  }
}
