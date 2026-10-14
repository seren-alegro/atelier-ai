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
  // loras는 settings 검증 대상이 아니므로 params에서 분리해 별도 필드로 전달.
  // 파이프라인의 LoRA 적용 지원 전까지 백엔드는 무시한다.
  const { loras, ...rest } = params;
  const body = { mode, params: rest, image, mask };
  if (loras) body.loras = loras;
  const res = await fetch(API_BASE + '/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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

/** 썸네일 등에 쓰는 절대 URL (API_BASE + path) */
export function apiUrl(path) {
  return API_BASE + path;
}

/** GET /api/history — 실패 시 null */
export async function fetchHistory(limit = 100) {
  try {
    const data = await getJson(`/api/history?limit=${limit}`);
    return Array.isArray(data?.history) ? data.history : null;
  } catch {
    return null;
  }
}

/** DELETE /api/history/{id} — 기록 삭제 (결과 파일은 유지) */
export async function deleteHistory(id) {
  const res = await fetch(
    API_BASE + '/api/history/' + encodeURIComponent(id),
    { method: 'DELETE' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '삭제에 실패했어요.');
  }
  return await res.json();
}

/** POST /api/history/{id}/favorite — 즐겨찾기 토글 */
export async function toggleHistoryFavorite(id) {
  const res = await fetch(
    API_BASE + '/api/history/' + encodeURIComponent(id) + '/favorite',
    { method: 'POST' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '즐겨찾기 변경에 실패했어요.');
  }
  return await res.json();
}

/** GET /api/models — 실패 시 null */
export async function fetchModels() {
  try {
    return await getJson('/api/models');
  } catch {
    return null;
  }
}

/** POST /api/models/active — 활성 체크포인트 전환 */
export async function setActiveModel(name) {
  const res = await fetch(API_BASE + '/api/models/active', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '모델 전환에 실패했어요.');
  }
  return await res.json();
}

/** DELETE /api/models/{name} */
export async function deleteModel(name) {
  const encoded = name.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(API_BASE + '/api/models/' + encoded, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || '삭제에 실패했어요.');
  }
  return await res.json();
}

/** POST /api/models/refresh — 실패 시 null */
export async function refreshModels() {
  try {
    const res = await fetch(API_BASE + '/api/models/refresh', {
      method: 'POST',
    });
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    return null;
  }
}
