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

/** POST /api/generate — job 생성 후 폴링으로 완료 대기, 결과 URL 목록 반환.
 *  반환된 promise에 .cancel()이 있다. 30분 타임아웃.
 *  네트워크 에러 메시지는 한국어로 바꿔서 던진다. */
export function requestGenerate(params, mode = 'txt2img', image = null, mask = null) {
  const TIMEOUT_MS = 30 * 60 * 1000;
  let cancelled = false;
  const promise = (async () => {
    // loras·upscale_strength는 settings 검증 대상이 아니므로 params에서 분리.
    // 백엔드는 요청의 loras를 파이프라인에 실제로 적용한다.
    const { loras, upscale_strength, ...rest } = params;
    const body = { mode, params: rest, image, mask };
    if (loras) body.loras = loras;
    let res;
    try {
      res = await fetch(API_BASE + '/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch {
      throw new Error('백엔드에 연결할 수 없어요. 백엔드가 켜져 있는지 확인해주세요.');
    }
    if (res.status === 422) {
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      throw new Error(
        typeof detail === 'string' ? detail
        : Array.isArray(detail) ? detail.join(', ')
        : '설정값이 올바르지 않아요.'
      );
    }
    if (res.status === 503) {
      // 백엔드가 보낸 detail을 우선 표시 (GPU 없음 / 모델 미설정 구분)
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      throw new Error(
        typeof detail === 'string' && detail
          ? detail
          : '모델이 로드되지 않았어요. 백엔드에서 모델 경로를 설정해주세요.'
      );
    }
    if (res.status === 413) {
      throw new Error('이미지가 너무 커요. 더 작은 이미지를 올려주세요.');
    }
    if (!res.ok) throw new Error(`요청 실패 (HTTP ${res.status})`);
    const { job_id } = await res.json();

    // 완료까지 폴링
    const start = Date.now();
    for (;;) {
      if (cancelled) throw new Error('취소됐어요.');
      if (Date.now() - start > TIMEOUT_MS) {
        throw new Error('시간이 너무 오래 걸려 중단했어요. (30분 초과)');
      }
      await new Promise((r) => setTimeout(r, 1000));
      let job;
      try {
        job = await getJson(`/api/jobs/${job_id}`);
      } catch {
        throw new Error('백엔드와의 연결이 끊겼어요.');
      }
      if (!job) throw new Error('작업 상태를 확인할 수 없어요.');
      if (job.status === 'done') {
        return job.results.map((u) => API_BASE + u);
      }
      if (job.status === 'error') {
        throw new Error(job.error || '생성 중 오류가 발생했어요.');
      }
    }
  })();
  promise.cancel = () => { cancelled = true; };
  return promise;
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

/** POST /api/pose-detect — 사진에서 OpenPose 스켈레톤 추출.
 *  첫 호출 시 모델 다운로드로 수십 초~수 분 걸릴 수 있어 타임아웃을 두지 않는다.
 *  성공 시 base64 PNG 문자열 반환. */
export async function requestPoseDetect(imageB64) {
  const res = await fetch(API_BASE + '/api/pose-detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageB64 }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  if (!data.skeleton) throw new Error('스켈레톤을 받지 못했어요.');
  return data.skeleton;
}

/** POST /api/upscale — RealESRGAN(4x) 업스케일.
 *  image: 결과 파일명(output dir 안) 또는 base64 이미지.
 *  strength: 0.5~1.0 (1=풀 업스케일, 0.5=절반만 적용).
 *  첫 호출 시 모델 다운로드(18MB)로 수 분 걸릴 수 있어 타임아웃을 두지 않는다.
 *  성공 시 완성된 이미지의 절대 URL 반환. */
export async function requestUpscale(image, strength = 0.5) {
  let res;
  try {
    res = await fetch(API_BASE + '/api/upscale', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, strength }),
    });
  } catch {
    throw new Error('백엔드에 연결할 수 없어요. 백엔드가 켜져 있는지 확인해주세요.');
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 422) {
    throw new Error(
      typeof data.detail === 'string' ? data.detail : '업스케일 설정값이 올바르지 않아요.'
    );
  }
  if (res.status === 503) {
    throw new Error(
      typeof data.detail === 'string' && data.detail
        ? data.detail
        : '업스케일에는 GPU가 필요해요.'
    );
  }
  if (res.status === 404) throw new Error('원본 이미지를 찾을 수 없어요.');
  if (res.status === 413) throw new Error('이미지가 너무 커요. 더 작은 이미지를 올려주세요.');
  if (!res.ok) throw new Error(data.detail || `업스케일 실패 (HTTP ${res.status})`);
  if (!data.url) throw new Error('업스케일 결과를 받지 못했어요.');
  return API_BASE + data.url;
}

/** POST /api/train — LoRA 학습 요청. {job_id, total_steps} 반환. */
export async function requestTrain(trainParams) {
  const res = await fetch(API_BASE + '/api/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(trainParams),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    throw new Error(
      typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.join(', ')
      : `학습 요청 실패 (HTTP ${res.status})`
    );
  }
  return data;
}

/** GET /api/train-jobs/{job_id} — 학습 진행 상태 조회. */
export async function getTrainJob(jobId) {
  const res = await fetch(API_BASE + '/api/train-jobs/' + jobId);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/** POST /api/caption — 이미지 한 장 자동 캡션. */
export async function requestCaption(imagePath, method = 'wd14') {
  const res = await fetch(API_BASE + '/api/caption', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_path: imagePath, method }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data.caption;
}

/** POST /api/caption/batch — 폴더 전체 일괄 캡션. */
export async function requestCaptionBatch(datasetDir, method = 'wd14', overwrite = false) {
  const res = await fetch(API_BASE + '/api/caption/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_dir: datasetDir, method, overwrite }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}
