// 백엔드 API 클라이언트. API가 없으면 null을 반환하고 폴백을 사용한다.
import { t } from './i18n.svelte.js';
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
      throw new Error(t('api.connectFail'));
    }
    if (res.status === 422) {
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      throw new Error(
        typeof detail === 'string' ? detail
        : Array.isArray(detail) ? detail.join(', ')
        : t('api.badSettings')
      );
    }
    if (res.status === 503) {
      // 백엔드가 보낸 detail을 우선 표시 (GPU 없음 / 모델 미설정 구분)
      const err = await res.json().catch(() => ({}));
      const detail = err.detail;
      throw new Error(
        typeof detail === 'string' && detail
          ? detail
          : t('api.noModel')
      );
    }
    if (res.status === 413) {
      throw new Error(t('api.tooLarge'));
    }
    if (!res.ok) throw new Error(t('api.httpFail', { status: res.status }));
    const { job_id } = await res.json();

    // 완료까지 폴링
    const start = Date.now();
    for (;;) {
      if (cancelled) throw new Error(t('api.cancelled'));
      if (Date.now() - start > TIMEOUT_MS) {
        throw new Error(t('api.timeout'));
      }
      await new Promise((r) => setTimeout(r, 1000));
      let job;
      try {
        job = await getJson(`/api/jobs/${job_id}`);
      } catch {
        throw new Error(t('api.disconnected'));
      }
      if (!job) throw new Error(t('api.noJob'));
      if (job.status === 'done') {
        return job.results.map((u) => API_BASE + u);
      }
      if (job.status === 'error') {
        throw new Error(job.error || t('api.generateFail'));
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
    throw new Error(err.detail || t('api.deleteFail'));
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
    throw new Error(err.detail || t('api.favFail'));
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

/** GET /api/tag-search — 단보루 태그 검색. {count, results} 반환.
 *  네트워크 에러 메시지는 한국어로 바꿔서 던진다. */
export async function searchTags(q, limit = 50) {
  let res;
  try {
    res = await fetch(
      API_BASE + '/api/tag-search?q=' + encodeURIComponent(q) + '&limit=' + limit
    );
  } catch {
    throw new Error(t('api.connectFail'));
  }
  if (res.status === 422) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      typeof err.detail === 'string' && err.detail
        ? err.detail
        : t('api.searchEmpty')
    );
  }
  if (res.status === 503) {
    throw new Error(t('api.tagDataFail'));
  }
  if (!res.ok) throw new Error(t('api.searchFail', { status: res.status }));
  return await res.json();
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
    throw new Error(err.detail || t('api.switchFail'));
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
    throw new Error(err.detail || t('api.deleteFail'));
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

/** GET /api/models/text-encoder — 선택된 텍스트 인코더 (실패 시 null) */
export async function fetchTextEncoder() {
  try {
    const res = await fetch(API_BASE + '/api/models/text-encoder');
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** POST /api/models/text-encoder — 텍스트 인코더 선택 (""이면 해제) */
export async function setTextEncoder(name) {
  const res = await fetch(API_BASE + '/api/models/text-encoder', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name || '' }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || t('api.switchFail'));
  }
  return await res.json();
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
  if (!data.skeleton) throw new Error(t('api.noSkeleton'));
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
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 422) {
    throw new Error(
      typeof data.detail === 'string' ? data.detail : t('api.badUpscale')
    );
  }
  if (res.status === 503) {
    throw new Error(
      typeof data.detail === 'string' && data.detail
        ? data.detail
        : t('api.needGpuUpscale')
    );
  }
  if (res.status === 404) throw new Error(t('api.notFound'));
  if (res.status === 413) throw new Error(t('api.tooLarge'));
  if (!res.ok) throw new Error(data.detail || t('api.upscaleFail', { status: res.status }));
  if (!data.url) throw new Error(t('api.upscaleNoResult'));
  return API_BASE + data.url;
}

/** POST /api/adetailer — ADetailer 스타일 얼굴 보정 (얼굴 검출 + 인페인트).
 *  image: 결과 파일명(output dir 안) 또는 base64 이미지.
 *  opts: {prompt, negative_prompt, denoising_strength(0~1, 기본 0.4),
 *         conf(기본 0.5), mask_padding(기본 32)}.
 *  첫 호출 시 모델 다운로드(6MB)로 시간이 걸릴 수 있어 타임아웃을 두지 않는다.
 *  성공 시 {url: 완성된 이미지의 절대 URL, faces_detected: 검출된 얼굴 수} 반환.
 *  얼굴이 없으면 faces_detected가 0이고 url은 원본 그대로. */
export async function requestAdetailer(image, opts = {}) {
  const {
    prompt = '',
    negative_prompt = '',
    denoising_strength = 0.4,
    conf = 0.5,
    mask_padding = 32,
  } = opts;
  let res;
  try {
    res = await fetch(API_BASE + '/api/adetailer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image, prompt, negative_prompt, denoising_strength, conf, mask_padding,
      }),
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 422) {
    throw new Error(
      typeof data.detail === 'string' ? data.detail : t('api.badAdetailer')
    );
  }
  if (res.status === 503) {
    throw new Error(
      typeof data.detail === 'string' && data.detail
        ? data.detail
        : t('api.needGpuAdetailer')
    );
  }
  if (res.status === 404) throw new Error(t('api.notFound'));
  if (res.status === 413) throw new Error(t('api.tooLarge'));
  if (!res.ok) throw new Error(data.detail || t('api.adetailerFail', { status: res.status }));
  if (!data.url) throw new Error(t('api.adetailerNoResult'));
  return { url: API_BASE + data.url, faces_detected: data.faces_detected ?? 0 };
}

/** POST /api/outpaint — 아웃페인팅 (캔버스 확장 + 새 영역 인페인트).
 *  image: 결과 파일명(output dir 안) 또는 base64 이미지.
 *  opts: {expand: {up, down, left, right} (각 64~512px),
 *         prompt, negative_prompt, denoise_strength(기본 0.6), mask_blur(기본 16)}.
 *  성공 시 완성된 이미지의 절대 URL 반환. */
export async function requestOutpaint(image, opts = {}) {
  const {
    expand = { up: 256, down: 256, left: 256, right: 256 },
    prompt = '',
    negative_prompt = '',
    denoise_strength = 0.6,
    mask_blur = 16,
  } = opts;
  let res;
  try {
    res = await fetch(API_BASE + '/api/outpaint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image, expand, prompt, negative_prompt, denoise_strength, mask_blur,
      }),
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 422) {
    throw new Error(
      typeof data.detail === 'string' ? data.detail : t('api.badOutpaint')
    );
  }
  if (res.status === 503) {
    throw new Error(
      typeof data.detail === 'string' && data.detail
        ? data.detail
        : t('api.needGpuOutpaint')
    );
  }
  if (res.status === 404) throw new Error(t('api.notFound'));
  if (res.status === 413) throw new Error(t('api.tooLarge'));
  if (!res.ok) throw new Error(data.detail || t('api.outpaintFail', { status: res.status }));
  if (!data.url) throw new Error(t('api.outpaintNoResult'));
  return API_BASE + data.url;
}

/** POST /api/xy-plot — X/Y 플롯 시작.
 *  xAxis: {type: 'sampler'|'cfg_scale'|'steps'|'seed', values: []}.
 *  yAxis: 같은 형식 또는 null (없음).
 *  baseParams: QuickPanel 공통 설정. seedRule: 'fixed' (기본, 비교용) | 'increment'.
 *  성공 시 {job_id, status} 반환. */
export async function requestXyPlot(xAxis, yAxis, baseParams, seedRule = 'fixed') {
  const { loras, upscale_strength, ...rest } = baseParams;
  const body = {
    x_axis: xAxis,
    y_axis: yAxis,
    base_params: rest,
    seed_rule: seedRule,
  };
  if (loras) body.loras = loras;
  let res;
  try {
    res = await fetch(API_BASE + '/api/xy-plot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  const detail = typeof data.detail === 'string' ? data.detail
    : Array.isArray(data.detail) ? data.detail.join(', ')
    : null;
  if (res.status === 422) {
    throw new Error(detail || t('api.badXy'));
  }
  if (res.status === 503) {
    throw new Error(detail || t('api.noModel'));
  }
  if (res.status === 409) {
    throw new Error(detail || t('api.xyBusy'));
  }
  if (!res.ok) throw new Error(detail || t('api.xyRequestFail', { status: res.status }));
  if (!data.job_id) throw new Error(t('api.xyNoJobId'));
  return data;
}

/** GET /api/xy-plot/{job_id} — X/Y 플롯 진행 상태 조회. */
export async function getXyPlotJob(jobId) {
  let res;
  try {
    res = await fetch(API_BASE + '/api/xy-plot/' + encodeURIComponent(jobId));
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 404) throw new Error(t('api.xyNotFound'));
  if (!res.ok) throw new Error(data.detail || t('api.statusFail', { status: res.status }));
  return data;
}

/** DELETE /api/xy-plot/{job_id} — X/Y 플롯 취소 요청. */
export async function cancelXyPlotJob(jobId) {
  let res;
  try {
    res = await fetch(API_BASE + '/api/xy-plot/' + encodeURIComponent(jobId), {
      method: 'DELETE',
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 404) throw new Error(t('api.xyNotFound'));
  if (!res.ok) throw new Error(data.detail || t('api.cancelFail', { status: res.status }));
  return data;
}

/** POST /api/batch — 배치 생성 큐 시작.
 *  prompts: 프롬프트 배열 (한 줄에 하나).
 *  params: QuickPanel 공통 설정. loras·upscale_strength는 분리해 별도 전송.
 *  성공 시 {job_id, status} 반환. */
export async function requestBatch(prompts, params) {
  const { loras, upscale_strength, ...rest } = params;
  const body = { prompts, params: rest };
  if (loras) body.loras = loras;
  let res;
  try {
    res = await fetch(API_BASE + '/api/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  const detail = typeof data.detail === 'string' ? data.detail
    : Array.isArray(data.detail) ? data.detail.join(', ')
    : null;
  if (res.status === 422) {
    throw new Error(detail || t('api.badBatch'));
  }
  if (res.status === 503) {
    throw new Error(detail || t('api.noModel'));
  }
  if (res.status === 409) {
    throw new Error(detail || t('api.xyBusy'));
  }
  if (!res.ok) throw new Error(detail || t('api.batchRequestFail', { status: res.status }));
  if (!data.job_id) throw new Error(t('api.batchNoJobId'));
  return data;
}

/** GET /api/batch/{job_id} — 배치 진행 상태 조회. */
export async function getBatchJob(jobId) {
  let res;
  try {
    res = await fetch(API_BASE + '/api/batch/' + encodeURIComponent(jobId));
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 404) throw new Error(t('api.batchNotFound'));
  if (!res.ok) throw new Error(data.detail || t('api.statusFail', { status: res.status }));
  return data;
}

/** DELETE /api/batch/{job_id} — 배치 취소 요청. */
export async function cancelBatchJob(jobId) {
  let res;
  try {
    res = await fetch(API_BASE + '/api/batch/' + encodeURIComponent(jobId), {
      method: 'DELETE',
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 404) throw new Error(t('api.batchNotFound'));
  if (!res.ok) throw new Error(data.detail || t('api.cancelFail', { status: res.status }));
  return data;
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

/** POST /api/png-info — PNG에서 생성 설정 읽기. {found, params} 반환. */
export async function requestPngInfo(file) {
  const form = new FormData();
  form.append('file', file);
  let res;
  try {
    res = await fetch(API_BASE + '/api/png-info', { method: 'POST', body: form });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/** GET /api/prompt-favorites — 실패 시 null */
export async function fetchFavorites() {
  try {
    const data = await getJson('/api/prompt-favorites');
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

/** POST /api/prompt-favorites — 즐겨찾기 추가 */
export async function addFavorite(name, prompt) {
  const res = await fetch(API_BASE + '/api/prompt-favorites', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, prompt }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/** DELETE /api/prompt-favorites/{id} — 즐겨찾기 삭제 */
export async function deleteFavorite(id) {
  const res = await fetch(
    API_BASE + '/api/prompt-favorites/' + encodeURIComponent(id),
    { method: 'DELETE' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || t('api.deleteFail'));
  }
  return await res.json();
}

/** GET /api/negative-presets — 내장+사용자 프리셋. 실패 시 null */
export async function fetchNegativePresets() {
  try {
    const data = await getJson('/api/negative-presets');
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

/** POST /api/negative-presets — 사용자 프리셋 추가 */
export async function addNegativePreset(name, text) {
  const res = await fetch(API_BASE + '/api/negative-presets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, text }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/** DELETE /api/negative-presets/{id} — 사용자 프리셋 삭제 */
export async function deleteNegativePreset(id) {
  const res = await fetch(
    API_BASE + '/api/negative-presets/' + encodeURIComponent(id),
    { method: 'DELETE' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || t('api.deleteFail'));
  }
  return await res.json();
}

/** GET /api/sampler-aliases — 샘플러 별칭 매핑. 실패 시 null. 결과는 캐시. */
let _samplerAliases = null;
let _samplerKeys = null;
let _samplerAliasesFailed = false;
export async function fetchSamplerAliases() {
  if (_samplerAliases || _samplerAliasesFailed) {
    return { aliases: _samplerAliases, keys: _samplerKeys };
  }
  try {
    const data = await getJson('/api/sampler-aliases');
    _samplerAliases = data?.aliases && typeof data.aliases === 'object' ? data.aliases : null;
    _samplerKeys = Array.isArray(data?.keys) ? data.keys : null;
  } catch {
    _samplerAliasesFailed = true;
  }
  return { aliases: _samplerAliases, keys: _samplerKeys };
}

/**
 * 샘플러 표시 이름/별칭 → 내부 키.
 * 백엔드 normalize_sampler_name과 같은 규칙(소문자 변환 + 공백 제거).
 * 찾지 못하면 null.
 */
export function resolveSamplerKey(name, aliases, keys) {
  if (typeof name !== 'string' || !name.trim()) return null;
  const norm = name.toLowerCase().replace(/\s+/g, '');
  const normOf = (s) => String(s).toLowerCase().replace(/\s+/g, '');
  if (Array.isArray(keys)) {
    const hit = keys.find((k) => normOf(k) === norm);
    if (hit) return hit;
  }
  if (aliases && typeof aliases === 'object') {
    for (const [alias, key] of Object.entries(aliases)) {
      if (normOf(alias) === norm) return key;
    }
  }
  return null;
}

/** GET /api/samplers — 샘플러별 스케줄러 정보. 실패 시 null. 결과는 캐시. */
let _samplersInfo = null;
let _samplersInfoFailed = false;
export async function fetchSamplers() {
  if (_samplersInfo || _samplersInfoFailed) return _samplersInfo;
  try {
    const data = await getJson('/api/samplers');
    _samplersInfo = data && typeof data === 'object' ? data : null;
  } catch {
    _samplersInfoFailed = true;
  }
  return _samplersInfo;
}

/** POST /api/downloads — URL 다운로드 시작. {job_id} 반환. */
export async function startDownload(url, kind = 'auto', packDir = null, apiKey = '') {
  const body = { url, kind };
  if (packDir) body.pack_dir = packDir;
  if (apiKey) body.civitai_api_key = apiKey;
  let res;
  try {
    res = await fetch(API_BASE + '/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(t('api.connectFail'));
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/** GET /api/downloads/{id} — 진행 상태. 실패 시 null */
export async function fetchDownloadStatus(jobId) {
  try {
    return await getJson('/api/downloads/' + encodeURIComponent(jobId));
  } catch {
    return null;
  }
}
