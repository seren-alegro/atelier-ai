// 모델 목록 공유 스토어 (QuickPanel의 모델 선택 + ModelsTab에서 함께 사용)
import { fetchModels, setActiveModel, deleteModel, refreshModels, fetchTextEncoder, setTextEncoder, fetchVae, setVae } from './api.js';
import { t } from './i18n.svelte.js';

export const modelStore = $state({
  data: null, // {models_dir, active, models: [{name, kind, size_mb, ...}]}
  loading: false,
  notice: '',
  textEncoder: null, // 선택된 텍스트 인코더 name (없으면 null)
  vae: null, // 선택된 VAE name (없으면 null)
});

/** GET /api/models — 목록 불러오기 */
export async function loadModels() {
  modelStore.loading = true;
  modelStore.notice = '';
  const res = await fetchModels();
  modelStore.loading = false;
  if (!res) {
    modelStore.notice = t('mdl.offline');
    modelStore.data = null;
    return;
  }
  modelStore.data = res;
  // 선택된 텍스트 인코더·VAE도 함께 조회
  try {
    const te = await fetchTextEncoder();
    modelStore.textEncoder = te?.name || null;
  } catch {
    modelStore.textEncoder = null;
  }
  try {
    const v = await fetchVae();
    modelStore.vae = v?.name || null;
  } catch {
    modelStore.vae = null;
  }
}

/** POST /api/models/refresh — 디렉토리 재스캔 */
export async function rescanModels() {
  modelStore.loading = true;
  modelStore.notice = '';
  const res = await refreshModels();
  modelStore.loading = false;
  if (res) {
    modelStore.data = res;
  } else {
    modelStore.notice = t('mdl.refreshFail');
  }
}

/** POST /api/models/active — 활성 체크포인트 전환 */
export async function activateModel(name) {
  modelStore.loading = true;
  modelStore.notice = '';
  try {
    const res = await setActiveModel(name);
    if (modelStore.data) {
      modelStore.data = { ...modelStore.data, active: res.active };
    }
  } catch (e) {
    modelStore.notice = e.message || t('mdl.switchFail');
  } finally {
    modelStore.loading = false;
  }
}

/** DELETE /api/models/{name} — 모델 삭제 후 목록 갱신 */
export async function removeModel(name) {
  modelStore.loading = true;
  modelStore.notice = '';
  try {
    await deleteModel(name);
    const res = await fetchModels();
    modelStore.data = res;
    if (!res) modelStore.notice = t('mdl.reloadFail');
  } catch (e) {
    modelStore.notice = e.message || t('mdl.deleteFail');
  } finally {
    modelStore.loading = false;
  }
}

/** 체크포인트만 (단일 파일) */
export function checkpoints() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'checkpoint');
}

/** Anima 3.8b 팩 (디렉토리 단위) */
export function anima38packs() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'anima38pack');
}

/** LoRA만 */
export function loras() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'lora');
}

/** ControlNet만 */
export function controlnets() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'controlnet');
}

/** VAE만 */
export function vaes() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'vae');
}

/** 텍스트 인코더만 */
export function textEncoders() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'text_encoder');
}

/** 텍스트 인코더 선택 (""이면 해제) */
export async function selectTextEncoder(name) {
  modelStore.loading = true;
  modelStore.notice = '';
  try {
    const res = await setTextEncoder(name || '');
    modelStore.textEncoder = res.name || null;
  } catch (e) {
    modelStore.notice = e.message || t('mdl.switchFail');
  } finally {
    modelStore.loading = false;
  }
}

/** VAE 선택 (""이면 해제) */
export async function selectVae(name) {
  modelStore.loading = true;
  modelStore.notice = '';
  try {
    const res = await setVae(name || '');
    modelStore.vae = res.name || null;
  } catch (e) {
    modelStore.notice = e.message || t('mdl.switchFail');
  } finally {
    modelStore.loading = false;
  }
}
