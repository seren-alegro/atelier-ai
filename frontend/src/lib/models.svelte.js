// 모델 목록 공유 스토어 (QuickPanel의 모델 선택 + ModelsTab에서 함께 사용)
import { fetchModels, setActiveModel, deleteModel, refreshModels } from './api.js';

export const modelStore = $state({
  data: null, // {models_dir, active, models: [{name, kind, size_mb, ...}]}
  loading: false,
  notice: '',
});

/** GET /api/models — 목록 불러오기 */
export async function loadModels() {
  modelStore.loading = true;
  modelStore.notice = '';
  const res = await fetchModels();
  modelStore.loading = false;
  if (!res) {
    modelStore.notice = '백엔드에 연결되지 않았어요. 백엔드를 먼저 실행해주세요.';
    modelStore.data = null;
    return;
  }
  modelStore.data = res;
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
    modelStore.notice = '새로고침에 실패했어요.';
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
    modelStore.notice = e.message || '모델 전환에 실패했어요.';
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
    if (!res) modelStore.notice = '목록을 다시 불러오지 못했어요.';
  } catch (e) {
    modelStore.notice = e.message || '삭제에 실패했어요.';
  } finally {
    modelStore.loading = false;
  }
}

/** 체크포인트만 */
export function checkpoints() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'checkpoint');
}

/** LoRA만 */
export function loras() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'lora');
}

/** ControlNet만 */
export function controlnets() {
  return (modelStore.data?.models ?? []).filter((m) => m.kind === 'controlnet');
}
