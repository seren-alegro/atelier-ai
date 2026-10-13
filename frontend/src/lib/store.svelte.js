// 설정 메타데이터 + 현재값 공유 스토어 (Svelte 5 runes)
import { fetchSettingsMeta } from './api.js';
import { FALLBACK_SETTINGS } from './fallbackSettings.js';

export const store = $state({
  meta: [],
  values: {},
  apiOnline: false,
  loaded: false,
});

/** /api/settings를 불러오고, 실패하면 폴백으로 초기화 */
export async function initStore() {
  const remote = await fetchSettingsMeta();
  const list = remote ?? FALLBACK_SETTINGS;
  store.meta = list;
  store.apiOnline = remote !== null;
  const values = {};
  for (const s of list) values[s.key] = s.recommended;
  // LoRA 선택 상태 (settings 검증 대상이 아님). {이름: {enabled, strength}}
  if (!values.loras) values.loras = {};
  store.values = values;
  store.loaded = true;
}

export function resetAll() {
  for (const s of store.meta) store.values[s.key] = s.recommended;
}

export function resetOne(key) {
  const s = store.meta.find((m) => m.key === key);
  if (s) store.values[key] = s.recommended;
}
