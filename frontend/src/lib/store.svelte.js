// 설정 메타데이터 + 현재값 공유 스토어 (Svelte 5 runes)
import { fetchSettingsMeta } from './api.js';
import { FALLBACK_SETTINGS } from './fallbackSettings.js';

/** 설정 목록으로부터 기본값 딕셔너리 생성 */
function defaultsFrom(list) {
  const values = {};
  for (const s of list) values[s.key] = s.recommended;
  // LoRA 선택 상태 (settings 검증 대상이 아님). {이름: {enabled, strength}}
  if (!values.loras) values.loras = {};
  return values;
}

// 폴백으로 동기 초기화: 컴포넌트 init 시점에 store.values.* 가
// undefined가 되면 Svelte 5의 bind: 가 props_invalid_value 에러를 낸다.
// (onMount → initStore() 는 init보다 늦게 실행되므로 여기서 미리 채운다.)
export const store = $state({
  meta: FALLBACK_SETTINGS,
  values: defaultsFrom(FALLBACK_SETTINGS),
  apiOnline: false,
  loaded: false,
});

/** /api/settings를 불러오고, 성공하면 원격 값으로 교체 */
export async function initStore() {
  const remote = await fetchSettingsMeta();
  if (remote) {
    store.meta = remote;
    store.apiOnline = true;
    const values = defaultsFrom(remote);
    // 사용자가 이미 건드린 LoRA 선택은 유지
    if (store.values.loras) values.loras = store.values.loras;
    store.values = values;
  }
  store.loaded = true;
}

export function resetAll() {
  for (const s of store.meta) store.values[s.key] = s.recommended;
}

export function resetOne(key) {
  const s = store.meta.find((m) => m.key === key);
  if (s) store.values[key] = s.recommended;
}
