// 설정 메타데이터 + 현재값 공유 스토어 (Svelte 5 runes)
import { fetchSettingsMeta } from './api.js';
import { FALLBACK_SETTINGS } from './fallbackSettings.js';

/** Civitai API 키 localStorage 보관 키 (민감값: 설정 메타와 분리 보관) */
export const CIVITAI_KEY_STORAGE = 'atelier.civitai_api_key';

/** 저장된 Civitai API 키 읽기 (없으면 '') */
export function loadSavedApiKey() {
  try {
    return localStorage.getItem(CIVITAI_KEY_STORAGE) || '';
  } catch {
    return '';
  }
}

/** Civitai API 키 저장 (비우면 삭제) */
export function saveApiKey(key) {
  try {
    if (key) localStorage.setItem(CIVITAI_KEY_STORAGE, key);
    else localStorage.removeItem(CIVITAI_KEY_STORAGE);
  } catch {
    /* 저장소 접근 실패 시 무시 */
  }
}

/** 설정 목록으로부터 기본값 딕셔너리 생성 */
function defaultsFrom(list) {
  const values = {};
  for (const s of list) values[s.key] = s.recommended;
  // LoRA 선택 상태 (settings 검증 대상이 아님). {이름: {enabled, strength}}
  if (!values.loras) values.loras = {};
  // 업스케일 강도 (QuickPanel 슬라이더, /api/upscale 요청에 사용)
  if (values.upscale_strength === undefined) values.upscale_strength = 0.5;
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

/** /api/settings를 불러오고, 성공하면 원격 값으로 교체.
 *  백엔드는 torch 임포트 때문에 시작이 수십 초~수 분 걸릴 수 있어서,
 *  최대 3분간 2초 간격으로 재시도한다.
 *  reconnect()로 새 루프가 시작되면 이전 루프는 epoch로 취소된다. */
let epoch = 0;

export async function initStore() {
  const myEpoch = ++epoch;
  const before = { ...store.values }; // 대기 중 사용자 입력을 기억
  const remote = await fetchWithRetry(90, 2000, myEpoch);
  if (myEpoch !== epoch) return; // 새 루프가 시작됨 → 이 루프는 중단
  if (remote) {
    store.meta = remote;
    store.apiOnline = true;
    const fresh = defaultsFrom(remote);
    // 사용자가 이미 건드린 LoRA 선택·업스케일 강도는 유지
    if (store.values.loras) fresh.loras = store.values.loras;
    if (store.values.upscale_strength !== undefined) {
      fresh.upscale_strength = store.values.upscale_strength;
    }
    // 연결 대기 중에 사용자가 바꾼 값은 덮어쓰지 않는다
    for (const k of Object.keys(fresh)) {
      if (k in before && store.values[k] !== before[k]) {
        fresh[k] = store.values[k];
      }
    }
    store.values = fresh;
    // 저장된 Civitai API 키 복원 (localStorage에만 보관)
    const savedKey = loadSavedApiKey();
    if (savedKey) store.values.civitai_api_key = savedKey;
  }
  store.loaded = true;
}

async function fetchWithRetry(tries, intervalMs, myEpoch) {
  for (let i = 0; i < tries; i++) {
    if (myEpoch !== epoch) return null;
    const remote = await fetchSettingsMeta();
    if (remote) return remote;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return null;
}

/** 헤더 배지 클릭 시 수동 재연결 */
export async function reconnect() {
  store.loaded = false;
  store.apiOnline = false;
  await initStore();
}

/** 백엔드 생존 확인 (주기적 체크용). 설정값은 건드리지 않고 배지만 갱신. */
export async function checkBackendAlive() {
  const remote = await fetchSettingsMeta();
  const alive = !!remote;
  if (alive && !store.apiOnline) {
    store.meta = remote; // 끊겼다가 살아나면 메타만 갱신
  }
  store.apiOnline = alive;
}

export function resetAll() {
  for (const s of store.meta) store.values[s.key] = s.recommended;
}

export function resetOne(key) {
  const s = store.meta.find((m) => m.key === key);
  if (s) store.values[key] = s.recommended;
}
