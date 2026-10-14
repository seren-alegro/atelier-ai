// 태그 한국어 라벨 (백엔드 /api/tag-labels).
// 첫 호출 시 백엔드가 원본 CSV를 내려받아 파싱하므로 느릴 수 있음.
// 실패해도 태그 입력은 정상 동작한다 (라벨만 없음).

const API_BASE = 'http://127.0.0.1:8079';

let cache = null;   // {tag: [category, description]} | null
let pending = null; // 진행 중 fetch

/** 라벨 dict를 한 번만 받아 메모리에 캐시. 실패 시 null. */
export async function ensureTagLabels() {
  if (cache) return cache;
  if (pending) return pending;
  pending = (async () => {
    try {
      const res = await fetch(API_BASE + '/api/tag-labels');
      if (!res.ok) return null;
      const data = await res.json();
      cache = data.labels ?? null;
      return cache;
    } catch {
      return null;
    } finally {
      pending = null;
    }
  })();
  return pending;
}

/** 태그의 한국어 설명 ("[카테고리] 설명" 형식) 또는 null. */
export function labelOf(tag) {
  if (!cache) return null;
  const entry = cache[tag];
  if (!entry) return null;
  const [cat, desc] = entry;
  if (cat && desc) return `[${cat}] ${desc}`;
  return desc || cat || null;
}

/** 테스트용 캐시 주입 */
export function _setCacheForTest(c) {
  cache = c;
}
