// 작업내역 공유 스토어 (작업내역 탭 + 라이브러리 탭에서 함께 사용)
import {
  fetchHistory,
  deleteHistory,
  toggleHistoryFavorite,
} from './api.js';

export const historyStore = $state({
  items: [], // [{id, timestamp, mode, model, params, results, favorite}]
  loading: false,
  notice: '',
  offline: false,
});

/** GET /api/history — 목록 불러오기 */
export async function loadHistory(limit = 100) {
  historyStore.loading = true;
  historyStore.notice = '';
  const items = await fetchHistory(limit);
  historyStore.loading = false;
  if (!items) {
    historyStore.offline = true;
    historyStore.notice = '백엔드에 연결되지 않았어요. 백엔드를 먼저 실행해주세요.';
    return;
  }
  historyStore.offline = false;
  historyStore.items = items;
}

/** DELETE /api/history/{id} — 기록 삭제 (결과 파일은 유지) */
export async function removeHistory(id) {
  historyStore.notice = '';
  try {
    await deleteHistory(id);
    historyStore.items = historyStore.items.filter((r) => r.id !== id);
  } catch (e) {
    historyStore.notice = e.message || '삭제에 실패했어요.';
  }
}

/** POST /api/history/{id}/favorite — 즐겨찾기 토글 */
export async function toggleFavorite(id) {
  historyStore.notice = '';
  try {
    const res = await toggleHistoryFavorite(id);
    const item = historyStore.items.find((r) => r.id === id);
    if (item) item.favorite = res.favorite;
  } catch (e) {
    historyStore.notice = e.message || '즐겨찾기 변경에 실패했어요.';
  }
}

/** 즐겨찾기만 */
export function favorites() {
  return historyStore.items.filter((r) => r.favorite);
}
