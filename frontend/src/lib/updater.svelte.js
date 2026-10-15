// 앱 자동 업데이트 (Tauri updater 플러그인).
// GitHub Releases의 latest.json을 보고 새 버전을 확인·설치한다.
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { invoke } from '@tauri-apps/api/core';
import { getVersion } from '@tauri-apps/api/app';
import { t } from './i18n.svelte.js';

export const updateState = $state({
  current: '',
  availableVersion: '',
  status: 'idle', // idle | checking | latest | available | downloading | error
  message: '',
  progress: 0, // 0~100
  downloaded: 0,
  total: 0,
});

export function isTauri() {
  return (
    typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
  );
}

export async function loadCurrentVersion() {
  if (!isTauri()) return;
  try {
    updateState.current = await getVersion();
  } catch {
    /* 무시 */
  }
}

/** 새 버전 확인. silent=true면 새 버전이 있을 때만 상태를 바꾼다. */
export async function checkForUpdates({ silent = false } = {}) {
  if (!isTauri()) {
    if (!silent) updateState.message = t('upd.notInAppMsg');
    return;
  }
  if (updateState.status === 'checking' || updateState.status === 'downloading')
    return;
  updateState.status = 'checking';
  updateState.progress = 0;
  if (!silent) updateState.message = t('upd.checkingMsg');
  try {
    const update = await check();
    if (!update) {
      updateState.status = 'latest';
      updateState.availableVersion = '';
      if (!silent) updateState.message = t('upd.latestMsg');
      return;
    }
    updateState.status = 'available';
    updateState.availableVersion = update.version;
    updateState.message = t('upd.foundMsg', { v: update.version });
  } catch (e) {
    updateState.status = 'error';
    if (!silent)
      updateState.message = t('upd.failMsg', { e: e?.message || e });
  }
}

/** 다운로드 후 설치하고 재시작. Windows 파일 잠금을 피하려고 백엔드를 먼저 끈다. */
export async function downloadAndInstall() {
  if (updateState.status === 'downloading') return;
  updateState.status = 'checking';
  updateState.message = t('upd.fetchingMsg');
  try {
    const update = await check();
    if (!update) {
      updateState.status = 'latest';
      updateState.message = t('upd.latestMsg');
      return;
    }
    updateState.status = 'downloading';
    updateState.progress = 0;
    updateState.downloaded = 0;
    let total = 0;
    await update.downloadAndInstall((event) => {
      if (event.event === 'Started') {
        total = event.data.contentLength || 0;
        updateState.total = total;
        updateState.message = '다운로드 중…';
      } else if (event.event === 'Progress') {
        updateState.downloaded += event.data.chunkLength;
        if (total > 0)
          updateState.progress = Math.round(
            (updateState.downloaded / total) * 100
          );
      } else if (event.event === 'Finished') {
        updateState.progress = 100;
        updateState.message = '설치하고 다시 시작하는 중…';
      }
    });
    // 설치 전 백엔드 종료 (Windows 파일 잠금 방지)
    try {
      await invoke('stop_backend');
    } catch {
      /* 이미 꺼져 있으면 무시 */
    }
    await relaunch();
  } catch (e) {
    updateState.status = 'error';
    updateState.message = `업데이트에 실패했어요: ${e?.message || e}`;
  }
}
