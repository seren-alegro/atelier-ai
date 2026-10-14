// updater.svelte.js 테스트 — Tauri 밖(브라우저/node)에서는 조용히 스킵.
import { describe, it, expect } from 'vitest';
import {
  updateState,
  checkForUpdates,
  isTauri,
} from './updater.svelte.js';

describe('updater', () => {
  it('Tauri 환경이 아니면 false', () => {
    expect(isTauri()).toBe(false);
  });

  it('Tauri 밖에서는 확인 불가 메시지만 남긴다', async () => {
    updateState.message = '';
    await checkForUpdates();
    expect(updateState.message).toContain('앱에서 실행 중일 때만');
    expect(updateState.status).toBe('idle');
  });

  it('silent 모드에서는 메시지도 남기지 않는다', async () => {
    updateState.message = '';
    await checkForUpdates({ silent: true });
    expect(updateState.message).toBe('');
  });
});
