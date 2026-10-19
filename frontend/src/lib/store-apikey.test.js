// store.svelte.js의 API 키 저장 헬퍼 테스트 (localStorage 스텁)
import { describe, it, expect, beforeEach, vi } from 'vitest';

function makeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
}

describe('civitai api key persistence', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', makeStorage());
  });

  it('saveApiKey → loadSavedApiKey 라운드트립', async () => {
    const { saveApiKey, loadSavedApiKey, CIVITAI_KEY_STORAGE } = await import(
      './store.svelte.js'
    );
    expect(loadSavedApiKey()).toBe('');
    saveApiKey('key123');
    expect(loadSavedApiKey()).toBe('key123');
    expect(localStorage.getItem(CIVITAI_KEY_STORAGE)).toBe('key123');
  });

  it('빈 키를 저장하면 삭제된다', async () => {
    const { saveApiKey, loadSavedApiKey } = await import('./store.svelte.js');
    saveApiKey('key123');
    saveApiKey('');
    expect(loadSavedApiKey()).toBe('');
  });
});
