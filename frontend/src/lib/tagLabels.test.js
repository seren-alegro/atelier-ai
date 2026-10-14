// tagLabels.js 테스트
import { describe, it, expect, vi, afterEach } from 'vitest';
import { ensureTagLabels, labelOf, _setCacheForTest } from './tagLabels.js';

afterEach(() => {
  _setCacheForTest(null);
  vi.unstubAllGlobals();
});

describe('labelOf', () => {
  it('캐시가 없으면 null', () => {
    expect(labelOf('1girl')).toBeNull();
  });

  it('[카테고리] 설명 형식으로 반환', () => {
    _setCacheForTest({ '1girl': ['인물 > 인원수', '여성 캐릭터 한 명'] });
    expect(labelOf('1girl')).toBe('[인물 > 인원수] 여성 캐릭터 한 명');
  });

  it('없는 태그는 null', () => {
    _setCacheForTest({ '1girl': ['인물', '설명'] });
    expect(labelOf('nope')).toBeNull();
  });
});

describe('ensureTagLabels', () => {
  it('API에서 받아 캐시한다', async () => {
    const labels = { solo: ['인물', '설명'] };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ labels, count: 1 }),
    });
    const r1 = await ensureTagLabels();
    expect(r1).toEqual(labels);
    // 두 번째 호출은 fetch 없이 캐시 반환
    const r2 = await ensureTagLabels();
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(r2).toEqual(labels);
  });

  it('실패해도 null (입력은 정상 동작)', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('offline'));
    expect(await ensureTagLabels()).toBeNull();
  });
});
