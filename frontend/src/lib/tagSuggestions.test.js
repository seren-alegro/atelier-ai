// tagSuggestions.js 테스트
import { describe, it, expect } from 'vitest';
import { suggestTags, TAG_SUGGESTIONS } from './tagSuggestions.js';

describe('suggestTags', () => {
  it('빈 입력에는 빈 목록', () => {
    expect(suggestTags('')).toEqual([]);
    expect(suggestTags('   ')).toEqual([]);
  });

  it('부분 일치 (대소문자 무시)', () => {
    const r = suggestTags('1GIRL', []);
    expect(r).toContain('1girl');
  });

  it('이미 쓴 태그는 제외', () => {
    const r = suggestTags('girl', ['1girl']);
    expect(r).not.toContain('1girl');
  });

  it('최대 8개', () => {
    const r = suggestTags('a', []);
    expect(r.length).toBeLessThanOrEqual(8);
  });

  it('목록이 비어 있지 않다', () => {
    expect(TAG_SUGGESTIONS.length).toBeGreaterThan(20);
  });
});
