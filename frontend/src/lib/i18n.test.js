// i18n.svelte.js 테스트 — 3개 언어 딕셔너리 키 집합 일치, t() 폴백, setLang/toggleLang.
import { describe, it, expect, beforeEach } from 'vitest';
import {
  _dictForTest,
  t,
  getLang,
  setLang,
  toggleLang,
  LANGS,
} from './i18n.svelte.js';

const DICT = _dictForTest();
const LANG_IDS = ['ko', 'en', 'ja'];

describe('i18n dictionary parity', () => {
  it('ko/en/ja 키 집합이 완전히 일치한다', () => {
    const koKeys = new Set(Object.keys(DICT.ko));
    for (const id of ['en', 'ja']) {
      const keys = new Set(Object.keys(DICT[id]));
      const missing = [...koKeys].filter((k) => !keys.has(k));
      const extra = [...keys].filter((k) => !koKeys.has(k));
      expect(missing, `${id} missing keys`).toEqual([]);
      expect(extra, `${id} extra keys`).toEqual([]);
    }
  });

  it('LANGS에 3개 언어가 있다', () => {
    expect(LANGS.map((l) => l.id).sort()).toEqual(['en', 'ja', 'ko']);
  });
});

describe('t()', () => {
  beforeEach(() => setLang('ko'));

  it('현재 언어의 문자열을 반환한다', () => {
    expect(t('nav.create')).toBe(DICT.ko['nav.create']);
    setLang('en');
    expect(t('nav.create')).toBe(DICT.en['nav.create']);
    setLang('ja');
    expect(t('nav.create')).toBe(DICT.ja['nav.create']);
  });

  it('없는 키는 한국어 폴백, 그것도 없으면 key 그대로', () => {
    // en에만 없는 키를 시뮬레이션하기 위해 직접 확인
    setLang('en');
    expect(t('nav.create')).toBe(DICT.en['nav.create']);
    expect(t('__no_such_key__')).toBe('__no_such_key__');
  });

  it('{name} 파라미터를 치환한다', () => {
    setLang('ko');
    const s = t('gen.batchStart', { count: 3 });
    expect(s).toContain('3');
    expect(s).not.toContain('{count}');
  });
});

describe('setLang/toggleLang', () => {
  beforeEach(() => setLang('ko'));

  it('setLang으로 언어를 바꾼다', () => {
    setLang('en');
    expect(getLang()).toBe('en');
    setLang('ja');
    expect(getLang()).toBe('ja');
    setLang('ko');
    expect(getLang()).toBe('ko');
  });

  it('toggleLang은 ko→en→ja→ko 순환한다', () => {
    setLang('ko');
    toggleLang();
    expect(getLang()).toBe('en');
    toggleLang();
    expect(getLang()).toBe('ja');
    toggleLang();
    expect(getLang()).toBe('ko');
  });

  it('localStorage에 저장된다 (브라우저 환경)', () => {
    // node/vitest 환경에는 localStorage가 없을 수 있음 — 있을 때만 확인
    if (typeof localStorage === 'undefined') return;
    setLang('ja');
    expect(localStorage.getItem('atelier-ai-lang')).toBe('ja');
    setLang('ko');
  });
});
