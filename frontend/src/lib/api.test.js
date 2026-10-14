// api.js 테스트 — fetch를 모킹해 요청 body 구성과 에러 메시지 변환을 검증.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { requestGenerate } from './api.js';

const API = 'http://127.0.0.1:8079';

function mockFetchOnce(status, body) {
  const json = vi.fn().mockResolvedValue(body);
  globalThis.fetch = vi.fn().mockResolvedValue({ status, ok: status < 400, json });
  return { json };
}

/** POST는 202, 폴링은 statusFn이 정한 상태를 반환 */
function mockPolling(statusFn) {
  globalThis.fetch = vi.fn().mockImplementation((url) => {
    if (url === API + '/api/generate') {
      return Promise.resolve({
        status: 202,
        ok: true,
        json: () => Promise.resolve({ job_id: 'abc' }),
      });
    }
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve(statusFn()),
    });
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('requestGenerate', () => {
  it('loras를 params에서 분리해 별도 필드로 보낸다', async () => {
    mockPolling(() => ({ status: 'done', results: ['/r/1.png'] }));

    const params = {
      prompt: '1girl',
      loras: { 'lora/a.safetensors': { enabled: true, strength: 0.7 } },
    };
    const results = await requestGenerate(params, 'txt2img');
    expect(results).toEqual(['http://127.0.0.1:8079/r/1.png']);

    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.mode).toBe('txt2img');
    expect(sentBody.params.loras).toBeUndefined();
    expect(sentBody.params.prompt).toBe('1girl');
    expect(sentBody.loras).toEqual({
      'lora/a.safetensors': { enabled: true, strength: 0.7 },
    });
  });

  it('loras가 없으면 loras 필드를 보내지 않는다', async () => {
    mockPolling(() => ({ status: 'done', results: [] }));

    await requestGenerate({ prompt: '1girl' });
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect('loras' in sentBody).toBe(false);
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '스텝은 1 이상이어야 해요.' });
    await expect(requestGenerate({ steps: 0 })).rejects.toThrow(
      '스텝은 1 이상이어야 해요.'
    );
  });

  it('503이면 백엔드의 한국어 detail을 그대로 던진다 (GPU 없음)', async () => {
    mockFetchOnce(503, {
      detail:
        'GPU를 찾을 수 없습니다. Atelier AI는 NVIDIA GPU(CUDA) 또는 Apple Silicon(Metal/MPS)이 필요합니다.',
    });
    await expect(requestGenerate({ prompt: 'x' })).rejects.toThrow(
      'GPU를 찾을 수 없습니다.'
    );
  });

  it('503에 detail이 없으면 모델 미설정 안내를 던진다', async () => {
    mockFetchOnce(503, {});
    await expect(requestGenerate({ prompt: 'x' })).rejects.toThrow(
      '모델이 로드되지 않았어요'
    );
  });

  it('네트워크 실패면 한국어 메시지를 던진다', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('boom'));
    await expect(requestGenerate({ prompt: 'x' })).rejects.toThrow(
      '백엔드에 연결할 수 없어요'
    );
  });

  it('취소하면 폴링이 멈춘다', async () => {
    mockPolling(() => ({ status: 'running' }));

    const p = requestGenerate({ prompt: 'x' });
    p.cancel();
    await expect(p).rejects.toThrow('취소');
  });
});
