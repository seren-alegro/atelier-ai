// api.js 테스트 — fetch를 모킹해 요청 body 구성과 에러 메시지 변환을 검증.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { requestGenerate, requestAdetailer, requestBatch, getBatchJob, cancelBatchJob } from './api.js';

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

describe('requestAdetailer', () => {
  it('성공하면 {url, faces_detected}를 반환한다', async () => {
    mockFetchOnce(200, {
      url: '/api/results/20261015-inpaint.png',
      faces_detected: 2,
    });
    const r = await requestAdetailer('a.png', {
      prompt: '1girl',
      negative_prompt: 'lowres',
      denoising_strength: 0.4,
    });
    expect(r).toEqual({
      url: API + '/api/results/20261015-inpaint.png',
      faces_detected: 2,
    });
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.image).toBe('a.png');
    expect(sentBody.prompt).toBe('1girl');
    expect(sentBody.negative_prompt).toBe('lowres');
    expect(sentBody.denoising_strength).toBe(0.4);
  });

  it('기본값을 채워 보낸다', async () => {
    mockFetchOnce(200, { url: '/r/1.png', faces_detected: 0 });
    await requestAdetailer('a.png');
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.denoising_strength).toBe(0.4);
    expect(sentBody.conf).toBe(0.5);
    expect(sentBody.mask_padding).toBe(32);
  });

  it('faces_detected가 없으면 0으로 둔다', async () => {
    mockFetchOnce(200, { url: '/r/1.png' });
    const r = await requestAdetailer('a.png');
    expect(r.faces_detected).toBe(0);
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '디노이즈 강도는 0~1 사이여야 해요.' });
    await expect(requestAdetailer('a.png', { denoising_strength: 2 })).rejects.toThrow(
      '디노이즈 강도는 0~1 사이여야 해요.'
    );
  });

  it('503이면 백엔드의 한국어 detail을 그대로 던진다 (GPU 없음)', async () => {
    mockFetchOnce(503, { detail: 'GPU를 찾을 수 없습니다.' });
    await expect(requestAdetailer('a.png')).rejects.toThrow('GPU를 찾을 수 없습니다.');
  });

  it('503에 detail이 없으면 GPU 안내를 던진다', async () => {
    mockFetchOnce(503, {});
    await expect(requestAdetailer('a.png')).rejects.toThrow('GPU가 필요해요');
  });

  it('네트워크 실패면 한국어 메시지를 던진다', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('boom'));
    await expect(requestAdetailer('a.png')).rejects.toThrow('백엔드에 연결할 수 없어요');
  });
});

describe('requestBatch', () => {
  it('loras를 params에서 분리해 별도 필드로 보낸다', async () => {
    mockFetchOnce(202, { job_id: 'b1', status: 'queued' });
    const r = await requestBatch(['1girl', '2girls'], {
      prompt: 'base',
      loras: { 'lora/a.safetensors': { enabled: true, strength: 0.8 } },
      upscale_strength: 0.7,
    });
    expect(r.job_id).toBe('b1');
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.prompts).toEqual(['1girl', '2girls']);
    expect(sentBody.params.loras).toBeUndefined();
    expect(sentBody.params.upscale_strength).toBeUndefined();
    expect(sentBody.loras).toEqual({
      'lora/a.safetensors': { enabled: true, strength: 0.8 },
    });
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '프롬프트가 비어 있어요.' });
    await expect(requestBatch([], {})).rejects.toThrow('프롬프트가 비어 있어요.');
  });

  it('409면 다른 배치 실행 중 메시지를 던진다', async () => {
    mockFetchOnce(409, { detail: '다른 배치 작업이 실행 중이에요.' });
    await expect(requestBatch(['a'], {})).rejects.toThrow('다른 배치 작업이 실행 중이에요.');
  });

  it('503이면 백엔드의 detail을 던진다', async () => {
    mockFetchOnce(503, { detail: '모델이 설정되지 않았어요.' });
    await expect(requestBatch(['a'], {})).rejects.toThrow('모델이 설정되지 않았어요.');
  });

  it('job_id가 없으면 에러를 던진다', async () => {
    mockFetchOnce(202, {});
    await expect(requestBatch(['a'], {})).rejects.toThrow('배치 작업 ID를 받지 못했어요.');
  });

  it('네트워크 실패면 한국어 메시지를 던진다', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('boom'));
    await expect(requestBatch(['a'], {})).rejects.toThrow('백엔드에 연결할 수 없어요');
  });
});

describe('getBatchJob / cancelBatchJob', () => {
  it('상태를 그대로 반환한다', async () => {
    const body = { job_id: 'b1', status: 'running', total: 2, done: 1, items: [] };
    mockFetchOnce(200, body);
    const r = await getBatchJob('b1');
    expect(r).toEqual(body);
    expect(globalThis.fetch.mock.calls[0][0]).toBe(API + '/api/batch/b1');
  });

  it('없는 작업이면 404 메시지를 던진다', async () => {
    mockFetchOnce(404, { detail: '배치 작업을 찾을 수 없어요.' });
    await expect(getBatchJob('nope')).rejects.toThrow('배치 작업을 찾을 수 없어요.');
  });

  it('취소는 DELETE로 요청한다', async () => {
    const body = { job_id: 'b1', status: 'cancelled', total: 2, done: 1, items: [] };
    mockFetchOnce(200, body);
    const r = await cancelBatchJob('b1');
    expect(r.status).toBe('cancelled');
    const [url, opts] = globalThis.fetch.mock.calls[0];
    expect(url).toBe(API + '/api/batch/b1');
    expect(opts.method).toBe('DELETE');
  });
});
