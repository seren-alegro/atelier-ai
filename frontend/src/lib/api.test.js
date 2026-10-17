// api.js 테스트 — fetch를 모킹해 요청 body 구성과 에러 메시지 변환을 검증.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { requestGenerate, requestAdetailer, requestBatch, getBatchJob, cancelBatchJob, requestXyPlot, getXyPlotJob, cancelXyPlotJob, requestOutpaint, searchTags } from './api.js';

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

describe('requestXyPlot', () => {
  it('축·base_params·seed_rule을 올바른 body로 보낸다', async () => {
    mockFetchOnce(202, { job_id: 'x1', status: 'queued' });
    const r = await requestXyPlot(
      { type: 'cfg_scale', values: [3, 5] },
      { type: 'steps', values: [20, 30] },
      {
        prompt: '1girl',
        loras: { 'lora/a.safetensors': { enabled: true, strength: 0.8 } },
        upscale_strength: 0.7,
      },
      'fixed'
    );
    expect(r.job_id).toBe('x1');
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.x_axis).toEqual({ type: 'cfg_scale', values: [3, 5] });
    expect(sentBody.y_axis).toEqual({ type: 'steps', values: [20, 30] });
    expect(sentBody.seed_rule).toBe('fixed');
    expect(sentBody.base_params.prompt).toBe('1girl');
    expect(sentBody.base_params.loras).toBeUndefined();
    expect(sentBody.base_params.upscale_strength).toBeUndefined();
    expect(sentBody.loras).toEqual({
      'lora/a.safetensors': { enabled: true, strength: 0.8 },
    });
  });

  it('y축 없이도 보낼 수 있다', async () => {
    mockFetchOnce(202, { job_id: 'x2', status: 'queued' });
    await requestXyPlot({ type: 'seed', values: [1, 2] }, null, { prompt: '1girl' });
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.y_axis).toBeNull();
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '한 번에 36칸까지만 가능해요.' });
    await expect(
      requestXyPlot({ type: 'seed', values: [1] }, null, {})
    ).rejects.toThrow('한 번에 36칸까지만 가능해요.');
  });

  it('409면 다른 배치 실행 중 메시지를 던진다', async () => {
    mockFetchOnce(409, { detail: '다른 배치 작업이 실행 중이에요.' });
    await expect(
      requestXyPlot({ type: 'seed', values: [1] }, null, {})
    ).rejects.toThrow('다른 배치 작업이 실행 중이에요.');
  });

  it('job_id가 없으면 에러를 던진다', async () => {
    mockFetchOnce(202, {});
    await expect(
      requestXyPlot({ type: 'seed', values: [1] }, null, {})
    ).rejects.toThrow('X/Y 플롯 작업 ID를 받지 못했어요.');
  });
});

describe('getXyPlotJob / cancelXyPlotJob', () => {
  it('상태를 조회한다', async () => {
    const body = { job_id: 'x1', status: 'running', total: 4, done: 1, items: [] };
    mockFetchOnce(200, body);
    const r = await getXyPlotJob('x1');
    expect(r).toEqual(body);
    expect(globalThis.fetch.mock.calls[0][0]).toBe(API + '/api/xy-plot/x1');
  });

  it('없는 작업이면 404 메시지를 던진다', async () => {
    mockFetchOnce(404, { detail: 'X/Y 플롯 작업을 찾을 수 없어요.' });
    await expect(getXyPlotJob('nope')).rejects.toThrow('X/Y 플롯 작업을 찾을 수 없어요.');
  });

  it('취소는 DELETE로 요청한다', async () => {
    mockFetchOnce(200, { job_id: 'x1', status: 'cancelled' });
    const r = await cancelXyPlotJob('x1');
    expect(r.status).toBe('cancelled');
    const [url, opts] = globalThis.fetch.mock.calls[0];
    expect(url).toBe(API + '/api/xy-plot/x1');
    expect(opts.method).toBe('DELETE');
  });
});

describe('requestOutpaint', () => {
  it('expand·옵션을 올바른 body로 보낸다', async () => {
    mockFetchOnce(200, { url: '/api/results/out.png' });
    const url = await requestOutpaint('orig.png', {
      expand: { up: 64, down: 64, left: 128, right: 128 },
      prompt: 'night sky',
      denoise_strength: 0.6,
      mask_blur: 8,
    });
    expect(url).toBe(API + '/api/results/out.png');
    const sentBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(sentBody.image).toBe('orig.png');
    expect(sentBody.expand).toEqual({ up: 64, down: 64, left: 128, right: 128 });
    expect(sentBody.denoise_strength).toBe(0.6);
    expect(sentBody.mask_blur).toBe(8);
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '확장 값은 64~512px 사이여야 해요.' });
    await expect(requestOutpaint('a.png', {})).rejects.toThrow('확장 값은 64~512px 사이여야 해요.');
  });

  it('503이면 GPU 필요 메시지를 던진다', async () => {
    mockFetchOnce(503, { detail: 'GPU를 찾을 수 없습니다.' });
    await expect(requestOutpaint('a.png', {})).rejects.toThrow('GPU를 찾을 수 없습니다.');
  });

  it('404면 원본 없음 메시지를 던진다', async () => {
    mockFetchOnce(404, {});
    await expect(requestOutpaint('nope.png', {})).rejects.toThrow('원본 이미지를 찾을 수 없어요.');
  });

  it('url이 없으면 에러를 던진다', async () => {
    mockFetchOnce(200, {});
    await expect(requestOutpaint('a.png', {})).rejects.toThrow('아웃페인팅 결과를 받지 못했어요.');
  });
});

describe('searchTags', () => {
  it('q와 limit을 쿼리스트링으로 보낸다', async () => {
    const body = { count: 83687, results: [{ tag: 'blue_eyes', category: 0, count: 4000000, desc_ko: '파란 눈' }] };
    mockFetchOnce(200, body);
    const r = await searchTags('파란눈', 50);
    expect(r).toEqual(body);
    const [url] = globalThis.fetch.mock.calls[0];
    expect(url).toBe(API + '/api/tag-search?q=' + encodeURIComponent('파란눈') + '&limit=50');
  });

  it('422면 백엔드의 한국어 detail을 그대로 던진다', async () => {
    mockFetchOnce(422, { detail: '검색어를 입력해주세요.' });
    await expect(searchTags('')).rejects.toThrow('검색어를 입력해주세요.');
  });

  it('503이면 태그 데이터 오류 메시지를 던진다', async () => {
    mockFetchOnce(503, {});
    await expect(searchTags('hair')).rejects.toThrow('태그 데이터를 불러오지 못했어요. 백엔드를 확인해주세요.');
  });

  it('연결 실패면 한국어 메시지를 던진다', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('boom'));
    await expect(searchTags('hair')).rejects.toThrow('백엔드에 연결할 수 없어요.');
  });
});

describe('startDownload / fetchDownloadStatus', () => {
  it('POST /api/downloads에 url·kind·pack_dir을 담아 보낸다', async () => {
    mockFetchOnce(202, { job_id: 'dl123' });
    const { startDownload } = await import('./api.js');
    const res = await startDownload('https://x.com/m.safetensors', 'lora');
    expect(res.job_id).toBe('dl123');
    const [url, opts] = globalThis.fetch.mock.calls[0];
    expect(url).toBe(API + '/api/downloads');
    const body = JSON.parse(opts.body);
    expect(body.url).toBe('https://x.com/m.safetensors');
    expect(body.kind).toBe('lora');
    expect('pack_dir' in body).toBe(false);
  });

  it('pack이면 pack_dir을 함께 보낸다', async () => {
    mockFetchOnce(202, { job_id: 'dl9' });
    const { startDownload } = await import('./api.js');
    await startDownload('https://x.com/dit.safetensors', 'pack', 'anima38');
    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.kind).toBe('pack');
    expect(body.pack_dir).toBe('anima38');
  });

  it('실패하면 detail 메시지를 던진다', async () => {
    mockFetchOnce(400, { detail: '나쁜 URL' });
    const { startDownload } = await import('./api.js');
    await expect(startDownload('ftp://x', 'auto')).rejects.toThrow('나쁜 URL');
  });

  it('fetchDownloadStatus는 GET /api/downloads/{id}를 호출한다', async () => {
    mockFetchOnce(200, { id: 'dl1', status: 'done' });
    const { fetchDownloadStatus } = await import('./api.js');
    const st = await fetchDownloadStatus('dl1');
    expect(st.status).toBe('done');
    expect(globalThis.fetch.mock.calls[0][0]).toBe(API + '/api/downloads/dl1');
  });

  it('fetchDownloadStatus 실패 시 null', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('down'));
    const { fetchDownloadStatus } = await import('./api.js');
    expect(await fetchDownloadStatus('dl1')).toBeNull();
  });
});
