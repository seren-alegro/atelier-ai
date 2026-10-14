// 학습 API 테스트 — fetch 모킹으로 요청 구성과 에러 변환 검증.
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  requestTrain,
  getTrainJob,
  requestCaption,
  requestCaptionBatch,
} from './api.js';

const API = 'http://127.0.0.1:8079';

function mockFetchOnce(status, body) {
  const json = vi.fn().mockResolvedValue(body);
  globalThis.fetch = vi.fn().mockResolvedValue({ status, ok: status < 400, json });
  return globalThis.fetch;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('requestTrain', () => {
  it('POST /api/train에 파라미터를 그대로 보낸다', async () => {
    const fetch = mockFetchOnce(202, { job_id: 't1', status: 'queued', total_steps: 1500 });
    const params = {
      model: 'anima.safetensors',
      dataset_dir: '/data/char',
      trigger_word: 'sks girl',
      rank: 16,
      lr: 0.0001,
      steps_per_image: 75,
      resolution: 1024,
      train_text_encoder: false,
      output_name: 'my-lora',
    };
    const r = await requestTrain(params);
    expect(r.job_id).toBe('t1');
    const [url, opts] = fetch.mock.calls[0];
    expect(url).toBe(API + '/api/train');
    expect(JSON.parse(opts.body)).toEqual(params);
  });

  it('422 detail을 한국어 메시지로 던진다', async () => {
    mockFetchOnce(422, { detail: '데이터셋 폴더를 찾을 수 없어요.' });
    await expect(requestTrain({})).rejects.toThrow('데이터셋 폴더를 찾을 수 없어요.');
  });
});

describe('getTrainJob', () => {
  it('진행 상태를 그대로 반환한다', async () => {
    mockFetchOnce(200, {
      job_id: 't1', status: 'running', step: 300, total_steps: 1500, loss: 0.123,
    });
    const j = await getTrainJob('t1');
    expect(j.step).toBe(300);
    expect(j.loss).toBe(0.123);
  });

  it('404이면 에러를 던진다', async () => {
    mockFetchOnce(404, { detail: '학습 작업을 찾을 수 없어요.' });
    await expect(getTrainJob('nope')).rejects.toThrow('학습 작업을 찾을 수 없어요.');
  });
});

describe('requestCaption', () => {
  it('POST /api/caption에 경로와 방식을 보낸다', async () => {
    const fetch = mockFetchOnce(200, { caption: '1girl, solo' });
    const c = await requestCaption('/data/a.png', 'blip');
    expect(c).toBe('1girl, solo');
    const [url, opts] = fetch.mock.calls[0];
    expect(url).toBe(API + '/api/caption');
    expect(JSON.parse(opts.body)).toEqual({ image_path: '/data/a.png', method: 'blip' });
  });
});

describe('requestCaptionBatch', () => {
  it('일괄 결과를 그대로 반환한다', async () => {
    mockFetchOnce(200, { done: ['a.png'], skipped: [], failed: [] });
    const r = await requestCaptionBatch('/data/char', 'wd14');
    expect(r.done).toEqual(['a.png']);
  });
});
