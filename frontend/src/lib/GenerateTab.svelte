<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import { historyStore, loadHistory } from './history.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate, requestUpscale, requestAdetailer, requestBatch, getBatchJob, cancelBatchJob, requestOutpaint, requestXyPlot, getXyPlotJob, cancelXyPlotJob, apiUrl } from './api.js';

  let { onLoad = null } = $props();

  let notice = $state('');
  let busy = $state(false);
  let results = $state([]);
  let currentJob = $state(null);
  let recentLoaded = $state(false);
  let upscaling = $state({}); // src → true (업스케일 진행 중인 이미지)
  let retouching = $state({}); // src → true (얼굴 보정 진행 중인 이미지)
  let outpainting = $state({}); // src → true (아웃페인팅 진행 중인 이미지)
  let outpaintForm = $state(null); // 폼이 열린 이미지 src (없으면 null)
  let outpaintExpand = $state({ up: 256, down: 256, left: 256, right: 256 });

  // 배치 생성 큐
  let batchText = $state('');
  let batchJob = $state(null); // {job_id, status, total, done, items}
  let batchBusy = $state(false);
  let batchNotice = $state('');
  let batchAdded = 0; // 갤러리에 이미 추가한 항목 수 (순서대로)
  let batchTimer = null;

  const batchCount = $derived(
    batchText.split('\n').map((s) => s.trim()).filter(Boolean).length
  );
  const batchStatusLabel = {
    queued: '대기중', running: '생성중…', done: '완료',
    error: '실패', cancelled: '취소됨',
  };

  // X/Y 플롯
  const XY_TYPES = [
    { value: 'sampler', label: '샘플러' },
    { value: 'cfg_scale', label: 'CFG 스케일' },
    { value: 'steps', label: '스텝 수' },
    { value: 'seed', label: '시드' },
  ];
  let xyXType = $state('cfg_scale');
  let xyXValues = $state('3, 5, 7');
  let xyYType = $state('steps');
  let xyYValues = $state('20, 30');
  let xyUseY = $state(false);
  let xyJob = $state(null); // {job_id, status, total, done, items, meta}
  let xyBusy = $state(false);
  let xyNotice = $state('');
  let xyTimer = null;
  let xyAdded = 0;

  function parseXyValues(text, type) {
    return text.split(',').map((s) => s.trim()).filter((s) => s !== '')
      .map((s) => (type === 'sampler' ? s : Number(s)));
  }

  async function onXyStart() {
    const xAxis = { type: xyXType, values: parseXyValues(xyXValues, xyXType) };
    const yAxis = xyUseY ? { type: xyYType, values: parseXyValues(xyYValues, xyYType) } : null;
    if (xAxis.values.length === 0 || (xyUseY && yAxis.values.length === 0)) {
      xyNotice = '축 값을 쉼표로 구분해 하나 이상 입력해주세요.';
      return;
    }
    if (xAxis.values.some((v) => typeof v === 'number' && Number.isNaN(v))
        || (xyUseY && yAxis.values.some((v) => typeof v === 'number' && Number.isNaN(v)))) {
      xyNotice = '숫자 축에는 숫자만 입력해주세요.';
      return;
    }
    xyNotice = '';
    xyBusy = true;
    xyAdded = 0;
    xyJob = null;
    try {
      const { job_id } = await requestXyPlot(xAxis, yAxis, { ...store.values }, 'fixed');
      const poll = async () => {
        try {
          const job = await getXyPlotJob(job_id);
          xyJob = job;
          // 새로 끝난 항목을 갤러리에 순서대로 추가
          while (xyAdded < job.items.length && job.items[xyAdded].status === 'done'
                 && job.items[xyAdded].url) {
            results = [...results, apiUrl(job.items[xyAdded].url)];
            xyAdded += 1;
          }
          if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled') {
            xyBusy = false;
            xyTimer = null;
            return;
          }
        } catch (e) {
          xyNotice = e.message || 'X/Y 플롯 상태를 확인할 수 없어요.';
          xyBusy = false;
          xyTimer = null;
          return;
        }
        xyTimer = setTimeout(poll, 2000);
      };
      poll();
    } catch (e) {
      xyNotice = e.message || 'X/Y 플롯 시작 중 오류가 발생했어요.';
      xyBusy = false;
    }
  }

  async function onXyCancel() {
    if (!xyJob) return;
    try {
      xyJob = await cancelXyPlotJob(xyJob.job_id);
    } catch (e) {
      xyNotice = e.message || 'X/Y 플롯 취소 중 오류가 발생했어요.';
    }
  }

  // X/Y 플롯 그리드용 셀 (row-major: y 바깥, x 안쪽)
  function xyCell(job, yi, xi) {
    const nx = job.meta?.x_labels?.length ?? 0;
    return job.items[yi * nx + xi];
  }

  async function onOutpaint(src) {
    if (outpainting[src]) return;
    notice = '';
    outpainting[src] = true;
    try {
      const name = src.split('/').pop();
      const url = await requestOutpaint(name, {
        expand: { ...outpaintExpand },
        prompt: store.values.prompt ?? '',
        negative_prompt: store.values.negative_prompt ?? '',
      });
      results = [...results, url];
      outpaintForm = null;
    } catch (e) {
      notice = e.message || '아웃페인팅 중 오류가 발생했어요.';
    } finally {
      delete outpainting[src];
    }
  }
  const recents = $derived(
    (historyStore.items ?? [])
      .filter((r) => r.results && r.results.length > 0)
      .slice(0, 8)
  );

  onMount(async () => {
    // 빈 캔버스용 최근 작업 (이미 로드돼 있으면 재사용)
    if (!historyStore.items || historyStore.items.length === 0) {
      await loadHistory(12);
    }
    recentLoaded = true;
  });

  async function onGenerate() {
    notice = '';
    busy = true;
    try {
      const params = { ...store.values };
      currentJob = requestGenerate(params);
      const paths = await currentJob;
      results = paths;
    } catch (e) {
      notice = e.message || '생성 중 오류가 발생했어요.';
    } finally {
      busy = false;
      currentJob = null;
    }
  }

  function onRecentClick(rec) {
    if (onLoad) onLoad(rec);
  }

  async function onUpscale(src) {
    if (upscaling[src]) return;
    notice = '';
    upscaling[src] = true;
    try {
      // 강도는 0.5~1.0 (QuickPanel의 "업스케일 강도" 슬라이더)
      const strength = Math.min(1, Math.max(0.5, store.values.upscale_strength ?? 0.5));
      // 결과 파일명만 전달 (베이스 URL 제거 → 백엔드가 output dir에서 찾음)
      const name = src.split('/').pop();
      const url = await requestUpscale(name, strength);
      results = [...results, url];
    } catch (e) {
      notice = e.message || '업스케일 중 오류가 발생했어요.';
    } finally {
      delete upscaling[src];
    }
  }

  async function onRetouch(src) {
    if (retouching[src]) return;
    notice = '';
    retouching[src] = true;
    try {
      // 원본 생성 프롬프트 재사용 (비어 있으면 백엔드가 폴백 프롬프트 사용)
      const name = src.split('/').pop();
      const { url, faces_detected } = await requestAdetailer(name, {
        prompt: store.values.prompt ?? '',
        negative_prompt: store.values.negative_prompt ?? '',
      });
      if (faces_detected > 0) {
        results = [...results, url];
      } else {
        notice = '얼굴을 찾지 못했어요. 원본 그대로예요.';
      }
    } catch (e) {
      notice = e.message || '얼굴 보정 중 오류가 발생했어요.';
    } finally {
      delete retouching[src];
    }
  }

  async function onBatchStart() {
    const prompts = batchText.split('\n').map((s) => s.trim()).filter(Boolean);
    if (prompts.length === 0) {
      batchNotice = '프롬프트를 한 줄에 하나씩 입력해주세요.';
      return;
    }
    batchNotice = '';
    batchBusy = true;
    batchAdded = 0;
    batchJob = null;
    try {
      const { job_id } = await requestBatch(prompts, { ...store.values });
      batchTimer = setInterval(() => pollBatch(job_id), 1500);
      await pollBatch(job_id);
    } catch (e) {
      batchNotice = e.message || '배치 시작 중 오류가 발생했어요.';
      batchBusy = false;
    }
  }

  async function pollBatch(jobId) {
    try {
      const job = await getBatchJob(jobId);
      batchJob = job;
      // 순서대로 완료된 항목을 갤러리에 추가
      for (let i = batchAdded; i < job.items.length; i++) {
        const it = job.items[i];
        if (it.status === 'done') {
          if (it.url) results = [...results, apiUrl(it.url)];
          batchAdded = i + 1;
        } else if (it.status === 'error' || it.status === 'cancelled') {
          batchAdded = i + 1;
        } else {
          break;
        }
      }
      if (job.status === 'done' || job.status === 'cancelled' || job.status === 'error') {
        if (batchTimer) { clearInterval(batchTimer); batchTimer = null; }
        batchBusy = false;
      }
    } catch (e) {
      batchNotice = e.message || '배치 상태를 확인할 수 없어요.';
      if (batchTimer) { clearInterval(batchTimer); batchTimer = null; }
      batchBusy = false;
    }
  }

  async function onBatchCancel() {
    if (!batchJob) return;
    try {
      batchJob = await cancelBatchJob(batchJob.job_id);
    } catch (e) {
      batchNotice = e.message || '배치 취소 중 오류가 발생했어요.';
    }
  }
</script>

<div class="generate-tab">
  <div class="prompt-card">
    <div class="field-head">
      <span class="field-label">✦ 프롬프트</span>
    </div>
    <TagInput
      bind:value={store.values.prompt}
      placeholder="태그 입력 (예: 1girl, solo) — 입력 중 자동완성이 떠요"
    />

    <details class="neg">
      <summary class="field-label">네거티브 프롬프트</summary>
      <div class="neg-body">
        <TagInput bind:value={store.values.negative_prompt} placeholder="제외할 태그 입력" />
      </div>
    </details>

    <div class="btn-row">
      <button class="btn primary cta" onclick={onGenerate} disabled={busy || batchBusy || !store.loaded}>
        {busy ? '생성 중…' : '✦ 그림 생성하기 →'}
      </button>
      {#if busy}
        <button class="btn" onclick={() => currentJob?.cancel()}>취소</button>
      {/if}
    </div>
    {#if notice}<p class="notice">{notice}</p>{/if}

    <details class="batch">
      <summary class="field-label">📋 배치 생성 (여러 프롬프트 한 번에)</summary>
      <div class="batch-body">
        <textarea
          class="batch-text"
          rows="6"
          bind:value={batchText}
          placeholder={'한 줄에 프롬프트 하나씩 입력\n예)\n1girl, solo, red dress, smile\n1girl, solo, blue kimono, night'}
          disabled={batchBusy}
        ></textarea>
        <p class="batch-hint">
          우측 패널의 설정(모델·해상도·샘플러·CFG·스텝·시드)을 공통으로 사용해요.
          시드가 고정돼 있으면 항목마다 1씩 증가하고, 랜덤이면 매번 새로 뽑아요.
        </p>
        <div class="btn-row">
          <button
            class="btn primary"
            onclick={onBatchStart}
            disabled={batchBusy || busy || batchCount === 0 || !store.loaded}
          >
            {batchBusy ? '배치 실행 중…' : `📋 배치 시작 (${batchCount}개)`}
          </button>
          {#if batchBusy}
            <button class="btn" onclick={onBatchCancel}>배치 취소</button>
          {/if}
        </div>
        {#if batchNotice}<p class="notice">{batchNotice}</p>{/if}
        {#if batchJob}
          <div class="batch-progress">
            <div
              class="batch-progress-bar"
              style="width: {batchJob.total > 0 ? Math.round((batchJob.done / batchJob.total) * 100) : 0}%"
            ></div>
          </div>
          <p class="batch-status">
            {batchJob.done} / {batchJob.total} —
            {batchStatusLabel[batchJob.status] ?? batchJob.status}
          </p>
          <ul class="batch-list">
            {#each batchJob.items as item, i (i)}
              <li class="batch-item" class:failed={item.status === 'error'}>
                <span class="batch-item-status">{batchStatusLabel[item.status] ?? item.status}</span>
                <span class="batch-item-prompt" title={item.prompt}>{item.prompt}</span>
                {#if item.status === 'done' && item.url}
                  <img class="batch-item-thumb" src={apiUrl(item.url)} alt="배치 결과" loading="lazy" />
                {/if}
                {#if item.status === 'error' && item.error}
                  <span class="batch-item-error" title={item.error}>⚠</span>
                {/if}
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    </details>

    <details class="batch">
      <summary class="field-label">📊 X/Y 플롯 (설정 비교 그리드)</summary>
      <div class="batch-body">
        <div class="xy-row">
          <label class="xy-axis">
            <span class="xy-label">X축</span>
            <select class="select" bind:value={xyXType} disabled={xyBusy}>
              {#each XY_TYPES as t (t.value)}
                <option value={t.value}>{t.label}</option>
              {/each}
            </select>
            <input
              class="text-input"
              bind:value={xyXValues}
              placeholder="쉼표로 구분 (예: 3, 5, 7)"
              disabled={xyBusy}
              aria-label="X축 값"
            />
          </label>
        </div>
        <div class="xy-row">
          <label class="xy-check">
            <input type="checkbox" bind:checked={xyUseY} disabled={xyBusy} />
            <span>Y축 사용</span>
          </label>
          {#if xyUseY}
            <label class="xy-axis">
              <span class="xy-label">Y축</span>
              <select class="select" bind:value={xyYType} disabled={xyBusy}>
                {#each XY_TYPES as t (t.value)}
                  <option value={t.value}>{t.label}</option>
                {/each}
              </select>
              <input
                class="text-input"
                bind:value={xyYValues}
                placeholder="쉼표로 구분 (예: 20, 30)"
                disabled={xyBusy}
                aria-label="Y축 값"
              />
            </label>
          {/if}
        </div>
        <p class="batch-hint">
          한 번에 최대 36칸까지. 시드는 기본으로 모든 칸에 고정돼서 공정하게 비교돼요
          (우측 패널 시드 사용, 랜덤이면 하나 뽑아 공유).
        </p>
        <div class="btn-row">
          <button
            class="btn primary"
            onclick={onXyStart}
            disabled={xyBusy || busy || batchBusy || !store.loaded}
          >
            {xyBusy ? 'X/Y 플롯 실행 중…' : '📊 X/Y 플롯 시작'}
          </button>
          {#if xyBusy}
            <button class="btn" onclick={onXyCancel}>취소</button>
          {/if}
        </div>
        {#if xyNotice}<p class="notice">{xyNotice}</p>{/if}
        {#if xyJob}
          <p class="batch-status">
            {xyJob.done} / {xyJob.total} —
            {batchStatusLabel[xyJob.status] ?? xyJob.status}
          </p>
          <div class="xy-grid-wrap">
            <table class="xy-grid">
              <thead>
                <tr>
                  <th></th>
                  {#each xyJob.meta?.x_labels ?? [] as xl}
                    <th>{xl}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each xyJob.meta?.y_labels?.length ? xyJob.meta.y_labels : [''] as yl, yi}
                  <tr>
                    <th>{yl}</th>
                    {#each xyJob.meta?.x_labels ?? [] as _, xi}
                      {@const cell = xyCell(xyJob, yi, xi)}
                      <td>
                        {#if cell?.status === 'done' && cell.url}
                          <button
                            class="xy-cell"
                            onclick={() => { results = [...results, apiUrl(cell.url)]; }}
                            title={cell.label ?? ''}
                          >
                            <img src={apiUrl(cell.url)} alt={cell.label ?? '플롯 결과'} loading="lazy" />
                          </button>
                        {:else}
                          <span class="xy-cell-status">{batchStatusLabel[cell?.status] ?? '…'}</span>
                        {/if}
                      </td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="batch-hint">칸을 누르면 갤러리에 크게 추가돼요.</p>
        {/if}
      </div>
    </details>
  </div>

  <div class="canvas" class:has-results={results.length > 0}>
    {#if results.length === 0}
      {#if recentLoaded && recents.length > 0}
        <p class="canvas-title">최근 작업 — 클릭하면 설정을 불러와요</p>
        <div class="recent-grid">
          {#each recents as rec (rec.id)}
            <button
              class="recent-thumb"
              onclick={() => onRecentClick(rec)}
              title="이 설정으로 불러오기"
            >
              <img src={apiUrl(rec.results[0])} alt="최근 생성 결과" loading="lazy" />
            </button>
          {/each}
        </div>
      {:else}
        <div class="canvas-empty">
          <div class="canvas-empty-icon">🖼</div>
          <p>아직 생성된 그림이 없어요.<br />위 프롬프트를 입력하고 생성 버튼을 눌러보세요.</p>
        </div>
      {/if}
    {:else}
      <div class="gallery">
        {#each results as src (src)}
          <figure class="gallery-item">
            <img {src} alt="생성 결과" />
            <button
              class="btn upscale-btn"
              onclick={() => onUpscale(src)}
              disabled={!!upscaling[src]}
              title="RealESRGAN으로 4배 키우기 (강도는 우측 패널의 '업스케일 강도')"
            >
              {upscaling[src] ? '업스케일 중…' : '🔍 업스케일'}
            </button>
            <button
              class="btn retouch-btn"
              onclick={() => onRetouch(src)}
              disabled={!!retouching[src]}
              title="얼굴을 자동 검출해 해당 영역만 다시 그리기 (ADetailer 방식)"
            >
              {retouching[src] ? '보정 중…' : '✨ 얼굴 보정'}
            </button>
            <button
              class="btn outpaint-btn"
              onclick={() => { outpaintForm = outpaintForm === src ? null : src; }}
              disabled={!!outpainting[src]}
              title="캔버스를 넓히고 새 영역을 AI로 채우기"
            >
              {outpainting[src] ? '확장 중…' : '🖼️ 확장'}
            </button>
            {#if outpaintForm === src}
              <div class="outpaint-form">
                {#each [['up', '위'], ['down', '아래'], ['left', '왼쪽'], ['right', '오른쪽']] as [k, label]}
                  <label class="outpaint-field">
                    <span>{label}</span>
                    <input
                      type="number"
                      class="number-box"
                      min="64"
                      max="512"
                      step="64"
                      bind:value={outpaintExpand[k]}
                      aria-label="확장 {label} (px)"
                    />
                  </label>
                {/each}
                <p class="hint">각 방향 64~512px. 새 영역만 다시 그려요.</p>
                <button
                  class="btn primary"
                  onclick={() => onOutpaint(src)}
                  disabled={!!outpainting[src]}
                >
                  확장 시작
                </button>
              </div>
            {/if}
          </figure>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .generate-tab {
    display: flex;
    flex-direction: column;
    gap: 14px;
    height: 100%;
  }
  .prompt-card {
    background: #1d1f38;
    border-radius: 16px;
    padding: 16px 18px;
  }
  .field-head { margin-bottom: 8px; }
  .field-label { font-weight: 700; font-size: 14px; }
  .neg { margin-top: 10px; }
  .neg summary {
    cursor: pointer;
    list-style: none;
    color: #9aa0c7;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
  }
  .neg summary::-webkit-details-marker { display: none; }
  .neg summary::before { content: '⊘ '; }
  .neg summary::after {
    content: '▾';
    margin-left: auto;
    color: #6b7090;
    transition: transform 0.18s;
  }
  .neg[open] summary::after { transform: rotate(180deg); }
  .neg summary:hover { color: #cfd2ea; }
  .neg-body { margin-top: 8px; }
  .btn-row { margin-top: 12px; }
  .cta {
    font-size: 16px;
    padding: 13px 40px;
  }
  .canvas {
    flex: 1;
    min-height: 320px;
    background: #14162b;
    border-radius: 16px;
    padding: 18px;
    overflow-y: auto;
  }
  .canvas-title {
    color: #9aa0c7;
    font-size: 13px;
    margin: 0 0 10px;
  }
  .recent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
  }
  .recent-thumb {
    padding: 0;
    border: none;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    background: #1d1f38;
    transition: transform 0.15s;
  }
  .recent-thumb:hover { transform: scale(1.03); }
  .recent-thumb img {
    width: 100%;
    aspect-ratio: 3 / 4;
    object-fit: cover;
    display: block;
  }
  .canvas-empty {
    height: 100%;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: #6b7090;
    text-align: center;
  }
  .canvas-empty-icon { font-size: 40px; opacity: 0.6; }
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
  }
  .gallery-item {
    margin: 0;
    position: relative;
  }
  .gallery-item img {
    width: 100%;
    border-radius: 12px;
    display: block;
  }
  .upscale-btn {
    position: absolute;
    right: 8px;
    bottom: 8px;
    font-size: 12.5px;
    padding: 7px 12px;
    background: rgba(20, 22, 43, 0.85);
    backdrop-filter: blur(4px);
  }
  .retouch-btn {
    position: absolute;
    left: 8px;
    bottom: 8px;
    font-size: 12.5px;
    padding: 7px 12px;
    background: rgba(20, 22, 43, 0.85);
    backdrop-filter: blur(4px);
  }
  .batch { margin-top: 12px; }
  .batch summary {
    cursor: pointer;
    list-style: none;
    color: #9aa0c7;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
  }
  .batch summary::-webkit-details-marker { display: none; }
  .batch summary::after {
    content: '▾';
    margin-left: auto;
    color: #6b7090;
    transition: transform 0.18s;
  }
  .batch[open] summary::after { transform: rotate(180deg); }
  .batch summary:hover { color: #cfd2ea; }
  .batch-body { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
  .batch-text {
    width: 100%;
    box-sizing: border-box;
    background: #14162b;
    border: 1px solid #2c2f52;
    border-radius: 10px;
    color: #e8eaf6;
    font-size: 13px;
    line-height: 1.5;
    padding: 10px 12px;
    resize: vertical;
  }
  .batch-text:disabled { opacity: 0.6; }
  .batch-hint {
    margin: 0;
    color: #6b7090;
    font-size: 12px;
    line-height: 1.5;
  }
  .batch-progress {
    height: 8px;
    background: #14162b;
    border-radius: 4px;
    overflow: hidden;
  }
  .batch-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #7c6cf0, #a78bfa);
    border-radius: 4px;
    transition: width 0.3s;
  }
  .batch-status {
    margin: 0;
    color: #9aa0c7;
    font-size: 13px;
  }
  .batch-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 220px;
    overflow-y: auto;
  }
  .batch-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #14162b;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12.5px;
  }
  .batch-item.failed { border: 1px solid #7a3040; }
  .batch-item-status {
    flex-shrink: 0;
    min-width: 52px;
    color: #9aa0c7;
    font-size: 12px;
  }
  .batch-item-prompt {
    flex: 1;
    color: #cfd2ea;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .batch-item-thumb {
    width: 40px;
    height: 40px;
    object-fit: cover;
    border-radius: 6px;
    flex-shrink: 0;
  }
  .batch-item-error { color: #e08090; flex-shrink: 0; }
  .xy-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }
  .xy-axis {
    display: flex;
    gap: 8px;
    align-items: center;
    flex: 1;
    min-width: 220px;
  }
  .xy-label { color: #cfd2ea; font-size: 13px; white-space: nowrap; }
  .xy-check {
    display: flex;
    gap: 6px;
    align-items: center;
    color: #cfd2ea;
    font-size: 13px;
    cursor: pointer;
  }
  .xy-grid-wrap { overflow-x: auto; margin-top: 10px; }
  .xy-grid { border-collapse: collapse; }
  .xy-grid th, .xy-grid td {
    padding: 4px;
    font-size: 12px;
    color: #cfd2ea;
    text-align: center;
    vertical-align: middle;
  }
  .xy-cell {
    padding: 0;
    border: 1px solid #3a3d63;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    background: none;
  }
  .xy-cell img {
    display: block;
    width: 128px;
    height: 128px;
    object-fit: cover;
  }
  .xy-cell-status { color: #8a8db0; font-size: 12px; }
  .outpaint-form {
    margin-top: 8px;
    padding: 10px;
    background: #23264a;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .outpaint-field {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #cfd2ea;
    font-size: 13px;
  }
</style>
