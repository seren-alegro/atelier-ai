<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import { historyStore, loadHistory } from './history.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate, requestUpscale, requestAdetailer, requestBatch, getBatchJob, cancelBatchJob, requestOutpaint, requestXyPlot, getXyPlotJob, cancelXyPlotJob, apiUrl } from './api.js';
  import { t } from './i18n.svelte.js';

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
  const batchStatusLabel = $derived({
    queued: t('gen.statusQueued'), running: t('gen.statusRunning'), done: t('gen.statusDone'),
    error: t('gen.statusError'), cancelled: t('gen.statusCancelled'),
  });

  // X/Y 플롯
  const XY_TYPES = $derived([
    { value: 'sampler', label: t('gen.xySampler') },
    { value: 'cfg_scale', label: t('gen.xyCfg') },
    { value: 'steps', label: t('gen.xySteps') },
    { value: 'seed', label: t('gen.xySeed') },
  ]);
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
      xyNotice = t('gen.xyNoValues');
      return;
    }
    if (xAxis.values.some((v) => typeof v === 'number' && Number.isNaN(v))
        || (xyUseY && yAxis.values.some((v) => typeof v === 'number' && Number.isNaN(v)))) {
      xyNotice = t('gen.xyNumOnly');
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
          xyNotice = e.message || t('gen.xyStatusFail');
          xyBusy = false;
          xyTimer = null;
          return;
        }
        xyTimer = setTimeout(poll, 2000);
      };
      poll();
    } catch (e) {
      xyNotice = e.message || t('gen.xyStartFail');
      xyBusy = false;
    }
  }

  async function onXyCancel() {
    if (!xyJob) return;
    try {
      xyJob = await cancelXyPlotJob(xyJob.job_id);
    } catch (e) {
      xyNotice = e.message || t('gen.xyCancelFail');
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
      notice = e.message || t('gen.outpaintFail');
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
      notice = e.message || t('gen.generateFail');
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
      notice = e.message || t('gen.upscaleFail');
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
        notice = t('gen.noFace');
      }
    } catch (e) {
      notice = e.message || t('gen.retouchFail');
    } finally {
      delete retouching[src];
    }
  }

  async function onBatchStart() {
    const prompts = batchText.split('\n').map((s) => s.trim()).filter(Boolean);
    if (prompts.length === 0) {
      batchNotice = t('gen.batchNeedPrompts');
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
      batchNotice = e.message || t('gen.batchStartFail');
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
      batchNotice = e.message || t('gen.batchStatusFail');
      if (batchTimer) { clearInterval(batchTimer); batchTimer = null; }
      batchBusy = false;
    }
  }

  async function onBatchCancel() {
    if (!batchJob) return;
    try {
      batchJob = await cancelBatchJob(batchJob.job_id);
    } catch (e) {
      batchNotice = e.message || t('gen.batchCancelFail');
    }
  }
</script>

<div class="generate-tab">
  <div class="prompt-card">
    <div class="field-head">
      <span class="field-label">{t('gen.promptTitle')}</span>
    </div>
    <TagInput
      bind:value={store.values.prompt}
      placeholder={t('gen.promptPlaceholder')}
    />

    <details class="neg">
      <summary class="field-label">{t('gen.negativeTitle')}</summary>
      <div class="neg-body">
        <TagInput bind:value={store.values.negative_prompt} placeholder={t('gen.negativePlaceholder')} />
      </div>
    </details>

    <div class="btn-row">
      <button class="btn primary cta" onclick={onGenerate} disabled={busy || batchBusy || !store.loaded}>
        {busy ? t('gen.generating') : t('gen.generate')}
      </button>
      {#if busy}
        <button class="btn" onclick={() => currentJob?.cancel()}>{t('common.cancel')}</button>
      {/if}
    </div>
    {#if notice}<p class="notice">{notice}</p>{/if}

    <details class="batch">
      <summary class="field-label">{t('gen.batchTitle')}</summary>
      <div class="batch-body">
        <textarea
          class="batch-text"
          rows="6"
          bind:value={batchText}
          placeholder={t('gen.batchPlaceholder')}
          disabled={batchBusy}
        ></textarea>
        <p class="batch-hint">{t('gen.batchHint')}</p>
        <div class="btn-row">
          <button
            class="btn primary"
            onclick={onBatchStart}
            disabled={batchBusy || busy || batchCount === 0 || !store.loaded}
          >
            {batchBusy ? t('gen.batchRunning') : t('gen.batchStart', { count: batchCount })}
          </button>
          {#if batchBusy}
            <button class="btn" onclick={onBatchCancel}>{t('gen.batchCancel')}</button>
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
                  <img class="batch-item-thumb" src={apiUrl(item.url)} alt={t('gen.batchThumbAlt')} loading="lazy" />
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
      <summary class="field-label">{t('gen.xyTitle')}</summary>
      <div class="batch-body">
        <div class="xy-row">
          <label class="xy-axis">
            <span class="xy-label">{t('gen.axisX')}</span>
            <select class="select" bind:value={xyXType} disabled={xyBusy}>
              {#each XY_TYPES as t (t.value)}
                <option value={t.value}>{t.label}</option>
              {/each}
            </select>
            <input
              class="text-input"
              bind:value={xyXValues}
              placeholder={t('gen.xyXPlaceholder')}
              disabled={xyBusy}
              aria-label={t('gen.xyXValuesAria')}
            />
          </label>
        </div>
        <div class="xy-row">
          <label class="xy-check">
            <input type="checkbox" bind:checked={xyUseY} disabled={xyBusy} />
            <span>{t('gen.xyUseY')}</span>
          </label>
          {#if xyUseY}
            <label class="xy-axis">
              <span class="xy-label">{t('gen.axisY')}</span>
              <select class="select" bind:value={xyYType} disabled={xyBusy}>
                {#each XY_TYPES as t (t.value)}
                  <option value={t.value}>{t.label}</option>
                {/each}
              </select>
              <input
                class="text-input"
                bind:value={xyYValues}
                placeholder={t('gen.xyYPlaceholder')}
                disabled={xyBusy}
                aria-label={t('gen.xyYValuesAria')}
              />
            </label>
          {/if}
        </div>
        <p class="batch-hint">{t('gen.xyHint')}</p>
        <div class="btn-row">
          <button
            class="btn primary"
            onclick={onXyStart}
            disabled={xyBusy || busy || batchBusy || !store.loaded}
          >
            {xyBusy ? t('gen.xyRunning') : t('gen.xyStart')}
          </button>
          {#if xyBusy}
            <button class="btn" onclick={onXyCancel}>{t('common.cancel')}</button>
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
                            <img src={apiUrl(cell.url)} alt={cell.label ?? t('gen.xyCellAlt')} loading="lazy" />
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
          <p class="batch-hint">{t('gen.xyCellHint')}</p>
        {/if}
      </div>
    </details>
  </div>

  <div class="canvas" class:has-results={results.length > 0}>
    {#if results.length === 0}
      {#if recentLoaded && recents.length > 0}
        <p class="canvas-title">{t('gen.recentTitle')}</p>
        <div class="recent-grid">
          {#each recents as rec (rec.id)}
            <button
              class="recent-thumb"
              onclick={() => onRecentClick(rec)}
              title={t('hist.loadSettings')}
            >
              <img src={apiUrl(rec.results[0])} alt={t('gen.recentAlt')} loading="lazy" />
            </button>
          {/each}
        </div>
      {:else}
        <div class="canvas-empty">
          <div class="canvas-empty-icon">🖼</div>
          <p>{t('gen.empty1')}<br />{t('gen.empty2')}</p>
        </div>
      {/if}
    {:else}
      <div class="gallery">
        {#each results as src (src)}
          <figure class="gallery-item">
            <img {src} alt={t('gen.galleryAlt')} />
            <button
              class="btn upscale-btn"
              onclick={() => onUpscale(src)}
              disabled={!!upscaling[src]}
              title={t('gen.upscaleTitle')}
            >
              {upscaling[src] ? t('gen.upscaling') : t('gen.upscale')}
            </button>
            <button
              class="btn retouch-btn"
              onclick={() => onRetouch(src)}
              disabled={!!retouching[src]}
              title={t('gen.retouchTitle')}
            >
              {retouching[src] ? t('gen.retouching') : t('gen.retouch')}
            </button>
            <button
              class="btn outpaint-btn"
              onclick={() => { outpaintForm = outpaintForm === src ? null : src; }}
              disabled={!!outpainting[src]}
              title={t('gen.outpaintTitle')}
            >
              {outpainting[src] ? t('gen.outpainting') : t('gen.outpaint')}
            </button>
            {#if outpaintForm === src}
              <div class="outpaint-form">
                {#each [['up', 'gen.dirUp'], ['down', 'gen.dirDown'], ['left', 'gen.dirLeft'], ['right', 'gen.dirRight']] as [k, labelKey]}
                  <label class="outpaint-field">
                    <span>{t(labelKey)}</span>
                    <input
                      type="number"
                      class="number-box"
                      min="64"
                      max="512"
                      step="64"
                      bind:value={outpaintExpand[k]}
                      aria-label={t('gen.expandAria', { label: t(labelKey) })}
                    />
                  </label>
                {/each}
                <p class="hint">{t('gen.expandHint')}</p>
                <button
                  class="btn primary"
                  onclick={() => onOutpaint(src)}
                  disabled={!!outpainting[src]}
                >
                  {t('gen.expandStart')}
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
