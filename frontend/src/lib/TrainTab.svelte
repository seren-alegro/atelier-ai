<script>
  import { onMount, onDestroy } from 'svelte';
  import { t } from './i18n.svelte.js';
  import {
    checkpoints,
    loadModels,
    rescanModels,
  } from './models.svelte.js';
  import {
    requestTrain,
    getTrainJob,
    requestCaptionBatch,
  } from './api.js';

  const cps = $derived(checkpoints());

  // 입력값 (추천값 표시용 상수와 함께)
  let baseModel = $state('');
  let datasetDir = $state('');
  let triggerWord = $state('');
  let rank = $state(16);
  let lr = $state(0.0001);
  let stepsPerImage = $state(75);
  let resolution = $state(1024);
  let trainTextEncoder = $state(false);
  let outputName = $state('my-lora');
  let captionMethod = $state('wd14');

  // 상태
  let captionBusy = $state(false);
  let captionResult = $state(null);
  let trainBusy = $state(false);
  let trainError = $state('');
  let job = $state(null);
  let pollTimer = $state(null);

  onMount(async () => {
    await loadModels();
    if (cps.length > 0 && !baseModel) baseModel = cps[0].name;
  });

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer);
  });

  async function onCaptionBatch() {
    if (!datasetDir.trim()) {
      captionResult = { error: t('train.needDataset') };
      return;
    }
    captionBusy = true;
    captionResult = null;
    try {
      const r = await requestCaptionBatch(datasetDir.trim(), captionMethod);
      captionResult = {
        ok: true,
        done: r.done.length,
        skipped: r.skipped.length,
        failed: r.failed.length,
      };
    } catch (e) {
      captionResult = { error: e.message };
    } finally {
      captionBusy = false;
    }
  }

  async function onTrain() {
    trainError = '';
    if (!baseModel) {
      trainError = t('train.needBase');
      return;
    }
    if (!datasetDir.trim()) {
      trainError = t('train.needDataset');
      return;
    }
    trainBusy = true;
    job = null;
    try {
      const r = await requestTrain({
        model: baseModel,
        dataset_dir: datasetDir.trim(),
        trigger_word: triggerWord.trim(),
        rank,
        lr,
        steps_per_image: stepsPerImage,
        resolution,
        train_text_encoder: trainTextEncoder,
        output_name: outputName.trim() || 'my-lora',
      });
      job = {
        job_id: r.job_id,
        status: 'queued',
        step: 0,
        total_steps: r.total_steps,
        loss: null,
        error: null,
        results: [],
      };
      pollTimer = setInterval(pollJob, 3000);
      pollJob();
    } catch (e) {
      trainError = e.message;
      trainBusy = false;
    }
  }

  async function pollJob() {
    if (!job) return;
    try {
      const j = await getTrainJob(job.job_id);
      job = j;
      if (j.status === 'done' || j.status === 'error') {
        clearInterval(pollTimer);
        pollTimer = null;
        trainBusy = false;
        if (j.status === 'done') rescanModels(); // LoRA 목록 갱신
      }
    } catch (e) {
      // 폴링 실패는 조용히 재시도
    }
  }

  const progressPct = $derived(
    job && job.total_steps > 0
      ? Math.min(100, Math.round((job.step / job.total_steps) * 100))
      : 0
  );
</script>

<div class="train-tab">
  <h2>{t('train.title')}</h2>
  <p class="tab-desc">{t('train.desc')}</p>

  <!-- 1. 베이스 모델 -->
  <section>
    <h3>{t('train.sec1')}</h3>
    <p class="field-desc">{t('train.sec1Desc')}</p>
    {#if cps.length === 0}
      <p class="warn">{t('train.noCkpt')}</p>
    {:else}
      <select bind:value={baseModel}>
        {#each cps as c (c.name)}
          <option value={c.name}>{c.name}</option>
        {/each}
      </select>
    {/if}
  </section>

  <!-- 2. 데이터셋 + 자동 캡션 -->
  <section>
    <h3>{t('train.sec2')}</h3>
    <p class="field-desc">
      {t('train.sec2Desc')}
    </p>
    <input
      type="text"
      placeholder="/home/seren/dataset/my-char"
      bind:value={datasetDir}
    />
    <div class="row">
      <label>
        {t('train.captionMethod')}
        <select bind:value={captionMethod}>
          <option value="wd14">{t('train.wd14')}</option>
          <option value="blip">{t('train.blip')}</option>
        </select>
      </label>
      <button
        class="secondary"
        onclick={onCaptionBatch}
        disabled={captionBusy}
      >
        {captionBusy ? t('train.captionBusy') : '일괄 자동 캡션'}
      </button>
    </div>
    {#if captionResult}
      {#if captionResult.error}
        <p class="error">{captionResult.error}</p>
      {:else}
        <p class="ok">
          {t('train.captionDone', { n: captionResult.done })}
          {#if captionResult.skipped > 0}{t('train.captionSkipped', { n: captionResult.skipped })}{/if}
          {#if captionResult.failed > 0}{t('train.captionFailed', { n: captionResult.failed })}{/if}
        </p>
      {/if}
    {/if}
  </section>

  <!-- 3. 학습 설정 -->
  <section>
    <h3>{t('train.sec3')}</h3>

    <label class="field">
      <span class="field-label">{t('train.trigger')}</span>
      <span class="field-desc">{t('train.triggerDesc')}</span>
      <input type="text" placeholder="sks girl" bind:value={triggerWord} />
    </label>

    <label class="field">
      <span class="field-label">{t('train.rank')}</span>
      <span class="field-desc">{t('train.rankDesc')}</span>
      <input type="number" min="1" max="128" bind:value={rank} />
    </label>

    <label class="field">
      <span class="field-label">{t('train.lr')}</span>
      <span class="field-desc">{t('train.lrDesc')}</span>
      <input type="number" min="0.00001" max="0.01" step="0.00001" bind:value={lr} />
    </label>

    <label class="field">
      <span class="field-label">{t('train.spi')}</span>
      <span class="field-desc">{t('train.spiDesc')}</span>
      <input type="number" min="1" max="1000" bind:value={stepsPerImage} />
    </label>

    <label class="field">
      <span class="field-label">{t('train.res')}</span>
      <span class="field-desc">{t('train.resDesc')}</span>
      <input type="number" min="256" max="2048" step="64" bind:value={resolution} />
    </label>

    <label class="field check">
      <input type="checkbox" bind:checked={trainTextEncoder} />
      <span class="field-label">{t('train.te')}</span>
    </label>
    <p class="field-desc">{t('train.teDesc')}</p>

    <label class="field">
      <span class="field-label">{t('train.outName')}</span>
      <span class="field-desc">{t('train.outNameDesc')}</span>
      <input type="text" placeholder="my-lora" bind:value={outputName} />
    </label>
  </section>

  <!-- 4. 학습 시작 -->
  <section>
    <button class="primary" onclick={onTrain} disabled={trainBusy}>
      {trainBusy ? t('train.busy') : t('train.start')}
    </button>
    {#if trainError}
      <p class="error">{trainError}</p>
    {/if}
    {#if job}
      <div class="progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: {progressPct}%"></div>
        </div>
        <p>
          {#if job.status === 'queued'}{t('train.queued')}{/if}
          {#if job.status === 'running'}
            {t('train.progress', { step: job.step, total: job.total_steps, pct: progressPct })}
            {#if job.loss != null}· loss {job.loss}{/if}
          {/if}
          {#if job.status === 'done'}
            <span class="ok">{t('train.doneMsg', { results: job.results.join(', ') })}</span>
          {/if}
          {#if job.status === 'error'}
            <span class="error">{t('train.failMsg', { error: job.error })}</span>
          {/if}
        </p>
      </div>
    {/if}
  </section>
</div>

<style>
  .train-tab { max-width: 640px; padding: 1rem; }
  .tab-desc { color: #888; margin-bottom: 1.5rem; }
  section { margin-bottom: 1.75rem; }
  h3 { margin-bottom: 0.25rem; }
  .field { display: block; margin-bottom: 1rem; }
  .field-label { font-weight: 600; display: block; margin-bottom: 0.15rem; }
  .field-desc { color: #888; font-size: 0.85rem; margin: 0 0 0.4rem; }
  .field.check { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.25rem; }
  .field.check .field-label { margin: 0; }
  input[type="text"], input[type="number"], select {
    width: 100%; padding: 0.5rem; border: 1px solid #444;
    border-radius: 6px; background: #1e1e1e; color: #eee;
  }
  .row { display: flex; gap: 0.75rem; align-items: end; margin-top: 0.75rem; }
  .row label { flex: 1; }
  button { padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; border: none; }
  button.primary { background: #7c5cff; color: #fff; font-weight: 600; }
  button.primary:disabled { opacity: 0.5; cursor: default; }
  button.secondary { background: #333; color: #eee; white-space: nowrap; }
  button.secondary:disabled { opacity: 0.5; }
  .error { color: #ff7b7b; }
  .ok { color: #7bff9e; }
  .warn { color: #ffcf7b; }
  .progress { margin-top: 1rem; }
  .progress-bar { height: 10px; background: #333; border-radius: 5px; overflow: hidden; }
  .progress-fill { height: 100%; background: #7c5cff; transition: width 0.5s; }
</style>
