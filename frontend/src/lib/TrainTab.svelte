<script>
  import { onMount, onDestroy } from 'svelte';
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
      captionResult = { error: '데이터셋 폴더를 입력해주세요.' };
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
      trainError = '베이스 체크포인트를 선택해주세요.';
      return;
    }
    if (!datasetDir.trim()) {
      trainError = '데이터셋 폴더를 입력해주세요.';
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
  <h2>LoRA 학습</h2>
  <p class="tab-desc">
    내 이미지로 캐릭터·화풍 LoRA를 직접 학습해요.
    결과는 모델 폴더의 lora/에 저장되어 바로 쓸 수 있어요.
    학습에는 12GB 이상 VRAM이 필요해요.
  </p>

  <!-- 1. 베이스 모델 -->
  <section>
    <h3>1. 베이스 모델</h3>
    <p class="field-desc">학습시킬 체크포인트예요. Anima(SDXL 계열) 권장.</p>
    {#if cps.length === 0}
      <p class="warn">체크포인트가 없어요. 모델 탭에서 먼저 확인해주세요.</p>
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
    <h3>2. 데이터셋</h3>
    <p class="field-desc">
      학습 이미지 폴더예요. 15~30장 권장. 캡션 .txt가 없으면 아래 자동 캡션으로 만들 수 있어요.
    </p>
    <input
      type="text"
      placeholder="/home/seren/dataset/my-char"
      bind:value={datasetDir}
    />
    <div class="row">
      <label>
        캡션 방식
        <select bind:value={captionMethod}>
          <option value="wd14">WD14 v3 (단보루 태그)</option>
          <option value="blip">BLIP (자연어 문장)</option>
        </select>
      </label>
      <button
        class="secondary"
        onclick={onCaptionBatch}
        disabled={captionBusy}
      >
        {captionBusy ? '캡션 생성 중…' : '일괄 자동 캡션'}
      </button>
    </div>
    {#if captionResult}
      {#if captionResult.error}
        <p class="error">{captionResult.error}</p>
      {:else}
        <p class="ok">
          캡션 {captionResult.done}장 생성
          {#if captionResult.skipped > 0}(건너뜀 {captionResult.skipped}장){/if}
          {#if captionResult.failed > 0}(실패 {captionResult.failed}장){/if}
        </p>
      {/if}
    {/if}
  </section>

  <!-- 3. 학습 설정 -->
  <section>
    <h3>3. 학습 설정</h3>

    <label class="field">
      <span class="field-label">트리거 워드</span>
      <span class="field-desc">캡션 맨 앞에 자동 삽입돼요. 생성할 때 이 단어로 캐릭터를 불러내요.</span>
      <input type="text" placeholder="sks girl" bind:value={triggerWord} />
    </label>

    <label class="field">
      <span class="field-label">Rank (dim) — 추천 16</span>
      <span class="field-desc">클수록 표현력이 오르지만 파일이 커지고 VRAM을 더 써요. 캐릭터 LoRA는 16이 무난해요.</span>
      <input type="number" min="1" max="128" bind:value={rank} />
    </label>

    <label class="field">
      <span class="field-label">학습률 — 추천 0.0001</span>
      <span class="field-desc">너무 크면 발산하고, 너무 작으면 안 배워요. AdamW 기준 0.0001이 무난해요.</span>
      <input type="number" min="0.00001" max="0.01" step="0.00001" bind:value={lr} />
    </label>

    <label class="field">
      <span class="field-label">이미지당 스텝 — 추천 75</span>
      <span class="field-desc">총 스텝 = 이미지 수 × 이 값. 20장이면 1500 스텝이에요.</span>
      <input type="number" min="1" max="1000" bind:value={stepsPerImage} />
    </label>

    <label class="field">
      <span class="field-label">해상도 — 추천 1024</span>
      <span class="field-desc">Anima(SDXL 계열) 네이티브 해상도예요. 클수록 VRAM을 더 써요.</span>
      <input type="number" min="256" max="2048" step="64" bind:value={resolution} />
    </label>

    <label class="field check">
      <input type="checkbox" bind:checked={trainTextEncoder} />
      <span class="field-label">텍스트 인코더도 학습 (기본 OFF)</span>
    </label>
    <p class="field-desc">켰을 때 품질이 오를 수 있지만 VRAM이 크게 늘어요. 처음엔 꺼두는 걸 권장해요.</p>

    <label class="field">
      <span class="field-label">저장 파일명</span>
      <span class="field-desc">lora/ 폴더에 이 이름으로 저장돼요.</span>
      <input type="text" placeholder="my-lora" bind:value={outputName} />
    </label>
  </section>

  <!-- 4. 학습 시작 -->
  <section>
    <button class="primary" onclick={onTrain} disabled={trainBusy}>
      {trainBusy ? '학습 중…' : '학습 시작'}
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
          {#if job.status === 'queued'}대기 중…{/if}
          {#if job.status === 'running'}
            {job.step} / {job.total_steps} 스텝 ({progressPct}%)
            {#if job.loss != null}· loss {job.loss}{/if}
          {/if}
          {#if job.status === 'done'}
            <span class="ok">완료! {job.results.join(', ')} — 모델 탭 LoRA에서 바로 쓸 수 있어요.</span>
          {/if}
          {#if job.status === 'error'}
            <span class="error">실패: {job.error}</span>
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
