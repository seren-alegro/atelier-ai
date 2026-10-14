<script>
  // 포즈 탭: 포즈맵(OpenPose 스켈레톤 PNG) 업로드 또는 사진에서 자동 추출
  // → ControlNet으로 포즈 지정 생성
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate, requestPoseDetect } from './api.js';
  import { modelStore, loadModels, controlnets } from './models.svelte.js';

  // backend/atelier/engine/controlnet.py의 DEFAULT_CONTROLNET과 동일하게 유지
  const DEFAULT_CN_LABEL = '기본값 사용 (xinsir/controlnet-openpose-sdxl-1.0, 자동 다운로드)';

  const cns = $derived(controlnets());
  const strengthMeta = () => store.meta.find((m) => m.key === 'controlnet_strength');
  const modelMeta = () => store.meta.find((m) => m.key === 'controlnet_model');

  let fileInput = $state(null);
  let photoInput = $state(null);
  let posemapDataUrl = $state('');
  let customModel = $state(false);
  let customRef = $state('');
  let busy = $state(false);
  let detecting = $state(false);
  let notice = $state('');
  let results = $state([]);

  onMount(() => {
    if (!modelStore.data && !modelStore.loading) loadModels();
  });

  function pickFile() {
    fileInput.click();
  }

  function onFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      posemapDataUrl = rd.result;
    };
    rd.readAsDataURL(f);
    e.target.value = '';
  }

  function effectiveModel() {
    return customModel ? customRef.trim() : (store.values.controlnet_model ?? '');
  }

  function pickPhoto() {
    photoInput.click();
  }

  async function onPhoto(e) {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (!f) return;
    notice = '';
    detecting = true;
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const rd = new FileReader();
        rd.onload = () => resolve(rd.result);
        rd.onerror = reject;
        rd.readAsDataURL(f);
      });
      const b64 = String(dataUrl).split(',')[1];
      const skeleton = await requestPoseDetect(b64);
      posemapDataUrl = 'data:image/png;base64,' + skeleton;
    } catch (err) {
      notice = err.message || '포즈 추출 중 오류가 발생했어요.';
    } finally {
      detecting = false;
    }
  }

  async function onPose() {
    notice = '';
    busy = true;
    try {
      const params = { ...store.values, controlnet_model: effectiveModel() };
      const image = posemapDataUrl.split(',')[1];
      const paths = await requestGenerate(params, 'pose', image);
      results = paths;
    } catch (e) {
      notice = e.message || '포즈 생성 중 오류가 발생했어요.';
    } finally {
      busy = false;
    }
  }
</script>

<div class="pose-tab">
  <h2>포즈 지정</h2>
  <p class="field-desc">OpenPose 스켈레톤 포즈맵을 올리면, 그 포즈 그대로 그림을 그려줘요.</p>

  <div class="field">
    <div class="field-head">
      <span class="field-label">포즈맵</span>
      <div class="field-actions">
        <button class="btn" onclick={pickFile} disabled={detecting}>포즈맵 불러오기</button>
        <button class="btn" onclick={pickPhoto} disabled={detecting}>
          {detecting ? '추출 중…' : '📷 사진에서 포즈 추출'}
        </button>
      </div>
    </div>
    <input
      type="file"
      accept="image/*"
      bind:this={fileInput}
      onchange={onFile}
      style="display:none"
    />
    <input
      type="file"
      accept="image/*"
      bind:this={photoInput}
      onchange={onPhoto}
      style="display:none"
    />
    <p class="field-desc">사진을 올리면 AI가 인물의 포즈를 스켈레톤으로 뽑아줘요. 첫 실행 시 포즈 모델(약 350MB)을 내려받아 시간이 걸릴 수 있어요.</p>
    {#if detecting}
      <div class="canvas-placeholder">포즈를 추출하고 있어요… (첫 실행은 모델 다운로드 중일 수 있어요)</div>
    {:else if !posemapDataUrl}
      <div class="canvas-placeholder">포즈맵 PNG를 불러오거나 사진에서 추출하세요</div>
    {:else}
      <div class="posemap-preview">
        <img src={posemapDataUrl} alt="포즈맵 미리보기" />
      </div>
    {/if}
  </div>

  <div class="field">
    <div class="field-head">
      <span class="field-label">ControlNet 모델</span>
      <button class="btn tool" onclick={() => (customModel = !customModel)}>
        {customModel ? '목록에서 선택' : '직접 입력'}
      </button>
    </div>
    {#if customModel}
      <input
        class="text-input"
        bind:value={customRef}
        placeholder="HF repo id 또는 로컬 경로 (예: xinsir/controlnet-openpose-sdxl-1.0)"
      />
    {:else}
      <select
        class="select"
        bind:value={store.values.controlnet_model}
        disabled={modelStore.loading}
        aria-label="ControlNet 모델 선택"
      >
        <option value="">{DEFAULT_CN_LABEL}</option>
        {#each cns as m (m.name)}
          <option value={m.name}>{m.name}</option>
        {/each}
      </select>
    {/if}
    <p class="field-desc">{modelMeta()?.description ?? ''}</p>
    {#if cns.length === 0 && !modelStore.loading}
      <p class="recommended">models/controlnet/ 폴더에 .safetensors를 넣으면 여기서 고를 수 있어요.</p>
    {/if}
  </div>

  <div class="field">
    <div class="field-head">
      <span class="field-label">ControlNet 강도</span>
      <span class="recommended">추천: {strengthMeta()?.recommended ?? 0.8}</span>
    </div>
    <p class="field-desc">{strengthMeta()?.description ?? '포즈를 얼마나 따를지 정해요.'}</p>
    <div class="number-row">
      <input
        type="range"
        min={strengthMeta()?.min ?? 0}
        max={strengthMeta()?.max ?? 1}
        step="0.05"
        bind:value={store.values.controlnet_strength}
      />
      <span class="number-box">{store.values.controlnet_strength ?? 0.8}</span>
    </div>
  </div>

  <div class="field">
    <div class="field-head"><span class="field-label">프롬프트</span></div>
    <TagInput bind:value={store.values.prompt} placeholder="태그 입력 (예: 1girl, solo)" />
  </div>

  <details class="field">
    <summary class="field-label">네거티브 프롬프트</summary>
    <div class="neg-body">
      <TagInput bind:value={store.values.negative_prompt} placeholder="제외할 태그 입력" />
    </div>
  </details>

  <button
    class="btn primary"
    onclick={onPose}
    disabled={busy || detecting || !posemapDataUrl || !store.loaded}
  >
    {busy ? '그리는 중…' : '포즈로 생성하기'}
  </button>
  {#if !posemapDataUrl}
    <p class="recommended">포즈맵을 먼저 불러오세요.</p>
  {/if}
  {#if notice}<p class="notice">{notice}</p>{/if}

  <h3>결과</h3>
  {#if results.length === 0}
    <div class="gallery-empty">아직 결과가 없어요.</div>
  {:else}
    <div class="gallery">
      {#each results as src}
        <img {src} alt="포즈 생성 결과" />
      {/each}
    </div>
  {/if}
</div>

<style>
  .posemap-preview {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #35385f;
    box-shadow: var(--shadow-sm);
    background: #000;
    display: inline-block;
    max-width: 100%;
  }
  .posemap-preview img {
    display: block;
    max-width: 320px;
    max-height: 420px;
    width: auto;
    height: auto;
  }
  .btn.tool {
    padding: 7px 14px;
    font-size: 13px;
  }
  .field-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
</style>
