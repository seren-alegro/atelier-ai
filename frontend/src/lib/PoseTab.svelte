<script>
  // 포즈 탭: 포즈맵(OpenPose 스켈레톤 PNG) 업로드 또는 사진에서 자동 추출
  // → ControlNet으로 포즈 지정 생성
  import { onMount } from 'svelte';
  import { t, metaDesc } from './i18n.svelte.js';
  import { store } from './store.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate, requestPoseDetect } from './api.js';
  import { modelStore, loadModels, controlnets } from './models.svelte.js';

  // 백엔드 controlnet.py의 DEFAULT_CONTROLNET과 동기화:
  // 설정 설명문의 "기본값(...)"에서 파싱하므로 백엔드 설명이 바뀌면 따라간다.
  const defaultCnId = $derived(
    modelMeta()?.description?.match(/기본값\(([^)]+)\)/)?.[1]
      ?? 'xinsir/controlnet-openpose-sdxl-1.0'
  );
  const defaultCnLabel = $derived(t('pose.defaultUse', { id: defaultCnId }));

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
  let currentJob = $state(null);

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
      notice = err.message || t('pose.extractFail');
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
      currentJob = requestGenerate(params, 'pose', image);
      const paths = await currentJob;
      results = paths;
    } catch (e) {
      notice = e.message || t('pose.generateFail');
    } finally {
      busy = false;
      currentJob = null;
    }
  }
</script>

<div class="pose-tab">
  <h2>{t('hist.modePose')}</h2>
  <p class="field-desc">{t('pose.desc')}</p>

  <div class="field">
    <div class="field-head">
      <span class="field-label">{t('pose.mapLabel')}</span>
      <div class="field-actions">
        <button class="btn" onclick={pickFile} disabled={detecting}>{t('pose.loadMap')}</button>
        <button class="btn" onclick={pickPhoto} disabled={detecting}>
          {detecting ? t('pose.extracting') : t('pose.extract')}
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
    <p class="field-desc">{t('pose.photoHint')}</p>
    {#if detecting}
      <div class="canvas-placeholder">{t('pose.extractingHint')}</div>
    {:else if !posemapDataUrl}
      <div class="canvas-placeholder">{t('pose.mapHint')}</div>
    {:else}
      <div class="posemap-preview">
        <img src={posemapDataUrl} alt={t('pose.mapAlt')} />
      </div>
    {/if}
  </div>

  <div class="field">
    <div class="field-head">
      <span class="field-label">{t('pose.cnLabel')}</span>
      <button class="btn tool" onclick={() => (customModel = !customModel)}>
        {customModel ? t('pose.customList') : t('pose.customInput')}
      </button>
    </div>
    {#if customModel}
      <input
        class="text-input"
        bind:value={customRef}
        placeholder={t('pose.cnPlaceholder')}
      />
    {:else}
      <select
        class="select"
        bind:value={store.values.controlnet_model}
        disabled={modelStore.loading}
        aria-label={t('pose.cnAria')}
      >
        <option value="">{defaultCnLabel}</option>
        {#each cns as m (m.name)}
          <option value={m.name}>{m.name}</option>
        {/each}
      </select>
    {/if}
    <p class="field-desc">{metaDesc(modelMeta()) ?? ''}</p>
    {#if cns.length === 0 && !modelStore.loading}
      <p class="recommended">{t('pose.cnEmpty')}</p>
    {/if}
  </div>

  <div class="field">
    <div class="field-head">
      <span class="field-label">{t('pose.strengthLabel')}</span>
      <span class="recommended">{t('common.recommended')} {strengthMeta()?.recommended ?? 0.8}</span>
    </div>
    <p class="field-desc">{metaDesc(strengthMeta()) ?? t('pose.strengthDesc')}</p>
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
    <div class="field-head"><span class="field-label">{t('inp.prompt')}</span></div>
    <TagInput bind:value={store.values.prompt} placeholder={t('inp.promptPh')} />
  </div>

  <details class="field">
    <summary class="field-label">{t('inp.negative')}</summary>
    <div class="neg-body">
      <TagInput bind:value={store.values.negative_prompt} placeholder={t('gen.negativePlaceholder')} />
    </div>
  </details>

  <button
    class="btn primary"
    onclick={onPose}
    disabled={busy || detecting || !posemapDataUrl || !store.loaded}
  >
    {busy ? t('common.painting') : t('pose.generate')}
  </button>
  {#if busy}
    <button class="btn" onclick={() => currentJob?.cancel()}>{t('common.cancel')}</button>
  {/if}
  {#if !posemapDataUrl}
    <p class="recommended">{t('pose.needMap')}</p>
  {/if}
  {#if notice}<p class="notice">{notice}</p>{/if}

  <h3>{t('common.resultsTitle')}</h3>
  {#if results.length === 0}
    <div class="gallery-empty">{t('common.noResults')}</div>
  {:else}
    <div class="gallery">
      {#each results as src}
        <img {src} alt={t('pose.resultAlt')} />
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
