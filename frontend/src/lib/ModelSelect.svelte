<script>
  // 퀵 패널용 모델 선택 3칸 (체크포인트 / VAE / 텍스트 인코더)
  import { onMount } from 'svelte';
  import { t } from './i18n.svelte.js';
  import {
    modelStore,
    loadModels,
    activateModel,
    checkpoints,
    anima38packs,
    vaes,
    textEncoders,
    selectVae,
    selectTextEncoder,
  } from './models.svelte.js';

  const cps = $derived(checkpoints());
  const packs = $derived(anima38packs());
  const allModels = $derived([...cps, ...packs]);
  const vs = $derived(vaes());
  const tes = $derived(textEncoders());

  onMount(() => {
    if (!modelStore.data && !modelStore.loading) loadModels();
  });

  function onChange(e) {
    if (e.target.value) activateModel(e.target.value);
  }
</script>

{#if modelStore.notice && !modelStore.data}
  <p class="hint">{modelStore.notice}</p>
{/if}
<div class="msel-3">
  <div class="msel-row">
    <span class="msel-label">{t('msel.checkpoint')}</span>
    <select
      class="select"
      value={modelStore.data?.active ?? ''}
      onchange={onChange}
      disabled={modelStore.loading || allModels.length === 0}
      aria-label={t('msel.label')}
    >
      {#if allModels.length === 0}
        <option value="">{t('msel.noModel')}</option>
      {/if}
      {#each cps as m (m.name)}
        <option value={m.name}>{m.name}</option>
      {/each}
      {#each packs as m (m.name)}
        <option value={m.name}>{m.name} {t('msel.packSuffix')}</option>
      {/each}
    </select>
  </div>
  <div class="msel-row">
    <span class="msel-label">{t('msel.vae')}</span>
    <select
      class="select"
      value={modelStore.vae ?? ''}
      onchange={(e) => selectVae(e.target.value)}
      disabled={modelStore.loading}
      aria-label={t('msel.vae')}
    >
      <option value="">{t('msel.auto')}</option>
      {#each vs as m (m.name)}
        <option value={m.name}>{m.name}</option>
      {/each}
    </select>
  </div>
  <div class="msel-row">
    <span class="msel-label">{t('msel.textEncoder')}</span>
    <select
      class="select"
      value={modelStore.textEncoder ?? ''}
      onchange={(e) => selectTextEncoder(e.target.value)}
      disabled={modelStore.loading}
      aria-label={t('msel.textEncoder')}
    >
      <option value="">{t('msel.auto')}</option>
      {#each tes as m (m.name)}
        <option value={m.name}>{m.name}</option>
      {/each}
    </select>
  </div>
</div>
{#if modelStore.loading}
  <p class="hint">{t('msel.switching')}</p>
{/if}

<style>
  .msel-3 {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }
  .msel-row {
    display: flex;
    align-items: center;
    gap: 0.6em;
  }
  .msel-label {
    flex: 0 0 auto;
    min-width: 5.5em;
    font-size: 0.85em;
    color: var(--muted, #999);
  }
  .msel-row .select {
    flex: 1;
    min-width: 0;
  }
  .hint {
    color: var(--muted, #999);
    font-size: 0.9em;
  }
</style>
