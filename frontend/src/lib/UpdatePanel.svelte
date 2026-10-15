<script>
  import { t } from './i18n.svelte.js';
  import {
    updateState,
    checkForUpdates,
    downloadAndInstall,
    loadCurrentVersion,
    isTauri,
  } from './updater.svelte.js';
  import { onMount } from 'svelte';

  onMount(() => {
    loadCurrentVersion();
  });
</script>

<div class="update-panel">
  <div class="panel-head">
    <h2>{t('upd.title')}</h2>
  </div>
  {#if !isTauri()}
    <p class="dim">{t('upd.notInApp')}</p>
  {:else}
    <p class="dim">
      {t('upd.current')} <strong>{updateState.current || t('upd.unknown')}</strong>
    </p>
    {#if updateState.status === 'available'}
      <p>{t('upd.newPrefix')}<strong>{updateState.availableVersion}</strong>{t('upd.newSuffix')}</p>
      <button class="btn" onclick={downloadAndInstall}>
        {t('upd.download')}
      </button>
    {:else if updateState.status === 'downloading'}
      <div class="progress">
        <div class="bar" style="width: {updateState.progress}%"></div>
      </div>
      <p class="dim">{updateState.progress}% — {updateState.message}</p>
    {:else}
      <button
        class="btn"
        onclick={() => checkForUpdates()}
        disabled={updateState.status === 'checking'}
      >
        {updateState.status === 'checking' ? t('upd.checking') : t('upd.check')}
      </button>
    {/if}
    {#if updateState.message && updateState.status !== 'downloading'}
      <p class="dim">{updateState.message}</p>
    {/if}
  {/if}
</div>

<style>
  .update-panel {
    border-top: 1px solid var(--border, #e5e5e5);
    margin-top: 1.5rem;
    padding-top: 1rem;
  }
  .panel-head h2 {
    margin: 0 0 0.5rem;
  }
  .progress {
    height: 8px;
    background: var(--border, #e5e5e5);
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
  }
  .bar {
    height: 100%;
    background: var(--accent, #7c6cf0);
    transition: width 0.2s;
  }
</style>
