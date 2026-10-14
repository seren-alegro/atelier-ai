<script>
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
    <h2>앱 업데이트</h2>
  </div>
  {#if !isTauri()}
    <p class="dim">업데이트 확인은 앱에서 실행할 때만 가능해요.</p>
  {:else}
    <p class="dim">
      현재 버전: <strong>{updateState.current || '알 수 없음'}</strong>
    </p>
    {#if updateState.status === 'available'}
      <p>새 버전 <strong>{updateState.availableVersion}</strong>이 있어요.</p>
      <button class="btn" onclick={downloadAndInstall}>
        다운로드하고 다시 시작
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
        {updateState.status === 'checking' ? '확인 중…' : '업데이트 확인'}
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
