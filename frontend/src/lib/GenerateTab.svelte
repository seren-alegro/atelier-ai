<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import { historyStore, loadHistory } from './history.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate, apiUrl } from './api.js';

  let { onLoad = null } = $props();

  let notice = $state('');
  let busy = $state(false);
  let results = $state([]);
  let currentJob = $state(null);
  let recentLoaded = $state(false);

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
      <button class="btn primary cta" onclick={onGenerate} disabled={busy || !store.loaded}>
        {busy ? '생성 중…' : '✦ 그림 생성하기 →'}
      </button>
      {#if busy}
        <button class="btn" onclick={() => currentJob?.cancel()}>취소</button>
      {/if}
    </div>
    {#if notice}<p class="notice">{notice}</p>{/if}
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
        {#each results as src}
          <img {src} alt="생성 결과" />
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
  .gallery img {
    width: 100%;
    border-radius: 12px;
    display: block;
  }
</style>
