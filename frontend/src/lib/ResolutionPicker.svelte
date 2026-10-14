<script>
  // 해상도 선택: 그룹별 프리셋 + 커스텀 슬라이더 (최대 2048)
  import { store } from './store.svelte.js';

  const GROUPS = [
    {
      id: 'square',
      name: 'Square',
      presets: [{ name: '', w: 1536, h: 1536 }],
    },
    {
      id: 'portrait',
      name: 'Portrait',
      basic: true,
      presets: [
        { name: '', w: 768, h: 1536 },
        { name: '', w: 1024, h: 1536 },
        { name: '', w: 1152, h: 1536 },
        { name: '', w: 1152, h: 1440 },
        { name: '', w: 864, h: 1536 },
      ],
    },
    {
      id: 'landscape',
      name: 'Landscape',
      basic: true,
      presets: [
        { name: '', w: 1536, h: 768 },
        { name: '', w: 1536, h: 1024 },
        { name: '', w: 1536, h: 1152 },
        { name: '', w: 1440, h: 1152 },
        { name: '', w: 1536, h: 864 },
      ],
    },
    {
      id: 'lab',
      name: '실험실',
      presets: [
        { name: 'Square', w: 1024, h: 1024 },
        { name: 'Portrait', w: 896, h: 1152 },
        { name: 'Landscape', w: 1152, h: 896 },
        { name: 'Panorama', w: 1216, h: 832 },
        { name: 'Vertical Panorama', w: 832, h: 1216 },
        { name: 'Cinematic Wide', w: 1536, h: 640 },
        { name: 'Cinematic Portrait', w: 768, h: 1344 },
        { name: 'Extended Portrait', w: 640, h: 1536 },
      ],
    },
  ];

  let groupId = $state('portrait');
  const group = $derived(GROUPS.find((g) => g.id === groupId) ?? GROUPS[1]);

  function setRes(w, h, gid) {
    store.values.width = w;
    store.values.height = h;
    if (gid) groupId = gid;
  }

  function isActive(p) {
    return store.values.width === p.w && store.values.height === p.h;
  }

  function onCustomInput() {
    groupId = 'custom';
  }
</script>

<div class="res-picker">
  <div class="res-groups" role="tablist" aria-label="해상도 그룹">
    {#each GROUPS as g (g.id)}
      <button
        class="res-group-btn"
        class:active={groupId === g.id}
        onclick={() => { groupId = g.id; }}
        role="tab"
      >
        {g.name}{g.basic ? ' · 기본' : ''}
      </button>
    {/each}
    <button
      class="res-group-btn"
      class:active={groupId === 'custom'}
      onclick={() => { groupId = 'custom'; }}
      role="tab"
    >
      커스텀
    </button>
  </div>

  {#if groupId === 'custom'}
    <div class="res-custom">
      <label class="res-slider">
        <span class="res-slider-head"><span>가로</span><b>{store.values.width}px</b></span>
        <input
          type="range" min="256" max="2048" step="64"
          bind:value={store.values.width}
          oninput={onCustomInput}
          aria-label="가로 해상도"
        />
      </label>
      <label class="res-slider">
        <span class="res-slider-head"><span>세로</span><b>{store.values.height}px</b></span>
        <input
          type="range" min="256" max="2048" step="64"
          bind:value={store.values.height}
          oninput={onCustomInput}
          aria-label="세로 해상도"
        />
      </label>
      <p class="res-hint">최대 2048px까지. 클수록 VRAM을 많이 써요.</p>
    </div>
  {:else}
    <div class="seg-grid">
      {#each group.presets as p (`${p.w}x${p.h}`)}
        <button
          class="seg-btn"
          class:active={isActive(p)}
          onclick={() => setRes(p.w, p.h, group.id)}
          title={p.name ? `${p.name} ${p.w}×${p.h}` : `${p.w}×${p.h}`}
        >
          {#if p.name}<span class="seg-name">{p.name}</span>{/if}
          <span>{p.w}×{p.h}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .res-groups {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
  }
  .res-group-btn {
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .res-group-btn:hover {
    border-color: var(--accent);
    color: var(--text);
  }
  .res-group-btn.active {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
    border-color: var(--accent);
    color: var(--text);
    font-weight: 600;
  }
  .res-custom {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .res-slider {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .res-slider-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 12px;
    color: var(--text-dim);
  }
  .res-slider-head b {
    color: var(--text);
    font-size: 13px;
  }
  .res-hint {
    margin: 0;
    font-size: 11px;
    color: var(--text-dim);
  }
  .seg-name {
    display: block;
    font-size: 10px;
    opacity: 0.75;
    margin-bottom: 2px;
  }
</style>
