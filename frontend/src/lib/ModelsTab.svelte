<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import {
    fetchModels,
    setActiveModel,
    deleteModel,
    refreshModels,
  } from './api.js';

  let data = $state(null); // {models_dir, active, models}
  let notice = $state('');
  let busy = $state(false);

  const checkpoints = $derived(
    (data?.models ?? []).filter((m) => m.kind === 'checkpoint')
  );
  const loras = $derived((data?.models ?? []).filter((m) => m.kind === 'lora'));

  function loraState(name) {
    if (!store.values.loras[name]) {
      store.values.loras[name] = { enabled: false, strength: 0.8 };
    }
    return store.values.loras[name];
  }

  async function load() {
    notice = '';
    const res = await fetchModels();
    if (!res) {
      notice = '백엔드에 연결되지 않았어요. 백엔드를 먼저 실행해주세요.';
      data = null;
      return;
    }
    data = res;
    // 사라진 LoRA의 선택 상태 정리
    const names = new Set(loras.map((m) => m.name));
    for (const k of Object.keys(store.values.loras)) {
      if (!names.has(k)) delete store.values.loras[k];
    }
  }

  async function onSelect(name) {
    notice = '';
    busy = true;
    try {
      const res = await setActiveModel(name);
      data = { ...data, active: res.active };
    } catch (e) {
      notice = e.message || '모델 전환에 실패했어요.';
    } finally {
      busy = false;
    }
  }

  async function onDelete(name) {
    if (!confirm(`'${name}'을(를) 삭제할까요? 되돌릴 수 없어요.`)) return;
    notice = '';
    busy = true;
    try {
      await deleteModel(name);
      await load();
    } catch (e) {
      notice = e.message || '삭제에 실패했어요.';
    } finally {
      busy = false;
    }
  }

  async function onRefresh() {
    busy = true;
    try {
      const res = await refreshModels();
      if (res) {
        data = res;
        notice = '';
      } else {
        notice = '새로고침에 실패했어요.';
      }
    } finally {
      busy = false;
    }
  }

  function fmtSize(mb) {
    return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
  }

  onMount(load);
</script>

<div class="models-tab">
  <div class="models-head">
    <h3>모델 매니저</h3>
    <button class="btn" onclick={onRefresh} disabled={busy}>
      {busy ? '처리 중…' : '새로고침'}
    </button>
  </div>
  {#if notice}<p class="notice">{notice}</p>{/if}

  {#if data}
    {#if checkpoints.length === 0 && loras.length === 0}
      <div class="gallery-empty">
        모델이 하나도 없어요.<br />
        체크포인트(.safetensors / .ckpt)를 아래 폴더에 넣어주세요.<br />
        <code>{data.models_dir}</code>
      </div>
    {:else}
      <h4>체크포인트 ({checkpoints.length})</h4>
      {#if checkpoints.length === 0}
        <p class="hint">체크포인트가 없어요.</p>
      {/if}
      <ul class="model-list">
        {#each checkpoints as m (m.name)}
          <li>
            <label class="model-row">
              <input
                type="radio"
                name="active-model"
                checked={data.active === m.name}
                onchange={() => onSelect(m.name)}
                disabled={busy}
              />
              <span class="model-name">{m.name}</span>
              <span class="model-meta">{fmtSize(m.size_mb)}</span>
              {#if data.active === m.name}
                <span class="badge">사용 중</span>
              {/if}
            </label>
            <button
              class="btn danger small"
              onclick={() => onDelete(m.name)}
              disabled={busy || data.active === m.name}
              title={data.active === m.name ? '사용 중인 모델은 삭제할 수 없어요' : '삭제'}
            >
              삭제
            </button>
          </li>
        {/each}
      </ul>

      <h4>LoRA ({loras.length})</h4>
      {#if loras.length === 0}
        <p class="hint">
          LoRA가 없어요. <code>lora/</code> 폴더에 넣으면 여기서 보여요.
        </p>
      {/if}
      <ul class="model-list">
        {#each loras as m (m.name)}
          {@const st = loraState(m.name)}
          <li class="lora-row">
            <label class="lora-toggle">
              <input type="checkbox" bind:checked={st.enabled} />
              <span class="model-name">{m.name}</span>
              <span class="model-meta">{fmtSize(m.size_mb)}</span>
            </label>
            <div class="lora-strength">
              <span class="hint">강도</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                bind:value={st.strength}
                disabled={!st.enabled}
              />
              <span class="model-meta">{st.strength.toFixed(2)}</span>
            </div>
            <button
              class="btn danger small"
              onclick={() => onDelete(m.name)}
              disabled={busy}
            >
              삭제
            </button>
          </li>
        {/each}
      </ul>
      {#if loras.length > 0}
        <p class="hint">
          활성화한 LoRA는 생성 시 함께 전달돼요. (파이프라인 적용은 준비 중)
        </p>
      {/if}
    {/if}
  {/if}
</div>

<style>
  .models-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
  }
  .model-list {
    list-style: none;
    padding: 0;
    margin: 0 0 1.4em;
    display: flex;
    flex-direction: column;
    gap: 0.55em;
  }
  .model-list li {
    display: flex;
    align-items: center;
    gap: 0.7em;
    padding: 0.65em 0.9em;
    background: linear-gradient(180deg, #272945, var(--bg2));
    border: 1px solid #35385f;
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s, transform 0.15s;
  }
  .model-list li:hover { border-color: #4a4f80; }
  .model-row, .lora-toggle {
    display: flex;
    align-items: center;
    gap: 0.7em;
    flex: 1;
    min-width: 0;
    cursor: pointer;
  }
  .model-row input[type='radio'], .lora-toggle input[type='checkbox'] {
    accent-color: var(--accent);
    width: 17px;
    height: 17px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .model-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 500;
  }
  .model-meta {
    color: var(--muted, #999);
    font-size: 0.85em;
    white-space: nowrap;
  }
  .badge {
    font-size: 0.72em;
    font-weight: 600;
    padding: 0.25em 0.7em;
    border-radius: 999px;
    background: linear-gradient(135deg, #8aa8f5, #5a76c8);
    color: #101223;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(122, 162, 247, 0.35);
  }
  .lora-row {
    flex-wrap: wrap;
  }
  .lora-strength {
    display: flex;
    align-items: center;
    gap: 0.6em;
    flex: 1 1 200px;
  }
  .lora-strength input[type='range'] {
    flex: 1;
  }
  .btn.small {
    padding: 0.3em 0.8em;
    font-size: 0.85em;
  }
  .btn.danger {
    border-color: rgba(200, 90, 90, 0.5);
    color: #f2a3a3;
    background: linear-gradient(180deg, rgba(120, 50, 50, 0.25), rgba(90, 35, 35, 0.2));
  }
  .btn.danger:hover:not(:disabled) {
    border-color: #d66;
    color: #ffc4c4;
    transform: translateY(-1px);
  }
  .btn.danger:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .hint {
    color: var(--muted, #999);
    font-size: 0.9em;
    line-height: 1.7;
  }
  code {
    font-size: 0.88em;
    word-break: break-all;
    background: rgba(255, 255, 255, 0.05);
    padding: 0.15em 0.45em;
    border-radius: 6px;
    border: 1px solid var(--border, #333);
  }
  h4 {
    font-size: 14.5px;
    letter-spacing: -0.01em;
    margin: 1.4em 0 0.7em;
    color: var(--fg);
  }
  h4:first-of-type { margin-top: 0.6em; }
</style>
