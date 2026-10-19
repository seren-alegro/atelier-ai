<script>
  import { onMount } from 'svelte';
  import { t } from './i18n.svelte.js';
  import { store } from './store.svelte.js';
  import {
    modelStore,
    loadModels,
    rescanModels,
    activateModel,
    removeModel,
    checkpoints,
    loras,
    controlnets,
    anima38packs,
  } from './models.svelte.js';
  import { startDownload, fetchDownloadStatus } from './api.js';

  const cps = $derived(checkpoints());
  const lrs = $derived(loras());
  const cns = $derived(controlnets());
  const packs = $derived(anima38packs());

  // -- URL 다운로드 ------------------------------------------------------
  let dlUrl = $state('');
  let dlKind = $state('auto');
  let dlJob = $state(null); // {id, status, filename, downloaded_bytes, total_bytes, error}
  let dlTimer = null;
  let packUrls = $state({}); // pack name → url

  function pollDownload() {
    clearInterval(dlTimer);
    dlTimer = setInterval(async () => {
      if (!dlJob) {
        clearInterval(dlTimer);
        return;
      }
      const st = await fetchDownloadStatus(dlJob.id);
      if (!st) return;
      dlJob = st;
      if (st.status === 'done') {
        clearInterval(dlTimer);
        dlUrl = '';
        modelStore.notice = t('mdl.dlDone');
        await rescanModels();
      } else if (st.status === 'error') {
        clearInterval(dlTimer);
        modelStore.notice =
          (st.error ? st.error + ' — ' : '') + t('mdl.dlFail');
      }
    }, 1000);
  }

  async function beginDownload(url, kind, packDir = null) {
    url = (url ?? '').trim();
    if (!url) return;
    try {
      const { job_id } = await startDownload(
        url, kind, packDir, store.values.civitai_api_key || ''
      );
      dlJob = {
        id: job_id,
        status: 'downloading',
        filename: '',
        downloaded_bytes: 0,
        total_bytes: null,
        error: null,
      };
      modelStore.notice = t('mdl.dlStarting');
      pollDownload();
    } catch (e) {
      modelStore.notice = e.message || t('mdl.dlFail');
    }
  }

  function onDownloadStart() {
    beginDownload(dlUrl, dlKind);
  }

  function onPackDownload(packName) {
    beginDownload(packUrls[packName], 'pack', packName);
  }

  function fmtDlBytes(b) {
    if (b == null) return '?';
    if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(2) + ' GB';
    if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(1) + ' MB';
    return Math.round(b / 1024) + ' KB';
  }

  function packRows(pack) {
    const st = pack.pack_status ?? {};
    return [
      { label: t('mdl.packDit'), ok: !!st.dit, detail: st.dit ?? '' },
      { label: t('mdl.packTe4b'), ok: !!st.qwen35_4b },
      { label: t('mdl.packTe06b'), ok: !!st.qwen3_06b },
      { label: t('mdl.packAdapter'), ok: !!st.adapter, detail: st.adapter ?? '' },
      {
        label: t('mdl.packTok'),
        ok: (st.tokenizer_dirs ?? []).length > 0,
      },
      { label: t('mdl.packVae'), ok: !!st.vae_dir },
    ];
  }

  function packIncomplete(pack) {
    return packRows(pack).some((r) => !r.ok);
  }

  function loraState(name) {
    if (!store.values.loras[name]) {
      store.values.loras[name] = { enabled: false, strength: 0.8 };
    }
    return store.values.loras[name];
  }

  async function load() {
    await loadModels();
    // 사라진 LoRA의 선택 상태 정리
    const names = new Set((modelStore.data?.models ?? []).map((m) => m.name));
    for (const k of Object.keys(store.values.loras)) {
      if (!names.has(k)) delete store.values.loras[k];
    }
  }

  function onSelect(name) {
    activateModel(name);
  }

  function onDelete(name) {
    if (!confirm(t('mdl.deleteConfirm', { name }))) return;
    removeModel(name);
  }

  function onRefresh() {
    rescanModels();
  }

  function fmtSize(mb) {
    return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
  }

  onMount(load);
</script>

<div class="models-tab">
  <div class="models-head">
    <h3>{t('mdl.title')}</h3>
    <button class="btn" onclick={onRefresh} disabled={modelStore.loading}>
      {modelStore.loading ? t('mdl.processing') : t('common.refresh')}
    </button>
  </div>
  {#if modelStore.notice}<p class="notice">{modelStore.notice}</p>{/if}

  <div class="dl-box">
    <h4>{t('mdl.dlTitle')}</h4>
    <div class="dl-row">
      <input
        class="text-input dl-url"
        type="url"
        bind:value={dlUrl}
        placeholder={t('mdl.dlUrlPh')}
      />
      <select class="select" bind:value={dlKind} aria-label={t('mdl.dlTitle')}>
        <option value="auto">{t('mdl.dlKindAuto')}</option>
        <option value="checkpoint">{t('mdl.dlKindCkpt')}</option>
        <option value="lora">{t('mdl.dlKindLora')}</option>
      </select>
      <button
        class="btn"
        onclick={onDownloadStart}
        disabled={!dlUrl.trim() || (dlJob && dlJob.status === 'downloading')}
      >
        {t('mdl.dlStart')}
      </button>
    </div>
    {#if dlJob}
      <div class="dl-progress">
        <span class="model-name">{dlJob.filename || dlJob.id}</span>
        {#if dlJob.status === 'downloading'}
          {#if dlJob.total_bytes}
            {@const pct = Math.round(
              (dlJob.downloaded_bytes / dlJob.total_bytes) * 100,
            )}
            <div class="bar"><div class="fill" style="width: {pct}%"></div></div>
            <span class="model-meta">
              {pct}% · {fmtDlBytes(dlJob.downloaded_bytes)} / {fmtDlBytes(
                dlJob.total_bytes,
              )}
            </span>
          {:else}
            <span class="model-meta">{fmtDlBytes(dlJob.downloaded_bytes)} ↓</span>
          {/if}
        {:else if dlJob.status === 'done'}
          <span class="badge">✓</span>
        {:else}
          <span class="model-meta">{dlJob.error ?? ''}</span>
        {/if}
      </div>
    {/if}
  </div>

  {#if modelStore.data}
    {#if cps.length === 0 && lrs.length === 0 && packs.length === 0}
      <div class="gallery-empty">
        {t('mdl.empty')}<br />
        {t('mdl.emptyHint')}<br />
        <code>{modelStore.data.models_dir}</code>
      </div>
    {:else}
      <h4>{t('mdl.ckptTitle', { count: cps.length })}</h4>
      {#if cps.length === 0}
        <p class="hint">{t('mdl.ckptEmpty')}</p>
      {/if}
      <ul class="model-list">
        {#each cps as m (m.name)}
          <li>
            <label class="model-row">
              <input
                type="radio"
                name="active-model"
                checked={modelStore.data.active === m.name}
                onchange={() => onSelect(m.name)}
                disabled={modelStore.loading}
              />
              <span class="model-name">{m.name}</span>
              <span class="model-meta">{fmtSize(m.size_mb)}</span>
              {#if modelStore.data.active === m.name}
                <span class="badge">{t('mdl.active')}</span>
              {/if}
            </label>
            <button
              class="btn danger small"
              onclick={() => onDelete(m.name)}
              disabled={modelStore.loading || modelStore.data.active === m.name}
              title={modelStore.data.active === m.name ? t('mdl.activeDeleteTitle') : t('common.delete')}
            >
              {t('common.delete')}
            </button>
          </li>
        {/each}
      </ul>

      <h4>{t('mdl.packTitle', { count: packs.length })}</h4>
      {#if packs.length > 0}
        <ul class="model-list">
          {#each packs as p (p.name)}
            <li class="pack-row">
              <div class="pack-head">
                <span class="model-name">{p.name}</span>
                <span class="model-meta">{fmtSize(p.size_mb)}</span>
              </div>
              <ul class="pack-parts">
                {#each packRows(p) as row}
                  <li class={row.ok ? 'ok' : 'missing'}>
                    <span>{row.ok ? '✓' : '✗'} {row.label}</span>
                    {#if row.detail}<span class="model-meta">{row.detail}</span>{/if}
                  </li>
                {/each}
              </ul>
              {#if packIncomplete(p)}
                <p class="hint">{t('mdl.packMissingHint')}</p>
                <div class="dl-row">
                  <input
                    class="text-input dl-url"
                    type="url"
                    bind:value={packUrls[p.name]}
                    placeholder={t('mdl.dlUrlPh')}
                  />
                  <button
                    class="btn"
                    onclick={() => onPackDownload(p.name)}
                    disabled={!(packUrls[p.name] ?? '').trim() ||
                      (dlJob && dlJob.status === 'downloading')}
                  >
                    {t('mdl.dlStart')}
                  </button>
                </div>
              {:else}
                <p class="hint ok-line">✓ {t('mdl.packComplete')}</p>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}

      <h4>{t('mdl.loraTitle', { count: lrs.length })}</h4>
      {#if lrs.length === 0}
        <p class="hint">
          {t('mdl.loraEmpty')} <code>lora/</code> {t('mdl.putFolderHere')}
        </p>
      {/if}
      <ul class="model-list">
        {#each lrs as m (m.name)}
          {@const st = loraState(m.name)}
          <li class="lora-row">
            <label class="lora-toggle">
              <input type="checkbox" bind:checked={st.enabled} />
              <span class="model-name">{m.name}</span>
              <span class="model-meta">{fmtSize(m.size_mb)}</span>
            </label>
            <div class="lora-strength">
              <span class="hint">{t('mdl.strength')}</span>
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
              disabled={modelStore.loading}
            >
              {t('common.delete')}
            </button>
          </li>
        {/each}
      </ul>
      {#if lrs.length > 0}
        <p class="hint">
          {t('mdl.loraHint')}
        </p>
      {/if}

      <h4>{t('mdl.cnTitle', { count: cns.length })}</h4>
      {#if cns.length === 0}
        <p class="hint">
          {t('mdl.cnEmpty')} <code>controlnet/</code> {t('mdl.putSafetensors')}
        </p>
      {/if}
      <ul class="model-list">
        {#each cns as m (m.name)}
          <li class="lora-row">
            <span class="model-name">{m.name}</span>
            <span class="model-meta">{fmtSize(m.size_mb)}</span>
            <button
              class="btn danger small"
              onclick={() => onDelete(m.name)}
              disabled={modelStore.loading}
            >
              {t('common.delete')}
            </button>
          </li>
        {/each}
      </ul>
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
  .dl-box {
    background: linear-gradient(180deg, #23253f, var(--bg2));
    border: 1px solid #35385f;
    border-radius: 12px;
    padding: 0.9em 1em;
    margin-bottom: 1.2em;
  }
  .dl-box h4 { margin: 0 0 0.7em; }
  .dl-row {
    display: flex;
    gap: 0.6em;
    align-items: center;
    flex-wrap: wrap;
  }
  .dl-url { flex: 1 1 260px; min-width: 0; }
  .dl-progress {
    display: flex;
    align-items: center;
    gap: 0.7em;
    margin-top: 0.7em;
  }
  .bar {
    flex: 1;
    height: 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
    min-width: 80px;
  }
  .fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(135deg, #8aa8f5, #5a76c8);
    transition: width 0.4s;
  }
  .pack-row {
    flex-direction: column;
    align-items: stretch;
  }
  .pack-head {
    display: flex;
    align-items: center;
    gap: 0.7em;
  }
  .pack-parts {
    list-style: none;
    padding: 0.4em 0 0;
    margin: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.3em 1em;
    font-size: 0.88em;
  }
  .pack-parts li {
    display: flex;
    justify-content: space-between;
    gap: 0.5em;
    padding: 0;
    background: none;
    border: none;
    box-shadow: none;
  }
  .pack-parts li.ok { color: #9fe6a8; }
  .pack-parts li.missing { color: #f2a3a3; }
  .pack-row .hint { margin: 0.5em 0; }
  .ok-line { color: #9fe6a8; }
</style>
