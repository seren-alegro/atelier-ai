<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import { t } from './i18n.svelte.js';

  const STORAGE_KEY = 'atelier-ai-templates';

  // 내장 템플릿 (삭제 불가, 항상 표시)
  const BUILTINS = $derived([
    {
      name: t('tpl.builtin1Name'),
      desc: t('tpl.builtin1Desc'),
      builtin: true,
      values: {
        prompt:
          'masterpiece, best quality, highres, absurdres, newest, detailed eyes, ',
        negative_prompt: 'worst quality, low quality, bad anatomy, ',
        steps: 28,
        cfg_scale: 5.0,
        sampler: 'dpmpp_2m_karras',
        width: 832,
        height: 1216,
        seed: -1,
        batch_size: 1,
      },
    },
    {
      name: t('tpl.builtin2Name'),
      desc: t('tpl.builtin2Desc'),
      builtin: true,
      values: {
        prompt:
          'masterpiece, portrait, close-up, looking at viewer, detailed face, beautiful detailed eyes, ',
        negative_prompt: 'worst quality, low quality, bad anatomy, ',
        steps: 28,
        cfg_scale: 5.0,
        sampler: 'er_sde',
        width: 768,
        height: 1024,
        seed: -1,
        batch_size: 1,
      },
    },
  ]);

  let userTemplates = $state([]);
  let name = $state('');
  let notice = $state('');

  function readStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  }

  function writeStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userTemplates));
    } catch {
      notice = t('tpl.saveFail');
    }
  }

  function saveCurrent() {
    const n = name.trim();
    if (!n) {
      notice = t('tpl.needName');
      return;
    }
    // 깊은 복사로 현재 설정값을 통째로 저장
    const values = JSON.parse(JSON.stringify(store.values));
    userTemplates = [
      { name: n, desc: t('tpl.myTemplate'), builtin: false, values },
      ...userTemplates.filter((t) => t.name !== n),
    ];
    writeStorage();
    name = '';
    notice = t('tpl.saved', { name: n });
  }

  function applyTemplate(tpl) {
    store.values = { ...store.values, ...tpl.values };
    notice = t('tpl.applied', { name: tpl.name });
  }

  function deleteTemplate(tpl) {
    if (!confirm(t('tpl.deleteConfirm', { name: tpl.name }))) return;
    userTemplates = userTemplates.filter((t) => t.name !== tpl.name);
    writeStorage();
  }

  function fmtValues(values) {
    const bits = [];
    if (values.width && values.height)
      bits.push(`${values.width}×${values.height}`);
    if (values.steps) bits.push(t('common.stepsCount', { n: values.steps }));
    if (values.cfg_scale) bits.push(`CFG ${values.cfg_scale}`);
    if (values.sampler) bits.push(values.sampler);
    return bits.join(' · ');
  }

  onMount(() => {
    userTemplates = readStorage();
  });
</script>

<div class="templates-tab">
  <div class="tab-head">
    <h3>{t('tpl.title')}</h3>
  </div>
  {#if notice}<p class="notice">{notice}</p>{/if}

  <section class="save-box">
    <h4>{t('tpl.saveTitle')}</h4>
    <p class="hint">{t('tpl.saveHint')}</p>
    <div class="save-row">
      <input
        class="text-input"
        type="text"
        bind:value={name}
        placeholder={t('tpl.namePlaceholder')}
        maxlength="40"
        onkeydown={(e) => e.key === 'Enter' && saveCurrent()}
      />
      <button class="btn primary" onclick={saveCurrent}>{t('common.save')}</button>
    </div>
  </section>

  <section>
    <h4>{t('tpl.builtinTitle')}</h4>
    <div class="tpl-grid">
      {#each BUILTINS as tpl (tpl.name)}
        <article class="tpl-card">
          <h5>{tpl.name}</h5>
          <p class="hint">{tpl.desc}</p>
          <p class="meta dim">{fmtValues(tpl.values)}</p>
          <p class="prompt-preview" title={tpl.values.prompt}>
            {tpl.values.prompt}
          </p>
          <button class="btn small" onclick={() => applyTemplate(tpl)}>
            {t('common.apply')}
          </button>
        </article>
      {/each}
    </div>
  </section>

  <section>
    <h4>{t('tpl.myTitle', { count: userTemplates.length })}</h4>
    {#if userTemplates.length === 0}
      <div class="gallery-empty">
        {t('tpl.empty')}
      </div>
    {:else}
      <div class="tpl-grid">
        {#each userTemplates as tpl (tpl.name)}
          <article class="tpl-card">
            <h5>{tpl.name}</h5>
            <p class="meta dim">{fmtValues(tpl.values)}</p>
            <p class="prompt-preview" title={tpl.values.prompt}>
              {tpl.values.prompt || t('hist.noPrompt')}
            </p>
            <div class="card-actions">
              <button class="btn small" onclick={() => applyTemplate(tpl)}>
                {t('common.apply')}
              </button>
              <button
                class="btn small ghost danger"
                onclick={() => deleteTemplate(tpl)}
              >
                {t('common.delete')}
              </button>
            </div>
          </article>
        {/each}
      </div>
    {/if}
  </section>
</div>

<style>
  .templates-tab {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .tab-head h3 {
    margin: 0;
  }

  .templates-tab h4 {
    margin: 0 0 10px;
    font-size: 14px;
  }

  .save-box {
    background: linear-gradient(180deg, var(--bg2), var(--bg));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
  }

  .save-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }

  .save-row .text-input {
    flex: 1;
  }

  .tpl-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }

  .tpl-card {
    background: linear-gradient(180deg, var(--bg2), var(--bg));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }

  .tpl-card h5 {
    margin: 0 0 4px;
    font-size: 14px;
  }

  .prompt-preview {
    font-size: 12px;
    color: var(--muted);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin: 8px 0 10px;
    word-break: keep-all;
  }

  .meta.dim {
    font-size: 12px;
    color: var(--muted);
    opacity: 0.75;
    margin: 4px 0 0;
  }

  .card-actions {
    display: flex;
    gap: 8px;
  }

  .btn.small {
    font-size: 12px;
    padding: 6px 10px;
  }

  .btn.ghost {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
  }

  .btn.ghost.danger:hover {
    color: #ff7a90;
    border-color: #ff7a90;
  }
</style>
