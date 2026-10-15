<script>
  import { store, resetAll } from './store.svelte.js';
  import SettingField from './SettingField.svelte';
  import TagInput from './TagInput.svelte';
  import UpdatePanel from './UpdatePanel.svelte';
  import { t, getLang, setLang, LANGS, metaLabel, metaDesc } from './i18n.svelte.js';

  // 프롬프트 계열은 칩 입력 UI 사용
  const TEXT_AS_TAGS = new Set(['prompt', 'negative_prompt']);

  function fieldLabel(setting) {
    return setting.labelKey ? t(setting.labelKey) : metaLabel(setting);
  }
  function fieldDesc(setting) {
    return setting.descKey ? t(setting.descKey) : metaDesc(setting);
  }
</script>

<div class="settings-panel">
  <div class="panel-head">
    <h2>{t('set.title')}</h2>
    <button class="btn" onclick={resetAll}>{t('set.resetAll')}</button>
  </div>

  {#if !store.loaded}
    <p>{t('set.loading')}</p>
  {:else}
    {#each store.meta as setting (setting.key)}
      {#if TEXT_AS_TAGS.has(setting.key)}
        <div class="field">
          <div class="field-head">
            <span class="field-label">{fieldLabel(setting)}</span>
          </div>
          <p class="field-desc">{fieldDesc(setting)}</p>
          <TagInput bind:value={store.values[setting.key]} />
          <p class="recommended">{t('common.recommended')} {setting.recommended}</p>
        </div>
      {:else}
        <SettingField {setting} bind:value={store.values[setting.key]} />
      {/if}
    {/each}
  {/if}

  <div class="lang-section">
    <div class="field-head">
      <span class="field-label">{t('set.langTitle')}</span>
    </div>
    <p class="field-desc">{t('set.langDesc')}</p>
    <div class="lang-btns">
      {#each LANGS as l (l.id)}
        <button
          class="btn"
          class:active={getLang() === l.id}
          onclick={() => setLang(l.id)}
        >{l.label}</button>
      {/each}
    </div>
  </div>

  <UpdatePanel />
</div>

<style>
  .lang-section {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border, #e5e5e5);
  }
  .lang-btns {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  .lang-btns .btn.active {
    border-color: var(--accent);
    color: #cdd9ff;
    background: linear-gradient(180deg, rgba(122, 162, 247, 0.22), rgba(122, 162, 247, 0.1));
  }
</style>
