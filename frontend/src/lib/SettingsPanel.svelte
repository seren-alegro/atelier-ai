<script>
  import { store, resetAll } from './store.svelte.js';
  import SettingField from './SettingField.svelte';
  import TagInput from './TagInput.svelte';
  import UpdatePanel from './UpdatePanel.svelte';

  // 프롬프트 계열은 칩 입력 UI 사용
  const TEXT_AS_TAGS = new Set(['prompt', 'negative_prompt']);
</script>

<div class="settings-panel">
  <div class="panel-head">
    <h2>설정</h2>
    <button class="btn" onclick={resetAll}>전체 추천값으로 되돌리기</button>
  </div>

  {#if !store.loaded}
    <p>설정을 불러오는 중…</p>
  {:else}
    {#each store.meta as setting (setting.key)}
      {#if TEXT_AS_TAGS.has(setting.key)}
        <div class="field">
          <div class="field-head">
            <span class="field-label">{setting.label}</span>
          </div>
          <p class="field-desc">{setting.description}</p>
          <TagInput bind:value={store.values[setting.key]} />
          <p class="recommended">추천값: {setting.recommended}</p>
        </div>
      {:else}
        <SettingField {setting} bind:value={store.values[setting.key]} />
      {/if}
    {/each}
  {/if}

  <UpdatePanel />
</div>
