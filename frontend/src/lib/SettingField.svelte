<script>
  import { resetOne } from './store.svelte.js';
  // setting: 설정 메타, value: 현재값 (bindable)
  let { setting, value = $bindable() } = $props();

  const isInt = $derived(
    Number.isInteger(setting.min ?? 0) && Number.isInteger(setting.max ?? 0)
  );
  const step = $derived(isInt ? 1 : 0.05);

  function onSlider(e) {
    const v = parseFloat(e.target.value);
    value = isInt ? Math.round(v) : v;
  }

  function onNumber(e) {
    const v = isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
    if (!Number.isNaN(v)) value = v;
  }
</script>

<div class="field">
  <div class="field-head">
    <span class="field-label">{setting.label}</span>
    <button class="link-btn" onclick={() => resetOne(setting.key)} title="추천값으로 되돌리기">
      추천값으로
    </button>
  </div>
  <p class="field-desc">{setting.description}</p>

  {#if setting.kind === 'number'}
    <div class="number-row">
      <input
        type="range"
        min={setting.min ?? 0}
        max={setting.max ?? 100}
        {step}
        value={value ?? setting.recommended}
        oninput={onSlider}
      />
      <input
        type="number"
        min={setting.min}
        max={setting.max}
        {step}
        value={value ?? setting.recommended}
        oninput={onNumber}
        class="number-box"
      />
    </div>
    <p class="recommended">추천값: {setting.recommended}{setting.unit ? ` ${setting.unit}` : ''}</p>
  {:else if setting.kind === 'choice'}
    <select bind:value class="select">
      {#each setting.options as opt}
        <option value={opt}>{opt}</option>
      {/each}
    </select>
    <p class="recommended">추천값: {setting.recommended}</p>
  {:else if setting.kind === 'toggle'}
    <label class="switch">
      <input type="checkbox" bind:checked={value} />
      <span class="track"><span class="thumb"></span></span>
      <span>{value ? '켬' : '끔'}</span>
    </label>
    <p class="recommended">추천값: {setting.recommended ? '켬' : '끔'}</p>
  {:else}
    <!-- kind === 'text': 일반 텍스트는 textarea (프롬프트 계열은 TagInput 사용) -->
    <textarea bind:value rows="3" class="textarea"></textarea>
    <p class="recommended">추천값: {setting.recommended}</p>
  {/if}
</div>
