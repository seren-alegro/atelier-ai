<script>
  import { resetOne } from './store.svelte.js';
  import { t } from './i18n.svelte.js';
  // setting: 설정 메타, value: 현재값 (bindable)
  // label/description은 labelKey/descKey가 있으면 t()로 푼다 (폴백 메타용).
  let { setting, value = $bindable() } = $props();

  const label = $derived(setting.labelKey ? t(setting.labelKey) : setting.label);
  const description = $derived(setting.descKey ? t(setting.descKey) : setting.description);
  const unit = $derived(setting.unitKey ? t(setting.unitKey) : setting.unit);

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
    <span class="field-label">{label}</span>
    <button class="link-btn" onclick={() => resetOne(setting.key)} title={t('common.recommendedResetTitle')}>
      {t('common.recommendedReset')}
    </button>
  </div>
  <p class="field-desc">{description}</p>

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
    <p class="recommended">{t('common.recommended')} {setting.recommended}{unit ? ` ${unit}` : ''}</p>
  {:else if setting.kind === 'choice'}
    <select bind:value class="select">
      {#each setting.options as opt}
        <option value={opt}>{opt}</option>
      {/each}
    </select>
    <p class="recommended">{t('common.recommended')} {setting.recommended}</p>
  {:else if setting.kind === 'toggle'}
    <label class="switch">
      <input type="checkbox" bind:checked={value} />
      <span class="track"><span class="thumb"></span></span>
      <span>{value ? t('common.on') : t('common.off')}</span>
    </label>
    <p class="recommended">{t('common.recommended')} {setting.recommended ? t('common.on') : t('common.off')}</p>
  {:else}
    <!-- kind === 'text': 일반 텍스트는 textarea (프롬프트 계열은 TagInput 사용) -->
    <textarea bind:value rows="3" class="textarea"></textarea>
    <p class="recommended">{t('common.recommended')} {setting.recommended}</p>
  {/if}
</div>
