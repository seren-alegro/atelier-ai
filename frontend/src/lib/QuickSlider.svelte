<script>
  // 퀵 패널용 간결 숫자 컨트롤 (슬라이더 + 숫자박스)
  import { store } from './store.svelte.js';

  let { setting } = $props();

  const isInt = $derived(
    Number.isInteger(setting.min ?? 0) && Number.isInteger(setting.max ?? 0)
  );
  const step = $derived(isInt ? 1 : 0.1);
  const val = $derived(store.values[setting.key] ?? setting.recommended);

  function onSlider(e) {
    const v = parseFloat(e.target.value);
    if (!Number.isNaN(v)) {
      store.values[setting.key] = isInt ? Math.round(v) : Math.round(v * 10) / 10;
    }
  }

  function onNumber(e) {
    const v = parseFloat(e.target.value);
    if (!Number.isNaN(v)) {
      store.values[setting.key] = isInt ? Math.round(v) : v;
    }
  }
</script>

<div class="quick-slider">
  <div class="quick-slider-head">
    <span class="quick-label">{setting.label}</span>
    <input
      type="number"
      class="number-box qs-num"
      min={setting.min}
      max={setting.max}
      {step}
      value={val}
      oninput={onNumber}
      aria-label={setting.label}
    />
  </div>
  <input
    type="range"
    min={setting.min ?? 0}
    max={setting.max ?? 100}
    {step}
    value={val}
    oninput={onSlider}
    aria-label={setting.label}
  />
</div>
