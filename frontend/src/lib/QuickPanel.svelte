<script>
  // 오른쪽 설정 퀵 패널: 도우미 + 모델 + 해상도 + 샘플러 + CFG + 스텝 + 고급 설정
  // 모든 컨트롤은 store.values에 바인딩되어 생성 요청에 그대로 반영된다.
  import { store } from './store.svelte.js';
  import PromptHelper from './PromptHelper.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import QuickSlider from './QuickSlider.svelte';
  import SettingField from './SettingField.svelte';

  const RES_PRESETS = [
    { label: '512×512', w: 512, h: 512 },
    { label: '768×768', w: 768, h: 768 },
    { label: '1024×1024', w: 1024, h: 1024 },
    { label: '832×1216', w: 832, h: 1216, tag: '세로·추천' },
  ];

  const SAMPLER_NAMES = {
    er_sde: 'ER SDE (추천)',
    dpmpp_2m_sde_gpu: 'DPM++ 2M SDE (GPU)',
    dpmpp_2m_sde: 'DPM++ 2M SDE',
    euler_ancestral: 'Euler Ancestral',
    dpmpp_2m_karras: 'DPM++ 2M Karras',
    ddim: 'DDIM',
  };

  const samplerMeta = $derived(store.meta.find((s) => s.key === 'sampler'));
  const samplerOptions = $derived(samplerMeta?.options ?? []);
  const cfgMeta = $derived(store.meta.find((s) => s.key === 'cfg_scale'));
  const stepsMeta = $derived(store.meta.find((s) => s.key === 'steps'));

  // 고급 설정: 프롬프트(중앙 패널)·퀵 패널 항목 제외, 나머지는 SettingField 재사용
  const QUICK_KEYS = new Set([
    'prompt', 'negative_prompt',
    'width', 'height', 'sampler', 'cfg_scale', 'steps',
  ]);
  const advancedSettings = $derived(
    store.meta.filter((s) => !QUICK_KEYS.has(s.key))
  );

  function setRes(p) {
    store.values.width = p.w;
    store.values.height = p.h;
  }

  function isResActive(p) {
    return store.values.width === p.w && store.values.height === p.h;
  }

  function samplerName(opt) {
    return SAMPLER_NAMES[opt] ?? opt;
  }
</script>

<PromptHelper />

<div class="qp-section">
  <div class="qp-title">🖼 모델 선택</div>
  <ModelSelect />
</div>

<div class="qp-section">
  <div class="qp-title">🖼 해상도</div>
  <div class="seg-grid">
    {#each RES_PRESETS as p (p.label)}
      <button
        class="seg-btn"
        class:active={isResActive(p)}
        onclick={() => setRes(p)}
        title={p.tag ? `${p.label} (${p.tag})` : p.label}
      >
        <span>{p.label}</span>
        {#if p.tag}<span class="seg-tag">{p.tag}</span>{/if}
      </button>
    {/each}
  </div>
</div>

<div class="qp-section">
  <div class="qp-title">🎲 샘플링 방법</div>
  <select class="select" bind:value={store.values.sampler} aria-label="샘플링 방법">
    {#each samplerOptions as opt (opt)}
      <option value={opt}>{samplerName(opt)}</option>
    {/each}
  </select>
</div>

<div class="qp-section">
  {#if cfgMeta}
    <QuickSlider setting={cfgMeta} />
  {/if}
</div>

<div class="qp-section">
  {#if stepsMeta}
    <QuickSlider setting={stepsMeta} />
  {/if}
</div>

<details class="qp-section qp-details">
  <summary class="qp-title">⚙ 고급 설정</summary>
  <div class="qp-advanced">
    {#each advancedSettings as setting (setting.key)}
      <SettingField {setting} bind:value={store.values[setting.key]} />
    {/each}
  </div>
</details>
