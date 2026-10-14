<script>
  // 오른쪽 인스펙터: 기본 모드(도우미·모델·해상도) + 고급 설정(접기)
  // 모든 컨트롤은 store.values에 바인딩되어 생성 요청에 그대로 반영된다.
  // hideHelper: 편집 화면처럼 프롬프트 입력이 탭 안에 있을 때 도우미를 숨긴다.
  let { hideHelper = false } = $props();

  import { store } from './store.svelte.js';
  import PromptHelper from './PromptHelper.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import QuickSlider from './QuickSlider.svelte';
  import ResolutionPicker from './ResolutionPicker.svelte';
  import SettingField from './SettingField.svelte';

  const SAMPLER_NAMES = {
    er_sde: 'ER SDE (추천)',
    dpmpp_2m_sde_gpu: 'DPM++ 2M SDE (GPU)',
    dpmpp_2m_sde: 'DPM++ 2M SDE',
    euler_ancestral: 'Euler Ancestral',
    dpmpp_2m_karras: 'DPM++ 2M Karras',
    ddim: 'DDIM',
    restart: 'Restart (실험실)',
    sa_solver: 'SA-Solver (실험실)',
    deis: 'DEIS (실험실)',
    lcm: 'LCM (실험실)',
    tcd: 'TCD (실험실)',
  };
  const LAB_SAMPLERS = new Set(['restart', 'sa_solver', 'deis', 'lcm', 'tcd']);

  const samplerMeta = $derived(store.meta.find((s) => s.key === 'sampler'));
  const samplerOptions = $derived(samplerMeta?.options ?? []);
  const cfgMeta = $derived(store.meta.find((s) => s.key === 'cfg_scale'));
  const stepsMeta = $derived(store.meta.find((s) => s.key === 'steps'));

  // 기본 모드에 보이는 키: 프롬프트(중앙)·도우미·모델·해상도
  // 고급 설정에 들어가는 키: 샘플러·CFG·스텝 + 나머지 전부
  const BASIC_KEYS = new Set(['prompt', 'negative_prompt', 'width', 'height']);
  const advancedSettings = $derived(
    store.meta.filter((s) => !BASIC_KEYS.has(s.key))
  );

  function samplerName(opt) {
    return SAMPLER_NAMES[opt] ?? opt;
  }
</script>

{#if !hideHelper}
  <PromptHelper />
{/if}

<div class="qp-section">
  <div class="qp-title">🖼 모델 선택</div>
  <ModelSelect />
</div>

<div class="qp-section">
  <div class="qp-title">🖼 해상도</div>
  <ResolutionPicker />
</div>

<details class="qp-section qp-details">
  <summary class="qp-title">⚙ 고급 설정</summary>
  <div class="qp-advanced">
    <div class="qp-sub">🎲 샘플링 방법</div>
    <select class="select" bind:value={store.values.sampler} aria-label="샘플링 방법">
      <optgroup label="일반">
        {#each samplerOptions.filter((o) => !LAB_SAMPLERS.has(o)) as opt (opt)}
          <option value={opt}>{samplerName(opt)}</option>
        {/each}
      </optgroup>
      <optgroup label="실험실">
        {#each samplerOptions.filter((o) => LAB_SAMPLERS.has(o)) as opt (opt)}
          <option value={opt}>{samplerName(opt)}</option>
        {/each}
      </optgroup>
    </select>
    {#if cfgMeta}
      <QuickSlider setting={cfgMeta} />
    {/if}
    {#if stepsMeta}
      <QuickSlider setting={stepsMeta} />
    {/if}
    {#each advancedSettings.filter((s) => !['sampler', 'cfg_scale', 'steps'].includes(s.key)) as setting (setting.key)}
      <SettingField {setting} bind:value={store.values[setting.key]} />
    {/each}
  </div>
</details>
