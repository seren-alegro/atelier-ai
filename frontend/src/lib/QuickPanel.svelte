<script>
  // 오른쪽 인스펙터: 기본 모드(도우미·모델·해상도) + 고급 설정(접기)
  // 모든 컨트롤은 store.values에 바인딩되어 생성 요청에 그대로 반영된다.
  // hideHelper: 편집 화면처럼 프롬프트 입력이 탭 안에 있을 때 도우미를 숨긴다.
  let { hideHelper = false } = $props();

  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';
  import { modelStore, loadModels, loras } from './models.svelte.js';
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

  // LoRA 바로 선택 (모델 선택 바로 아래). ModelsTab.svelte의 loraState 패턴 재사용.
  const lrs = $derived(loras());
  function loraState(name) {
    if (!store.values.loras[name]) {
      store.values.loras[name] = { enabled: false, strength: 0.8 };
    }
    return store.values.loras[name];
  }

  // 업스케일 강도 (갤러리 업스케일 버튼용). QuickSlider용 설정 디스크립터.
  const upscaleSetting = {
    key: 'upscale_strength',
    label: '업스케일 강도',
    min: 0.5,
    max: 1.0,
    recommended: 0.5,
  };

  // Hires Fix 섹션 (고급 영역). QuickSlider용 설정 디스크립터.
  const hiresFixScaleSetting = {
    key: 'hires_fix_scale',
    label: 'Hires Fix 배율',
    min: 1.25,
    max: 3.0,
    recommended: 2.0,
  };
  const hiresFixDenoiseSetting = {
    key: 'hires_fix_denoise',
    label: 'Hires Fix 디노이즈',
    min: 0.1,
    max: 0.7,
    recommended: 0.35,
  };
  const HIRES_FIX_KEYS = new Set([
    'hires_fix_enabled', 'hires_fix_scale', 'hires_fix_denoise',
  ]);

  onMount(() => {
    if (!modelStore.data && !modelStore.loading) loadModels();
  });
</script>

{#if !hideHelper}
  <PromptHelper />
{/if}

<div class="qp-section">
  <div class="qp-title">🖼 모델 선택</div>
  <ModelSelect />
</div>

<div class="qp-section">
  <div class="qp-title">🧩 LoRA 바로 선택</div>
  {#if modelStore.loading}
    <p class="hint">불러오는 중…</p>
  {:else if lrs.length === 0}
    <p class="hint">LoRA가 없어요. 모델 탭에서 확인해주세요.</p>
  {:else}
    <ul class="qp-lora-list">
      {#each lrs as m (m.name)}
        {@const st = loraState(m.name)}
        <li class="qp-lora-row">
          <label class="qp-lora-toggle">
            <input type="checkbox" bind:checked={st.enabled} />
            <span class="qp-lora-name" title={m.name}>{m.name}</span>
          </label>
          <div class="qp-lora-strength">
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.05"
              bind:value={st.strength}
              disabled={!st.enabled}
              aria-label="{m.name} 강도"
            />
            <span class="qp-lora-val">{st.strength.toFixed(2)}</span>
          </div>
        </li>
      {/each}
    </ul>
    <p class="hint">활성화한 LoRA는 생성 시 함께 적용돼요.</p>
  {/if}
</div>

<div class="qp-section">
  <div class="qp-title">🖼 해상도</div>
  <ResolutionPicker />
</div>

<div class="qp-section">
  <div class="qp-title">🔍 업스케일 강도</div>
  <QuickSlider setting={upscaleSetting} />
  <p class="hint">
    갤러리의 업스케일 버튼에 적용돼요. 1이면 RealESRGAN 결과를 그대로 쓰고,
    0.5면 원본을 부드럽게 살려서 절반만 섞어요.
  </p>
</div>

<details class="qp-section qp-details">
  <summary class="qp-title">⚙ 고급 설정</summary>
  <div class="qp-advanced">
    <div class="qp-sub">✨ Hires Fix</div>
    <label class="qp-lora-toggle">
      <input type="checkbox" bind:checked={store.values.hires_fix_enabled} />
      <span>고해상도 보정 사용</span>
    </label>
    <p class="hint">
      작은 해상도로 먼저 뽑고 키운 뒤 살짝만 다시 그려요.
      시간이 약 2배 걸리지만 잔붕괴가 줄어요.
    </p>
    <QuickSlider setting={hiresFixScaleSetting} />
    <QuickSlider setting={hiresFixDenoiseSetting} />
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
    {#each advancedSettings.filter((s) => !['sampler', 'cfg_scale', 'steps', ...HIRES_FIX_KEYS].includes(s.key)) as setting (setting.key)}
      <SettingField {setting} bind:value={store.values[setting.key]} />
    {/each}
  </div>
</details>

<style>
  .qp-lora-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .qp-lora-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }
  .qp-lora-toggle input[type='checkbox'] {
    accent-color: var(--accent);
    width: 16px;
    height: 16px;
    flex-shrink: 0;
    cursor: pointer;
  }
  .qp-lora-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
    font-weight: 500;
  }
  .qp-lora-strength {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
    padding-left: 24px;
  }
  .qp-lora-strength input[type='range'] {
    flex: 1;
    accent-color: var(--accent);
  }
  .qp-lora-val {
    color: var(--muted, #999);
    font-size: 12px;
    min-width: 32px;
    text-align: right;
  }
</style>
