// 백엔드(api/settings)에 연결 실패 시 사용하는 폴백.
// backend/atelier/settings.py의 SETTINGS와 동일하게 유지할 것.
// label/description은 i18n 키로 들고, 렌더 시점에 t()로 푼다 (언어 전환 반응형).
import { t } from './i18n.svelte.js';

export function getFallbackSettings() {
  return [
    {
      key: 'prompt',
      labelKey: 'setting.prompt',
      descKey: 'setting.promptDesc',
      recommended: '1girl, solo',
      kind: 'text',
      min: null, max: null, options: [], unit: '',
    },
    {
      key: 'negative_prompt',
      labelKey: 'setting.negative',
      descKey: 'setting.negativeDesc',
      recommended: 'worst quality, low quality, bad anatomy',
      kind: 'text',
      min: null, max: null, options: [], unit: '',
    },
    {
      key: 'steps',
      labelKey: 'setting.steps',
      descKey: 'setting.stepsDesc',
      recommended: 50,
      kind: 'number',
      min: 20, max: 50, options: [], unit: '',
    },
    {
      key: 'cfg_scale',
      labelKey: 'setting.cfg',
      descKey: 'setting.cfgDesc',
      recommended: 5.0,
      kind: 'number',
      min: 1.0, max: 9.0, options: [], unit: '',
    },
    {
      key: 'sampler',
      labelKey: 'setting.sampler',
      descKey: 'setting.samplerDesc',
      recommended: 'er_sde',
      kind: 'choice',
      min: null, max: null,
      options: ['er_sde', 'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_karras', 'dpmpp_2s_ancestral', 'dpmpp_2s_ancestral_karras', 'dpmpp_sde', 'dpmpp_sde_karras', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_2m_sde_karras', 'dpmpp_2m_sde_heun', 'dpmpp_2m_sde_heun_karras', 'dpmpp_3m_sde', 'dpmpp_3m_sde_karras', 'dpm2', 'dpm2_ancestral', 'dpm_fast', 'dpm_adaptive', 'lms', 'heun', 'ddim', 'plms', 'unipc', 'restart', 'sa_solver', 'deis', 'lcm', 'tcd'],
      unit: '',
    },
    {
      key: 'width',
      labelKey: 'setting.width',
      descKey: 'setting.widthDesc',
      recommended: 832,
      kind: 'number',
      min: 256, max: 2048, options: [], unit: 'px',
    },
    {
      key: 'height',
      labelKey: 'setting.height',
      descKey: 'setting.heightDesc',
      recommended: 1216,
      kind: 'number',
      min: 256, max: 2048, options: [], unit: 'px',
    },
    {
      key: 'seed',
      labelKey: 'setting.seed',
      descKey: 'setting.seedDesc',
      recommended: -1,
      kind: 'number',
      min: -1, max: null, options: [], unit: '',
    },
    {
      key: 'denoise_strength',
      labelKey: 'setting.denoise',
      descKey: 'setting.denoiseDesc',
      recommended: 0.75,
      kind: 'number',
      min: 0.0, max: 1.0, options: [], unit: '',
    },
    {
      key: 'hires_fix_enabled',
      labelKey: 'setting.hiresUse',
      descKey: 'setting.hiresUseDesc',
      recommended: false,
      kind: 'toggle',
      min: null, max: null, options: [], unit: '',
    },
    {
      key: 'hires_fix_scale',
      labelKey: 'setting.hiresScale',
      descKey: 'setting.hiresScaleDesc',
      recommended: 2.0,
      kind: 'number',
      min: 1.25, max: 3.0, options: [], unitKey: 'setting.hiresScaleUnit',
    },
    {
      key: 'hires_fix_denoise',
      labelKey: 'setting.hiresDenoise',
      descKey: 'setting.hiresDenoiseDesc',
      recommended: 0.35,
      kind: 'number',
      min: 0.1, max: 0.7, options: [], unit: '',
    },
    {
      key: 'batch_size',
      labelKey: 'setting.batchSize',
      descKey: 'setting.batchSizeDesc',
      recommended: 1,
      kind: 'number',
      min: 1, max: 8, options: [], unitKey: 'setting.batchSizeUnit',
    },
    {
      key: 'controlnet_model',
      labelKey: 'setting.cnModel',
      descKey: 'setting.cnModelDesc',
      recommended: '',
      kind: 'text',
      min: null, max: null, options: [], unit: '',
    },
    {
      key: 'controlnet_strength',
      labelKey: 'setting.cnStrength',
      descKey: 'setting.cnStrengthDesc',
      recommended: 0.8,
      kind: 'number',
      min: 0.0, max: 1.0, options: [], unit: '',
    },
  ];
}

// 하위 호환: 기존처럼 정적 배열이 필요하면 getFallbackSettings() 호출 결과를 쓴다.
export const FALLBACK_SETTINGS = getFallbackSettings();
