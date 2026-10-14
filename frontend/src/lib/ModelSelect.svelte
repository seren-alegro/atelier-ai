<script>
  // 퀵 패널용 모델 선택 드롭다운 (체크포인트 전환)
  import { onMount } from 'svelte';
  import { modelStore, loadModels, activateModel, checkpoints } from './models.svelte.js';

  const cps = $derived(checkpoints());

  onMount(() => {
    if (!modelStore.data && !modelStore.loading) loadModels();
  });

  function onChange(e) {
    if (e.target.value) activateModel(e.target.value);
  }
</script>

{#if modelStore.notice && !modelStore.data}
  <p class="hint">{modelStore.notice}</p>
{/if}
<select
  class="select"
  value={modelStore.data?.active ?? ''}
  onchange={onChange}
  disabled={modelStore.loading || cps.length === 0}
  aria-label="모델 선택"
>
  {#if cps.length === 0}
    <option value="">모델 없음</option>
  {/if}
  {#each cps as m (m.name)}
    <option value={m.name}>{m.name}</option>
  {/each}
</select>
{#if modelStore.loading}
  <p class="hint">전환 중…</p>
{/if}
