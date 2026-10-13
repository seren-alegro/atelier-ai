<script>
  import { onMount } from 'svelte';
  import { store, initStore } from './lib/store.svelte.js';
  import GenerateTab from './lib/GenerateTab.svelte';
  import InpaintTab from './lib/InpaintTab.svelte';
  import SettingsPanel from './lib/SettingsPanel.svelte';

  let tab = $state('generate');
  const tabs = [
    { id: 'generate', label: '생성' },
    { id: 'inpaint', label: '인페인트' },
    { id: 'settings', label: '설정' },
  ];

  onMount(() => {
    initStore();
  });
</script>

<header class="app-header">
  <h1>Atelier AI</h1>
  <span class="api-badge" class:online={store.apiOnline} class:offline={store.loaded && !store.apiOnline}>
    {#if !store.loaded}
      연결 확인 중…
    {:else if store.apiOnline}
      백엔드 연결됨
    {:else}
      백엔드 없음 · 기본값 사용 중
    {/if}
  </span>
</header>

<nav class="tabs">
  {#each tabs as t}
    <button class:active={tab === t.id} onclick={() => (tab = t.id)}>
      {t.label}
    </button>
  {/each}
</nav>

<main>
  {#if tab === 'generate'}
    <GenerateTab />
  {:else if tab === 'inpaint'}
    <InpaintTab />
  {:else}
    <SettingsPanel />
  {/if}
</main>
