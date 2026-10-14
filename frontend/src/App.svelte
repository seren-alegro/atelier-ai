<script>
  import { onMount } from 'svelte';
  import { store, initStore, reconnect } from './lib/store.svelte.js';
  import GenerateTab from './lib/GenerateTab.svelte';
  import InpaintTab from './lib/InpaintTab.svelte';
  import ModelsTab from './lib/ModelsTab.svelte';
  import SettingsPanel from './lib/SettingsPanel.svelte';

  let tab = $state('generate');
  const tabs = [
    { id: 'generate', label: '생성' },
    { id: 'inpaint', label: '인페인트' },
    { id: 'models', label: '모델' },
    { id: 'settings', label: '설정' },
  ];

  onMount(() => {
    initStore();
  });
</script>

<header class="app-header">
  <h1>Atelier AI</h1>
  <button
    class="api-badge"
    class:online={store.apiOnline}
    class:offline={store.loaded && !store.apiOnline}
    onclick={() => { if (store.loaded && !store.apiOnline) reconnect(); }}
    title={store.loaded && !store.apiOnline ? '클릭해서 다시 연결' : ''}
  >
    {#if !store.loaded}
      연결 확인 중…
    {:else if store.apiOnline}
      백엔드 연결됨
    {:else}
      백엔드 없음 · 기본값 사용 중 (클릭해서 재시도)
    {/if}
  </button>
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
  {:else if tab === 'models'}
    <ModelsTab />
  {:else}
    <SettingsPanel />
  {/if}
</main>
