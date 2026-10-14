<script>
  import { onMount } from 'svelte';
  import { store, initStore, reconnect } from './lib/store.svelte.js';
  import SideNav from './lib/SideNav.svelte';
  import QuickPanel from './lib/QuickPanel.svelte';
  import HistoryTab from './lib/HistoryTab.svelte';
  import LibraryTab from './lib/LibraryTab.svelte';
  import TemplatesTab from './lib/TemplatesTab.svelte';
  import GenerateTab from './lib/GenerateTab.svelte';
  import InpaintTab from './lib/InpaintTab.svelte';
  import ModelsTab from './lib/ModelsTab.svelte';
  import SettingsPanel from './lib/SettingsPanel.svelte';

  // 왼쪽 내비: create(작업공간) | library | templates | history
  let nav = $state('create');
  // 가운데 탭: generate | inpaint | models | settings
  let tab = $state('generate');

  const tabs = [
    { id: 'generate', label: '생성' },
    { id: 'inpaint', label: '인페인트' },
    { id: 'models', label: '모델' },
    { id: 'settings', label: '설정' },
  ];

  // 사이드바 하이라이트: 설정 탭이 열려 있으면 '설정'을 활성화
  const sideActive = $derived(
    nav !== 'create' ? nav : tab === 'settings' ? 'settings' : 'create'
  );

  function onNav(id) {
    if (id === 'settings') {
      nav = 'create';
      tab = 'settings';
    } else {
      nav = id;
    }
  }

  /** 작업내역/라이브러리의 "설정 불러오기": 기록의 params를 스토어에 반영하고 생성 탭으로 이동 */
  function applyRecordParams(params) {
    if (params) {
      store.values = { ...store.values, ...params };
    }
    nav = 'create';
    tab = 'generate';
  }

  onMount(() => {
    initStore();
  });
</script>

<div class="app-layout">
  <SideNav nav={sideActive} onSelect={onNav} />

  <div class="app-main">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">✦</span>
        <h1>Atelier AI</h1>
        <span class="tagline">AI로 만드는 나만의 창작공간</span>
      </div>
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

    {#if nav === 'create'}
      <div class="workspace">
        <div class="center-col">
          <nav class="tabs" aria-label="작업 탭">
            {#each tabs as t (t.id)}
              <button
                class:active={tab === t.id}
                onclick={() => (tab = t.id)}
              >
                {t.label}
              </button>
            {/each}
          </nav>

          {#if tab === 'generate'}
            <GenerateTab />
          {:else if tab === 'inpaint'}
            <InpaintTab />
          {:else if tab === 'models'}
            <ModelsTab />
          {:else}
            <SettingsPanel />
          {/if}
        </div>
        <aside class="quick-panel" aria-label="설정 퀵 패널">
          <QuickPanel />
        </aside>
      </div>
    {:else if nav === 'library'}
      <LibraryTab onLoad={applyRecordParams} />
    {:else if nav === 'templates'}
      <TemplatesTab />
    {:else if nav === 'history'}
      <HistoryTab onLoad={applyRecordParams} />
    {/if}
  </div>
</div>
