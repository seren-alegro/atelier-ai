<script>
  import { onMount } from 'svelte';
  import { store, initStore, reconnect, checkBackendAlive } from './lib/store.svelte.js';
  import SideNav from './lib/SideNav.svelte';
  import QuickPanel from './lib/QuickPanel.svelte';
  import HistoryTab from './lib/HistoryTab.svelte';
  import LibraryTab from './lib/LibraryTab.svelte';
  import TemplatesTab from './lib/TemplatesTab.svelte';
  import GenerateTab from './lib/GenerateTab.svelte';
  import InpaintTab from './lib/InpaintTab.svelte';
  import PoseTab from './lib/PoseTab.svelte';
  import ModelsTab from './lib/ModelsTab.svelte';
  import TrainTab from './lib/TrainTab.svelte';
  import SettingsPanel from './lib/SettingsPanel.svelte';
  import { updateState, checkForUpdates, loadCurrentVersion } from './lib/updater.svelte.js';

  // 왼쪽 내비: create(작업공간) | library | templates | history
  let nav = $state('create');
  // 가운데 탭: generate | inpaint | pose | models | settings
  let tab = $state('generate');

  const tabs = [
    { id: 'generate', label: '생성' },
    { id: 'inpaint', label: '인페인트' },
    { id: 'pose', label: '포즈' },
    { id: 'models', label: '모델' },
    { id: 'train', label: 'LoRA 학습' },
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

  /** 작업내역/라이브러리의 "설정 불러오기": 기록의 params+LoRA를 스토어에 반영하고 생성 탭으로 이동.
   *  현재 메타에 없는 키(예: 구버전 설정)는 제외 — 모르는 키가 들어가면 다음 생성이 422가 된다. */
  function applyRecordParams(record) {
    const params = record?.params ?? record; // 구버전 호환: params 직접 전달
    if (params) {
      const known = new Set(store.meta.map((s) => s.key));
      const filtered = Object.fromEntries(
        Object.entries(params).filter(([k]) => known.has(k))
      );
      store.values = { ...store.values, ...filtered };
    }
    if (record?.loras && typeof record.loras === 'object') {
      store.values.loras = record.loras;
    }
    nav = 'create';
    tab = 'generate';
  }

  onMount(() => {
    initStore();
    // 시작 시 새 버전이 있는지 조용히 확인 (있을 때만 설정 탭에 표시)
    loadCurrentVersion();
    checkForUpdates({ silent: true });
    // 백엔드가 나중에 죽으면 배지에 반영 (30초 간격)
    const timer = setInterval(() => { checkBackendAlive(); }, 30000);
    return () => clearInterval(timer);
  });
</script>

<div class="app-layout">
  <SideNav nav={sideActive} onSelect={onNav} />

  <div class="app-main">
    <header class="topbar">
      <div class="brand">
        <img class="brand-logo" src="logo.png" alt="Atelier AI 로고" />
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
      {#if updateState.status === 'available'}
        <button
          class="update-badge"
          onclick={() => onNav('settings')}
          title="새 버전이 있어요. 클릭해서 업데이트"
        >
          새 버전 {updateState.availableVersion} 있음
        </button>
      {/if}
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
            <GenerateTab onLoad={applyRecordParams} />
          {:else if tab === 'inpaint'}
            <InpaintTab />
          {:else if tab === 'pose'}
            <PoseTab />
          {:else if tab === 'models'}
            <ModelsTab />
          {:else if tab === 'train'}
            <TrainTab />
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
