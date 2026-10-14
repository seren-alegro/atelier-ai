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

  // 왼쪽 내비: create(이미지 만들기) | edit(이미지 편집) | models(모델 및 학습)
  //            | library(라이브러리) | history(작업 기록) | settings(설정)
  let nav = $state('create');
  // 서브탭 (두 개 화면을 가진 메뉴용)
  let editTab = $state('inpaint');      // inpaint | pose
  let modelsTab = $state('manage');     // manage | train
  let libraryTab = $state('images');    // images | templates

  function onNav(id) {
    nav = id;
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
  <SideNav {nav} onSelect={onNav} />

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
      <!-- 이미지 만들기: 생성 탭 + 오른쪽 인스펙터 -->
      <div class="workspace">
        <div class="center-col">
          <GenerateTab onLoad={applyRecordParams} />
        </div>
        <aside class="quick-panel" aria-label="생성 설정">
          <QuickPanel />
        </aside>
      </div>
    {:else if nav === 'edit'}
      <!-- 이미지 편집: 인페인트 / 포즈 서브탭 + 오른쪽 인스펙터 -->
      <div class="workspace">
        <div class="center-col">
          <nav class="subtabs" aria-label="편집 도구">
            <button class:active={editTab === 'inpaint'} onclick={() => (editTab = 'inpaint')}>인페인트</button>
            <button class:active={editTab === 'pose'} onclick={() => (editTab = 'pose')}>포즈</button>
          </nav>
          {#if editTab === 'inpaint'}
            <InpaintTab />
          {:else}
            <PoseTab />
          {/if}
        </div>
        <aside class="quick-panel" aria-label="생성 설정">
          <QuickPanel hideHelper />
        </aside>
      </div>
    {:else if nav === 'models'}
      <!-- 모델 및 학습: 모델 관리 / LoRA 학습 서브탭 -->
      <div class="workspace">
        <div class="center-col">
          <nav class="subtabs" aria-label="모델 및 학습">
            <button class:active={modelsTab === 'manage'} onclick={() => (modelsTab = 'manage')}>모델 관리</button>
            <button class:active={modelsTab === 'train'} onclick={() => (modelsTab = 'train')}>LoRA 학습</button>
          </nav>
          {#if modelsTab === 'manage'}
            <ModelsTab />
          {:else}
            <TrainTab />
          {/if}
        </div>
      </div>
    {:else if nav === 'library'}
      <!-- 라이브러리: 이미지 / 템플릿 서브탭 -->
      <div class="workspace">
        <div class="center-col">
          <nav class="subtabs" aria-label="라이브러리">
            <button class:active={libraryTab === 'images'} onclick={() => (libraryTab = 'images')}>이미지</button>
            <button class:active={libraryTab === 'templates'} onclick={() => (libraryTab = 'templates')}>템플릿</button>
          </nav>
          {#if libraryTab === 'images'}
            <LibraryTab onLoad={applyRecordParams} />
          {:else}
            <TemplatesTab />
          {/if}
        </div>
      </div>
    {:else if nav === 'history'}
      <HistoryTab onLoad={applyRecordParams} />
    {:else if nav === 'settings'}
      <SettingsPanel />
    {/if}
  </div>
</div>
