<script>
  import { onMount } from 'svelte';
  import { t, getLang, setLang, LANGS } from './lib/i18n.svelte.js';
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

  // 언어 전환 시 <html lang> 동기화 (접근성)
  $effect(() => {
    document.documentElement.lang = getLang();
  });

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
        <img class="brand-logo" src="logo.png" alt={t('app.logoAlt')} />
        <h1>Atelier AI</h1>
        <span class="tagline">{t('app.tagline')}</span>
        <div class="lang-switch" role="group" aria-label={t('set.langTitle')}>
          {#each LANGS as l (l.id)}
            <button
              class:active={getLang() === l.id}
              onclick={() => setLang(l.id)}
              title={l.label}
            >{l.id === 'ko' ? '한' : l.id === 'ja' ? '日' : 'EN'}</button>
          {/each}
        </div>
      </div>
      <button
        class="api-badge"
        class:online={store.apiOnline}
        class:offline={store.loaded && !store.apiOnline}
        onclick={() => { if (store.loaded && !store.apiOnline) reconnect(); }}
        title={store.loaded && !store.apiOnline ? t('app.reconnect') : ''}
      >
        {#if !store.loaded}
          {t('app.backendChecking')}
        {:else if store.apiOnline}
          {t('app.backendOk')}
        {:else}
          {t('app.backendMissing')}
        {/if}
      </button>
      {#if updateState.status === 'available'}
        <button
          class="update-badge"
          onclick={() => onNav('settings')}
          title={t('app.updateTitle')}
        >
          {t('app.updateAvailable', { v: updateState.availableVersion })}
        </button>
      {/if}
    </header>

    {#if nav === 'create'}
      <!-- 이미지 만들기: 생성 탭 + 오른쪽 인스펙터 -->
      <div class="workspace">
        <div class="center-col">
          <GenerateTab onLoad={applyRecordParams} />
        </div>
        <aside class="quick-panel" aria-label={t('app.panelSettings')}>
          <QuickPanel />
        </aside>
      </div>
    {:else if nav === 'edit'}
      <!-- 이미지 편집: 인페인트 / 포즈 서브탭 + 오른쪽 인스펙터 -->
      <div class="workspace">
        <div class="center-col">
          <nav class="subtabs" aria-label={t('app.editTools')}>
            <button class:active={editTab === 'inpaint'} onclick={() => (editTab = 'inpaint')}>{t('app.editInpaint')}</button>
            <button class:active={editTab === 'pose'} onclick={() => (editTab = 'pose')}>{t('app.editPose')}</button>
          </nav>
          {#if editTab === 'inpaint'}
            <InpaintTab />
          {:else}
            <PoseTab />
          {/if}
        </div>
        <aside class="quick-panel" aria-label={t('app.panelSettings')}>
          <QuickPanel hideHelper />
        </aside>
      </div>
    {:else if nav === 'models'}
      <!-- 모델 및 학습: 모델 관리 / LoRA 학습 서브탭 -->
      <div class="workspace">
        <div class="center-col">
          <nav class="subtabs" aria-label={t('app.modelsTabs')}>
            <button class:active={modelsTab === 'manage'} onclick={() => (modelsTab = 'manage')}>{t('app.modelsManage')}</button>
            <button class:active={modelsTab === 'train'} onclick={() => (modelsTab = 'train')}>{t('app.modelsTrain')}</button>
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
          <nav class="subtabs" aria-label={t('app.libraryTabs')}>
            <button class:active={libraryTab === 'images'} onclick={() => (libraryTab = 'images')}>{t('app.libraryImages')}</button>
            <button class:active={libraryTab === 'templates'} onclick={() => (libraryTab = 'templates')}>{t('app.libraryTemplates')}</button>
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
