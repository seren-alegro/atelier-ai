<script>
  // 왼쪽 아이콘 내비게이션 바 — 작업 중심 6개 메뉴
  import { t } from './i18n.svelte.js';

  let { nav, onSelect } = $props();

  const items = $derived([
    { id: 'create', label: t('nav.create'), icon: '✦' },
    { id: 'edit', label: t('nav.edit'), icon: '🎨' },
    { id: 'models', label: t('nav.models'), icon: '📦' },
    { id: 'library', label: t('nav.library'), icon: '🖼' },
    { id: 'history', label: t('nav.history'), icon: '🕒' },
    { id: 'settings', label: t('nav.settings'), icon: '⚙' },
  ]);
</script>

<nav class="sidenav" aria-label={t('nav.main')}>
  {#each items as it (it.id)}
    <button
      class="nav-item"
      class:active={nav === it.id}
      onclick={() => onSelect(it.id)}
      title={it.label}
    >
      <span class="nav-icon" aria-hidden="true">{it.icon}</span>
      <span class="nav-label">{it.label}</span>
    </button>
  {/each}
</nav>
