<script>
  // 왼쪽 아이콘 내비게이션 바 — 작업 중심 6개 메뉴
  let { nav, onSelect } = $props();

  const items = [
    { id: 'create', label: '이미지 만들기', icon: '✦' },
    { id: 'edit', label: '이미지 편집', icon: '🎨' },
    { id: 'models', label: '모델 및 학습', icon: '📦' },
    { id: 'library', label: '라이브러리', icon: '🖼' },
    { id: 'history', label: '작업 기록', icon: '🕒' },
    { id: 'settings', label: '설정', icon: '⚙' },
  ];
</script>

<nav class="sidenav" aria-label="주 메뉴">
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
