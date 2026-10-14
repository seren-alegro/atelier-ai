<script>
  // 왼쪽 아이콘 내비게이션 바
  let { nav, onSelect } = $props();

  const items = [
    { id: 'create', label: '생성', icon: '✦' },
    { id: 'library', label: '라이브러리', icon: '🖼' },
    { id: 'templates', label: '템플릿', icon: '📋' },
    { id: 'history', label: '작업내역', icon: '🕒' },
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
