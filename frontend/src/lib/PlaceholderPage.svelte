<script>
  // "준비 중" 플레이스홀더 페이지 (라이브러리/템플릿/작업내역)
  let { icon, title, desc } = $props();
</script>

<div class="placeholder-wrap">
  <div class="placeholder-card">
    <div class="placeholder-icon" aria-hidden="true">{icon}</div>
    <h2>{title}</h2>
    <p>{desc}</p>
    <p class="hint">준비 중인 기능이에요.</p>
  </div>
</div>
