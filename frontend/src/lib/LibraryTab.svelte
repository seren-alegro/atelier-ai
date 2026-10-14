<script>
  import { onMount } from 'svelte';
  import { historyStore, loadHistory, favorites } from './history.svelte.js';
  import HistoryGrid from './HistoryGrid.svelte';

  let { onLoad = null } = $props();

  const favs = $derived(favorites());

  onMount(() => {
    if (historyStore.items.length === 0) loadHistory();
  });
</script>

<div class="library-tab">
  <div class="tab-head">
    <h3>라이브러리</h3>
    <button class="btn" onclick={() => loadHistory()} disabled={historyStore.loading}>
      {historyStore.loading ? '불러오는 중…' : '새로고침'}
    </button>
  </div>
  {#if historyStore.notice}<p class="notice">{historyStore.notice}</p>{/if}

  <HistoryGrid
    items={favs}
    {onLoad}
    emptyText={historyStore.offline
      ? '백엔드에 연결되지 않았어요.'
      : '즐겨찾기한 그림이 없어요. 작업내역에서 ★를 눌러 모아보세요.'}
  />
</div>

<style>
  .library-tab {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .tab-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .tab-head h3 {
    margin: 0;
  }
</style>
