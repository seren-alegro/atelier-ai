<script>
  import { onMount } from 'svelte';
  import { historyStore, loadHistory } from './history.svelte.js';
  import HistoryGrid from './HistoryGrid.svelte';

  let { onLoad = null } = $props();

  onMount(() => loadHistory());
</script>

<div class="history-tab">
  <div class="tab-head">
    <h3>작업내역</h3>
    <button class="btn" onclick={() => loadHistory()} disabled={historyStore.loading}>
      {historyStore.loading ? '불러오는 중…' : '새로고침'}
    </button>
  </div>
  {#if historyStore.notice}<p class="notice">{historyStore.notice}</p>{/if}

  <HistoryGrid
    items={historyStore.items}
    {onLoad}
    emptyText={historyStore.offline
      ? '백엔드에 연결되지 않았어요.'
      : '아직 생성한 그림이 없어요. 생성 탭에서 그림을 만들어보세요.'}
  />
</div>

<style>
  .history-tab {
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
