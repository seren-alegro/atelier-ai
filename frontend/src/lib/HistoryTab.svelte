<script>
  import { onMount } from 'svelte';
  import { historyStore, loadHistory } from './history.svelte.js';
  import HistoryGrid from './HistoryGrid.svelte';
  import { t } from './i18n.svelte.js';

  let { onLoad = null } = $props();

  onMount(() => loadHistory());
</script>

<div class="history-tab">
  <div class="tab-head">
    <h3>{t('hist.title')}</h3>
    <button class="btn" onclick={() => loadHistory()} disabled={historyStore.loading}>
      {historyStore.loading ? t('common.loading') : t('common.refresh')}
    </button>
  </div>
  {#if historyStore.notice}<p class="notice">{historyStore.notice}</p>{/if}

  <HistoryGrid
    items={historyStore.items}
    {onLoad}
    emptyText={historyStore.offline
      ? t('hist.offline')
      : t('hist.emptyGen')}
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
