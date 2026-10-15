<script>
  import { onMount } from 'svelte';
  import { historyStore, loadHistory, favorites } from './history.svelte.js';
  import HistoryGrid from './HistoryGrid.svelte';
  import { t } from './i18n.svelte.js';

  let { onLoad = null } = $props();

  const favs = $derived(favorites());

  onMount(() => {
    if (historyStore.items.length === 0) loadHistory();
  });
</script>

<div class="library-tab">
  <div class="tab-head">
    <h3>{t('lib.title')}</h3>
    <button class="btn" onclick={() => loadHistory()} disabled={historyStore.loading}>
      {historyStore.loading ? t('common.loading') : t('common.refresh')}
    </button>
  </div>
  {#if historyStore.notice}<p class="notice">{historyStore.notice}</p>{/if}

  <HistoryGrid
    items={favs}
    {onLoad}
    emptyText={historyStore.offline
      ? t('hist.offline')
      : t('lib.emptyFav')}
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
