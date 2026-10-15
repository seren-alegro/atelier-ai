<script>
  // 프롬프트 도우미: 태그 칩을 누르면 프롬프트에 추가
  import { store } from './store.svelte.js';
  import { searchTags } from './api.js';
  import { t, locale } from './i18n.svelte.js';

  const GROUPS = $derived([
    {
      id: 'tags',
      label: t('help.tabTags'),
      tags: ['1girl', 'solo', 'highres', 'best quality', 'masterpiece', 'detailed eyes', 'soft lighting', 'anime style'],
    },
    {
      id: 'style',
      label: t('help.tabStyle'),
      tags: ['anime screencap', 'watercolor', 'cel shaded', 'soft shading', 'vibrant colors'],
    },
    {
      id: 'pose',
      label: t('help.tabPose'),
      tags: ['portrait', 'upper body', 'full body', 'close-up', 'dynamic pose', 'looking at viewer'],
    },
  ]);

  let groupId = $state('tags');
  const group = $derived(GROUPS.find((g) => g.id === groupId) ?? GROUPS[0]);

  // 🔍 검색 탭 상태
  let searchQuery = $state('');
  let searchResults = $state([]); // [{tag, category, count, desc_ko}]
  let searchTotal = $state(0);
  let searchBusy = $state(false);
  let searchNotice = $state('');
  let searchTimer = null;

  function promptTags() {
    return String(store.values.prompt ?? '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function addTag(tag) {
    const list = promptTags();
    if (!list.includes(tag)) {
      list.push(tag);
      store.values.prompt = list.join(', ');
    }
  }

  function onSearchInput(e) {
    searchQuery = e.target.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(runSearch, 300);
  }

  async function runSearch() {
    const q = searchQuery.trim();
    if (!q) {
      searchResults = [];
      searchTotal = 0;
      searchNotice = '';
      searchBusy = false;
      return;
    }
    searchBusy = true;
    searchNotice = '';
    try {
      const data = await searchTags(q, 50);
      searchResults = data.results ?? [];
      searchTotal = data.count ?? 0;
      if (searchResults.length === 0) searchNotice = t('help.noResults');
    } catch (e) {
      searchNotice = e.message || t('help.searchError');
      searchResults = [];
    } finally {
      searchBusy = false;
    }
  }

  function fmtCount(n) {
    return Number(n || 0).toLocaleString(locale());
  }
</script>

<details class="qp-section qp-details" open>
  <summary class="qp-title">{t('help.title')}</summary>
  <div class="helper-tabs">
    {#each GROUPS as g (g.id)}
      <button
        class:active={g.id === groupId}
        onclick={() => (groupId = g.id)}
      >
        {g.label}
      </button>
    {/each}
    <button
      class:active={'search' === groupId}
      onclick={() => (groupId = 'search')}
    >
      {t('help.tabSearch')}
    </button>
  </div>

  {#if groupId === 'search'}
    <div class="helper-search">
      <input
        class="search-input"
        type="text"
        placeholder={t('help.searchPh')}
        value={searchQuery}
        oninput={onSearchInput}
        aria-label={t('help.searchAria')}
      />
      {#if searchBusy}
        <p class="hint">{t('help.searching')}</p>
      {:else if searchNotice}
        <p class="hint">{searchNotice}</p>
      {:else if searchResults.length > 0}
        <p class="hint">{t('help.resultsOf', { total: fmtCount(searchTotal), shown: searchResults.length })}</p>
        <div class="search-results">
          {#each searchResults as r (r.tag)}
            {@const added = promptTags().includes(r.tag)}
            <button
              class="search-row"
              class:added
              onclick={() => addTag(r.tag)}
              title={t('help.addTitle')}
            >
              <span class="search-tag">{added ? '✓ ' : '+ '}{r.tag}</span>
              {#if r.desc_ko}
                <span class="search-desc">{r.desc_ko}</span>
              {/if}
              <span class="search-count">{t('help.times', { n: fmtCount(r.count) })}</span>
            </button>
          {/each}
        </div>
      {:else}
        <p class="hint">{t('help.hint')}</p>
      {/if}
    </div>
  {:else}
    <div class="helper-chips">
      {#each group.tags as tag (tag)}
        {@const added = promptTags().includes(tag)}
        <button
          class="tag-chip"
          class:added
          onclick={() => addTag(tag)}
          title={t('help.addTitle')}
        >
          {added ? '✓ ' : '+ '}{tag}
        </button>
      {/each}
    </div>
    <p class="hint">{t('help.chipHint')}</p>
  {/if}
</details>

<style>
  .helper-search {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
  }
  .search-input {
    width: 100%;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid var(--border, #3a3a5c);
    background: var(--input-bg, #1e1e33);
    color: inherit;
    font-size: 13px;
  }
  .search-results {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 260px;
    overflow-y: auto;
  }
  .search-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    text-align: left;
    padding: 6px 8px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 12px;
  }
  .search-row:hover {
    border-color: var(--border, #3a3a5c);
    background: rgba(255, 255, 255, 0.04);
  }
  .search-row.added .search-tag {
    color: #7ee2a8;
  }
  .search-tag {
    font-family: ui-monospace, monospace;
    white-space: nowrap;
  }
  .search-desc {
    flex: 1;
    opacity: 0.75;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .search-count {
    opacity: 0.5;
    white-space: nowrap;
    font-size: 11px;
  }
</style>
