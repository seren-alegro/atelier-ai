<script>
  // Danbooru식 태그 입력: 쉼표/Enter로 칩 생성, 칩의 ×로 삭제.
  // value는 쉼표 구분 문자열로 바인딩된다.
  // 자동완성: 입력 중 후보 드롭다운 (↑↓ 이동, Enter/Tab 확정, Esc 닫기).
  // 한국어 라벨: 백엔드 /api/tag-labels에서 받아 칩 툴팁·자동완성에 표시.
  import { onMount } from 'svelte';
  import { suggestTags } from './tagSuggestions.js';
  import { ensureTagLabels, labelOf } from './tagLabels.js';

  let { value = $bindable(''), placeholder = '태그를 입력하고 쉼표를 누르세요' } = $props();
  let draft = $state('');
  let open = $state(false);
  let activeIdx = $state(0);
  let labelsReady = $state(false);

  onMount(async () => {
    // 라벨은 백그라운드로 — 없어도 입력은 정상 동작
    await ensureTagLabels();
    labelsReady = true;
  });

  function tagList() {
    return String(value ?? '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
  }

  const suggestions = $derived(suggestTags(draft, tagList()));

  function addTag(t) {
    const clean = t.trim().replace(/,+$/, '').trim();
    if (!clean) return;
    const list = tagList();
    if (!list.some((x) => x.toLowerCase() === clean.toLowerCase())) {
      value = [...list, clean].join(', ');
    }
    draft = '';
    open = false;
    activeIdx = 0;
  }

  function commitDraft() {
    addTag(draft);
  }

  function removeAt(i) {
    const list = tagList();
    list.splice(i, 1);
    value = list.join(', ');
  }

  function onKeydown(e) {
    if (open && suggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        activeIdx = (activeIdx + 1) % suggestions.length;
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        activeIdx = (activeIdx - 1 + suggestions.length) % suggestions.length;
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        addTag(suggestions[activeIdx] ?? draft);
        return;
      }
      if (e.key === 'Escape') {
        open = false;
        return;
      }
    }
    if (e.key === ',' || e.key === 'Enter') {
      e.preventDefault();
      commitDraft();
    } else if (e.key === 'Backspace' && !draft && tagList().length > 0) {
      removeAt(tagList().length - 1);
    }
  }

  function onInput() {
    activeIdx = 0;
    open = draft.trim().length > 0;
  }
</script>

<div class="tag-input">
  <div class="chips">
    {#each tagList() as tag, i (tag + i)}
      <span class="chip" title={labelOf(tag) ?? tag}>
        <span class="chip-tag">{tag}</span>
        <button class="chip-x" onclick={() => removeAt(i)} aria-label="태그 삭제">×</button>
      </span>
    {/each}
    <input
      class="draft"
      bind:value={draft}
      {placeholder}
      onkeydown={onKeydown}
      oninput={onInput}
      onblur={() => { commitDraft(); open = false; }}
      onfocus={onInput}
      autocomplete="off"
      spellcheck="false"
    />
    {#if draft.trim()}
      <button
        class="add-btn"
        onmousedown={(e) => { e.preventDefault(); commitDraft(); }}
        title="태그 추가"
      >＋ 추가</button>
    {/if}
  </div>
  {#if open && suggestions.length > 0}
    <ul class="suggest" role="listbox">
      {#each suggestions as s, i (s)}
        {@const label = labelOf(s)}
        <li
          role="option"
          aria-selected={i === activeIdx}
          class:active={i === activeIdx}
          onmousedown={(e) => { e.preventDefault(); addTag(s); }}
          onmouseenter={() => (activeIdx = i)}
        >
          <span class="suggest-tag">{s}</span>
          {#if label}<span class="suggest-label">{label}</span>{/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .tag-input { position: relative; }
  .add-btn {
    flex-shrink: 0;
    background: rgba(124, 92, 255, 0.16);
    border: 1px solid rgba(124, 92, 255, 0.45);
    color: #c9bfff;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 12.5px;
    cursor: pointer;
  }
  .add-btn:hover { background: rgba(124, 92, 255, 0.3); }
  .suggest {
    position: absolute;
    z-index: 30;
    left: 0; right: 0; top: 100%;
    margin: 4px 0 0;
    padding: 4px;
    list-style: none;
    background: #23253d;
    border: 1px solid #3a3d68;
    border-radius: 10px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
    max-height: 240px;
    overflow-y: auto;
  }
  .suggest li {
    padding: 7px 10px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 13.5px;
    color: #cfd2ea;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .suggest li.active, .suggest li:hover {
    background: rgba(124, 92, 255, 0.22);
    color: #fff;
  }
  .suggest-tag { font-weight: 600; }
  .suggest-label {
    font-size: 11.5px;
    color: #9aa0c7;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }
  .suggest li.active .suggest-label { color: #c9bfff; }
</style>
