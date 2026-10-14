<script>
  // Danbooru식 태그 입력: 쉼표/Enter로 칩 생성, 칩의 ×로 삭제.
  // value는 쉼표 구분 문자열로 바인딩된다.
  // TODO: 태그별 한국어 라벨 데이터 연동 (현재 자리만 표시)
  let { value = $bindable(''), placeholder = '태그를 입력하고 쉼표를 누르세요' } = $props();
  let draft = $state('');

  function tagList() {
    return String(value ?? '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function commitDraft() {
    const t = draft.trim().replace(/,+$/, '').trim();
    if (!t) {
      draft = '';
      return;
    }
    const list = tagList();
    // 중복 태그 방지 (대소문자 무시)
    if (!list.some((x) => x.toLowerCase() === t.toLowerCase())) {
      value = [...list, t].join(', ');
    }
    draft = '';
  }

  function removeAt(i) {
    const list = tagList();
    list.splice(i, 1);
    value = list.join(', ');
  }

  function onKeydown(e) {
    if (e.key === ',' || e.key === 'Enter') {
      e.preventDefault();
      commitDraft();
    } else if (e.key === 'Backspace' && !draft && tagList().length > 0) {
      removeAt(tagList().length - 1);
    }
  }
</script>

<div class="tag-input">
  <div class="chips">
    {#each tagList() as tag, i (tag + i)}
      <span class="chip" title="한국어 라벨: 준비 중 (TODO)">
        <span class="chip-tag">{tag}</span>
        <!-- TODO: 태그별 한국어 라벨 데이터 연동 시 여기에 표시 -->
        <button class="chip-x" onclick={() => removeAt(i)} aria-label="태그 삭제">×</button>
      </span>
    {/each}
    <input
      class="draft"
      bind:value={draft}
      {placeholder}
      onkeydown={onKeydown}
      onblur={commitDraft}
    />
  </div>
</div>
