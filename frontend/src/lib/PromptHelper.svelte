<script>
  // 프롬프트 도우미: 태그 칩을 누르면 프롬프트에 추가
  import { store } from './store.svelte.js';

  const GROUPS = [
    {
      id: 'tags',
      label: '추천 태그',
      tags: ['1girl', 'solo', 'highres', 'best quality', 'masterpiece', 'detailed eyes', 'soft lighting', 'anime style'],
    },
    {
      id: 'style',
      label: '스타일',
      tags: ['anime screencap', 'watercolor', 'cel shaded', 'soft shading', 'vibrant colors'],
    },
    {
      id: 'pose',
      label: '구도',
      tags: ['portrait', 'upper body', 'full body', 'close-up', 'dynamic pose', 'looking at viewer'],
    },
  ];

  let groupId = $state('tags');
  const group = $derived(GROUPS.find((g) => g.id === groupId) ?? GROUPS[0]);

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
</script>

<details class="qp-section qp-details" open>
  <summary class="qp-title">✦ 프롬프트 도우미</summary>
  <div class="helper-tabs">
    {#each GROUPS as g (g.id)}
      <button
        class:active={g.id === groupId}
        onclick={() => (groupId = g.id)}
      >
        {g.label}
      </button>
    {/each}
  </div>
  <div class="helper-chips">
    {#each group.tags as tag (tag)}
      {@const added = promptTags().includes(tag)}
      <button
        class="tag-chip"
        class:added
        onclick={() => addTag(tag)}
        title="프롬프트에 추가"
      >
        {added ? '✓ ' : '+ '}{tag}
      </button>
    {/each}
  </div>
  <p class="hint">칩을 누르면 프롬프트에 태그가 추가돼요.</p>
</details>
