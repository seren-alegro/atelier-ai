<script>
  import { store } from './store.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate } from './api.js';

  let notice = $state('');
  let busy = $state(false);
  // 목업 갤러리 (백엔드 연동 전)
  let results = $state([]);

  async function onGenerate() {
    notice = '';
    busy = true;
    try {
      const params = { ...store.values };
      const paths = await requestGenerate(params);
      results = paths;
    } catch (e) {
      notice = e.message || '생성 중 오류가 발생했어요.';
    } finally {
      busy = false;
    }
  }
</script>

<div class="generate-tab">
  <div class="field">
    <div class="field-head"><span class="field-label">프롬프트</span></div>
    <TagInput bind:value={store.values.prompt} placeholder="태그 입력 (예: 1girl, solo)" />
  </div>

  <details class="field">
    <summary class="field-label">네거티브 프롬프트</summary>
    <div class="neg-body">
      <TagInput bind:value={store.values.negative_prompt} placeholder="제외할 태그 입력" />
    </div>
  </details>

  <button class="btn primary" onclick={onGenerate} disabled={busy || !store.loaded}>
    {busy ? '생성 중…' : '그림 생성하기'}
  </button>
  {#if notice}<p class="notice">{notice}</p>{/if}

  <h3>결과</h3>
  {#if results.length === 0}
    <div class="gallery-empty">아직 생성된 그림이 없어요. 위 버튼을 눌러보세요.</div>
  {:else}
    <div class="gallery">
      {#each results as src}
        <img {src} alt="생성 결과" />
      {/each}
    </div>
  {/if}
</div>
