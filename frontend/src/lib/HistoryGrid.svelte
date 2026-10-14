<script>
  // 작업내역/라이브러리 공용 썸네일 그리드
  import { apiUrl } from './api.js';
  import { removeHistory, toggleFavorite } from './history.svelte.js';

  let {
    items = [],
    onLoad = null, // (params) => void — "설정 불러오기"
    emptyText = '아직 기록이 없어요.',
  } = $props();

  const modeLabel = {
    txt2img: '텍스트로 생성',
    img2img: '이미지로 생성',
    inpaint: '인페인트',
    pose: '포즈 지정',
  };

  function fmtDate(ts) {
    try {
      return new Date(ts).toLocaleString('ko-KR', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return ts;
    }
  }

  function summary(rec) {
    const p = rec.params ?? {};
    const bits = [];
    if (rec.model) bits.push(rec.model);
    if (p.width && p.height) bits.push(`${p.width}×${p.height}`);
    if (p.steps) bits.push(`${p.steps}스텝`);
    if (p.sampler) bits.push(p.sampler);
    return bits.join(' · ');
  }

  function askDelete(rec) {
    if (confirm('이 기록을 삭제할까요? (이미지 파일은 남아 있어요)')) {
      removeHistory(rec.id);
    }
  }
</script>

{#if items.length === 0}
  <div class="history-empty">{emptyText}</div>
{:else}
  <div class="history-grid">
    {#each items as rec (rec.id)}
      <article class="history-card">
        <div class="thumb">
          {#if rec.results?.length}
            <img src={apiUrl(rec.results[0])} alt="생성 결과" loading="lazy" />
          {:else}
            <div class="thumb-none">이미지 없음</div>
          {/if}
          <button
            class="fav-btn"
            class:on={rec.favorite}
            onclick={() => toggleFavorite(rec.id)}
            title={rec.favorite ? '즐겨찾기 해제' : '즐겨찾기'}
            aria-label="즐겨찾기 토글"
          >
            {rec.favorite ? '★' : '☆'}
          </button>
        </div>
        <div class="card-body">
          <p class="prompt" title={rec.params?.prompt ?? ''}>
            {rec.params?.prompt || '(프롬프트 없음)'}
          </p>
          <p class="meta">
            {modeLabel[rec.mode] ?? rec.mode} · {fmtDate(rec.timestamp)}
          </p>
          {#if summary(rec)}
            <p class="meta dim">{summary(rec)}</p>
          {/if}
          <div class="card-actions">
            {#if onLoad}
              <button class="btn small" onclick={() => onLoad(rec)}>
                설정 불러오기
              </button>
            {/if}
            <button class="btn small ghost danger" onclick={() => askDelete(rec)}>
              삭제
            </button>
          </div>
        </div>
      </article>
    {/each}
  </div>
{/if}

<style>
  .history-empty {
    padding: 48px 16px;
    text-align: center;
    color: var(--muted);
    border: 1px dashed var(--border);
    border-radius: 14px;
  }

  .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }

  .history-card {
    background: linear-gradient(180deg, var(--bg2), var(--bg));
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(0, 0, 0, 0.3));
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }

  .history-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md, 0 6px 20px rgba(0, 0, 0, 0.4));
  }

  .thumb {
    position: relative;
    aspect-ratio: 3 / 4;
    background: var(--bg3);
    overflow: hidden;
  }

  .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .thumb-none {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 13px;
  }

  .fav-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: rgba(20, 20, 35, 0.75);
    color: var(--muted);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
  }

  .fav-btn.on {
    color: #ffd76a;
    border-color: #ffd76a;
  }

  .card-body {
    padding: 12px;
  }

  .prompt {
    margin: 0 0 6px;
    font-size: 13px;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    word-break: keep-all;
  }

  .meta {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
  }

  .meta.dim {
    opacity: 0.75;
    margin-top: 2px;
  }

  .card-actions {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }

  .btn.small {
    font-size: 12px;
    padding: 6px 10px;
  }

  .btn.ghost {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
  }

  .btn.ghost.danger:hover {
    color: #ff7a90;
    border-color: #ff7a90;
  }
</style>
