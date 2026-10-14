<script>
  import { onMount } from 'svelte';
  import { store } from './store.svelte.js';

  const STORAGE_KEY = 'atelier-ai-templates';

  // 내장 템플릿 (삭제 불가, 항상 표시)
  const BUILTINS = [
    {
      name: '고화질 애니 기본',
      desc: '깔끔한 고화질 애니메이션 스타일의 기본 세팅이에요.',
      builtin: true,
      values: {
        prompt:
          'masterpiece, best quality, highres, absurdres, newest, detailed eyes, ',
        negative_prompt: 'worst quality, low quality, bad anatomy, ',
        steps: 28,
        cfg_scale: 5.0,
        sampler: 'dpmpp_2m_karras',
        width: 832,
        height: 1216,
        seed: -1,
        batch_size: 1,
      },
    },
    {
      name: '인물 클로즈업',
      desc: '얼굴 클로즈업에 맞춘 세팅이에요.',
      builtin: true,
      values: {
        prompt:
          'masterpiece, portrait, close-up, looking at viewer, detailed face, beautiful detailed eyes, ',
        negative_prompt: 'worst quality, low quality, bad anatomy, ',
        steps: 28,
        cfg_scale: 5.0,
        sampler: 'er_sde',
        width: 768,
        height: 1024,
        seed: -1,
        batch_size: 1,
      },
    },
  ];

  let userTemplates = $state([]);
  let name = $state('');
  let notice = $state('');

  function readStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  }

  function writeStorage() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userTemplates));
    } catch {
      notice = '브라우저 저장소에 저장하지 못했어요.';
    }
  }

  function saveCurrent() {
    const n = name.trim();
    if (!n) {
      notice = '템플릿 이름을 입력해주세요.';
      return;
    }
    // 깊은 복사로 현재 설정값을 통째로 저장
    const values = JSON.parse(JSON.stringify(store.values));
    userTemplates = [
      { name: n, desc: '내 템플릿', builtin: false, values },
      ...userTemplates.filter((t) => t.name !== n),
    ];
    writeStorage();
    name = '';
    notice = `'${n}'을(를) 저장했어요.`;
  }

  function applyTemplate(tpl) {
    store.values = { ...store.values, ...tpl.values };
    notice = `'${tpl.name}'을(를) 적용했어요. 생성 탭에서 확인해보세요.`;
  }

  function deleteTemplate(tpl) {
    if (!confirm(`'${tpl.name}'을(를) 삭제할까요?`)) return;
    userTemplates = userTemplates.filter((t) => t.name !== tpl.name);
    writeStorage();
  }

  function fmtValues(values) {
    const bits = [];
    if (values.width && values.height)
      bits.push(`${values.width}×${values.height}`);
    if (values.steps) bits.push(`${values.steps}스텝`);
    if (values.cfg_scale) bits.push(`CFG ${values.cfg_scale}`);
    if (values.sampler) bits.push(values.sampler);
    return bits.join(' · ');
  }

  onMount(() => {
    userTemplates = readStorage();
  });
</script>

<div class="templates-tab">
  <div class="tab-head">
    <h3>템플릿</h3>
  </div>
  {#if notice}<p class="notice">{notice}</p>{/if}

  <section class="save-box">
    <h4>현재 설정 저장하기</h4>
    <p class="hint">
      지금의 프롬프트·네거티브·주요 설정을 이름 붙여 저장해요. (이 브라우저에만
      저장돼요)
    </p>
    <div class="save-row">
      <input
        class="text-input"
        type="text"
        bind:value={name}
        placeholder="템플릿 이름 (예: 내 최애 세팅)"
        maxlength="40"
        onkeydown={(e) => e.key === 'Enter' && saveCurrent()}
      />
      <button class="btn primary" onclick={saveCurrent}>저장</button>
    </div>
  </section>

  <section>
    <h4>기본 템플릿</h4>
    <div class="tpl-grid">
      {#each BUILTINS as tpl (tpl.name)}
        <article class="tpl-card">
          <h5>{tpl.name}</h5>
          <p class="hint">{tpl.desc}</p>
          <p class="meta dim">{fmtValues(tpl.values)}</p>
          <p class="prompt-preview" title={tpl.values.prompt}>
            {tpl.values.prompt}
          </p>
          <button class="btn small" onclick={() => applyTemplate(tpl)}>
            적용
          </button>
        </article>
      {/each}
    </div>
  </section>

  <section>
    <h4>내 템플릿 ({userTemplates.length})</h4>
    {#if userTemplates.length === 0}
      <div class="gallery-empty">
        저장된 템플릿이 없어요. 위에서 현재 설정을 저장해보세요.
      </div>
    {:else}
      <div class="tpl-grid">
        {#each userTemplates as tpl (tpl.name)}
          <article class="tpl-card">
            <h5>{tpl.name}</h5>
            <p class="meta dim">{fmtValues(tpl.values)}</p>
            <p class="prompt-preview" title={tpl.values.prompt}>
              {tpl.values.prompt || '(프롬프트 없음)'}
            </p>
            <div class="card-actions">
              <button class="btn small" onclick={() => applyTemplate(tpl)}>
                적용
              </button>
              <button
                class="btn small ghost danger"
                onclick={() => deleteTemplate(tpl)}
              >
                삭제
              </button>
            </div>
          </article>
        {/each}
      </div>
    {/if}
  </section>
</div>

<style>
  .templates-tab {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .tab-head h3 {
    margin: 0;
  }

  .templates-tab h4 {
    margin: 0 0 10px;
    font-size: 14px;
  }

  .save-box {
    background: linear-gradient(180deg, var(--bg2), var(--bg));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
  }

  .save-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }

  .save-row .text-input {
    flex: 1;
  }

  .tpl-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }

  .tpl-card {
    background: linear-gradient(180deg, var(--bg2), var(--bg));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }

  .tpl-card h5 {
    margin: 0 0 4px;
    font-size: 14px;
  }

  .prompt-preview {
    font-size: 12px;
    color: var(--muted);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin: 8px 0 10px;
    word-break: keep-all;
  }

  .meta.dim {
    font-size: 12px;
    color: var(--muted);
    opacity: 0.75;
    margin: 4px 0 0;
  }

  .card-actions {
    display: flex;
    gap: 8px;
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
