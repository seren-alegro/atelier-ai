<script>
  // 인페인트 탭: 이미지 업로드 → 마스크 브러시 → inpaint 생성
  import { store } from './store.svelte.js';
  import TagInput from './TagInput.svelte';
  import { requestGenerate } from './api.js';

  let fileInput = $state(null);
  let imgEl = $state(null);
  let viewCanvas = $state(null); // 화면 표시용 오버레이 캔버스 (반투명 빨강)
  let viewCtx = null;
  let offCanvas = null; // 실제 마스크 (검정 바탕 + 흰색 스트로크)
  let offCtx = null;

  let imageDataUrl = $state('');
  let brushSize = $state(40);
  let erasing = $state(false);
  let painting = $state(false);
  let last = $state(null);
  let undoStack = $state([]);
  let busy = $state(false);
  let notice = $state('');
  let results = $state([]);
  let hasMask = $state(false);

  const denoiseMeta = () => store.meta.find((m) => m.key === 'denoise_strength');

  function pickFile() {
    fileInput.click();
  }

  function onFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      imageDataUrl = rd.result;
      undoStack = [];
      hasMask = false;
    };
    rd.readAsDataURL(f);
    e.target.value = '';
  }

  function onImageLoad() {
    const w = imgEl.naturalWidth;
    const h = imgEl.naturalHeight;
    viewCanvas.width = w;
    viewCanvas.height = h;
    viewCtx = viewCanvas.getContext('2d');
    offCanvas = document.createElement('canvas');
    offCanvas.width = w;
    offCanvas.height = h;
    offCtx = offCanvas.getContext('2d');
    offCtx.fillStyle = '#000';
    offCtx.fillRect(0, 0, w, h);
  }

  function evtPos(e) {
    const r = viewCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (viewCanvas.width / r.width),
      y: (e.clientY - r.top) * (viewCanvas.height / r.height),
    };
  }

  function pushUndo() {
    undoStack.push({ view: viewCanvas.toDataURL(), off: offCanvas.toDataURL() });
    if (undoStack.length > 30) undoStack.shift();
  }

  function drawStroke(ctx, x0, y0, x1, y1, erase, size, style) {
    ctx.save();
    if (erase) {
      ctx.globalCompositeOperation = 'destination-out';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = style;
      ctx.fillStyle = style;
    }
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x1, y1, size / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function strokeTo(p) {
    const s = brushSize;
    drawStroke(viewCtx, last.x, last.y, p.x, p.y, erasing, s, 'rgba(255,90,90,0.55)');
    drawStroke(offCtx, last.x, last.y, p.x, p.y, erasing, s, '#fff');
    last = p;
  }

  function onDown(e) {
    if (!imageDataUrl) return;
    e.preventDefault();
    pushUndo();
    painting = true;
    const p = evtPos(e);
    last = p;
    strokeTo(p);
    viewCanvas.setPointerCapture?.(e.pointerId);
  }

  function onMove(e) {
    if (painting) strokeTo(evtPos(e));
  }

  function onUp() {
    if (painting) {
      painting = false;
      hasMask = true;
    }
  }

  function restoreSnapshot(dataUrl, canvas) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const ctx = canvas.getContext('2d');
        ctx.save();
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        ctx.restore();
        resolve();
      };
      img.src = dataUrl;
    });
  }

  async function undo() {
    const s = undoStack.pop();
    if (!s) return;
    await restoreSnapshot(s.view, viewCanvas);
    await restoreSnapshot(s.off, offCanvas);
  }

  function clearMask() {
    if (!imageDataUrl) return;
    pushUndo();
    viewCtx.save();
    viewCtx.clearRect(0, 0, viewCanvas.width, viewCanvas.height);
    viewCtx.restore();
    offCtx.save();
    offCtx.globalCompositeOperation = 'source-over';
    offCtx.fillStyle = '#000';
    offCtx.fillRect(0, 0, offCanvas.width, offCanvas.height);
    offCtx.restore();
    hasMask = false;
  }

  function exportMaskB64() {
    // 검정 바탕 위에 마스크 합성 → base64 PNG (prefix 제거, 백엔드는 raw base64 기대)
    const c = document.createElement('canvas');
    c.width = offCanvas.width;
    c.height = offCanvas.height;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(offCanvas, 0, 0);
    return c.toDataURL('image/png').split(',')[1];
  }

  async function onInpaint() {
    notice = '';
    busy = true;
    try {
      const params = { ...store.values };
      const image = imageDataUrl.split(',')[1];
      const mask = exportMaskB64();
      const paths = await requestGenerate(params, 'inpaint', image, mask);
      results = paths;
    } catch (e) {
      notice = e.message || '인페인트 중 오류가 발생했어요.';
    } finally {
      busy = false;
    }
  }
</script>

<div class="inpaint-tab">
  <h2>인페인트</h2>
  <p class="field-desc">그림을 불러와서 고치고 싶은 부분을 칠하면 AI가 다시 그려줘요.</p>

  <div class="field">
    <div class="field-head">
      <span class="field-label">이미지</span>
      <button class="btn" onclick={pickFile}>이미지 불러오기</button>
    </div>
    <input
      type="file"
      accept="image/*"
      bind:this={fileInput}
      onchange={onFile}
      style="display:none"
    />
    {#if !imageDataUrl}
      <div class="canvas-placeholder">이미지를 불러오세요</div>
    {:else}
      <div class="canvas-stack">
        <img bind:this={imgEl} src={imageDataUrl} alt="인페인트 대상" onload={onImageLoad} />
        <canvas
          bind:this={viewCanvas}
          class="mask-layer"
          onpointerdown={onDown}
          onpointermove={onMove}
          onpointerup={onUp}
          onpointercancel={onUp}
        ></canvas>
      </div>
      <div class="toolbar">
        <button class="btn tool" class:active={!erasing} onclick={() => (erasing = false)}>브러시</button>
        <button class="btn tool" class:active={erasing} onclick={() => (erasing = true)}>지우개</button>
        <label class="brush-label">
          크기
          <input type="range" min="8" max="160" step="2" bind:value={brushSize} />
          <span>{brushSize}px</span>
        </label>
        <button class="btn tool" onclick={undo} disabled={undoStack.length === 0}>되돌리기</button>
        <button class="btn tool" onclick={clearMask}>지우기</button>
      </div>
      <p class="recommended">빨갛게 칠한 부분이 다시 그려져요.</p>
    {/if}
  </div>

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

  <div class="field">
    <div class="field-head">
      <span class="field-label">디노이즈 강도</span>
      <span class="recommended">추천: {denoiseMeta()?.recommended ?? 0.75}</span>
    </div>
    <p class="field-desc">{denoiseMeta()?.description ?? '원본을 얼마나 바꿀지 정해요.'}</p>
    <div class="number-row">
      <input
        type="range"
        min={denoiseMeta()?.min ?? 0}
        max={denoiseMeta()?.max ?? 1}
        step="0.05"
        bind:value={store.values.denoise_strength}
      />
      <span class="number-box">{store.values.denoise_strength ?? 0.75}</span>
    </div>
  </div>

  <button
    class="btn primary"
    onclick={onInpaint}
    disabled={busy || !imageDataUrl || !hasMask || !store.loaded}
  >
    {busy ? '그리는 중…' : '인페인트 실행'}
  </button>
  {#if !hasMask && imageDataUrl}
    <p class="recommended">마스크를 먼저 칠해주세요.</p>
  {/if}
  {#if notice}<p class="notice">{notice}</p>{/if}

  <h3>결과</h3>
  {#if results.length === 0}
    <div class="gallery-empty">아직 결과가 없어요.</div>
  {:else}
    <div class="gallery">
      {#each results as src}
        <img {src} alt="인페인트 결과" />
      {/each}
    </div>
  {/if}
</div>

<style>
  .canvas-stack {
    position: relative;
    line-height: 0;
  }
  .canvas-stack img {
    width: 100%;
    height: auto;
    border-radius: 8px;
    display: block;
  }
  .mask-layer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    touch-action: none;
    cursor: crosshair;
    border-radius: 8px;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
  }
  .btn.tool {
    padding: 6px 12px;
    font-size: 13px;
  }
  .btn.tool.active {
    border-color: var(--accent);
    color: var(--accent);
  }
  .btn.tool:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .brush-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--muted);
    flex: 1;
    min-width: 180px;
  }
  .brush-label input[type='range'] {
    flex: 1;
    accent-color: var(--accent);
  }
  .brush-label span {
    min-width: 48px;
    color: var(--fg);
  }
</style>
