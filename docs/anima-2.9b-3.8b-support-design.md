# Anima 2.9b / 3.8b 지원 설계안

작성: 2026-10-15 · 상태: 3.8b 구현 완료 (커밋 b712772 이후, GPU 실측 전)
관련: `backend/atelier/engine/anima38.py`, `scripts/inspect_anima.py`

## 1. 개요

ComfyUI에서 지원하는 Anima 2.9b / 3.8b 체크포인트를 Atelier AI에서도
지원하기 위한 설계안이다. 이번 단계는 조사·설계만 하며 앱 코드는 건드리지 않는다.

사용자가 알려준 스펙:

| 변형 | 구조 | 비고 |
|---|---|---|
| anima base | 레이어 28 | 현재 diffusers AutoPipeline으로 로드 중 (SDXL 기반) |
| anima 2.9b | 레이어 40 | base 튜닝, 지식 컷오프 2026년 7월 |
| anima 3.8b | 레이어 52 DiT | 2.9b 튜닝 + Qwen3.5 4B TE/어댑터 추가 (Qwen3 0.6B 병용) |

3.8b 구성 요소 (전부 단일 `.safetensors`, ComfyUI식):

| 파일 | 용도 |
|---|---|
| Anima 3.8B DiT `.safetensors` | 52블록 확산 모델 본체 |
| `qwen35_4b.safetensors` | 추가 텍스트 인코더 (Qwen3.5 4B) |
| `qwen_3_06b_base.safetensors` | 기존 Anima 네이티브 텍스트 인코더 (Qwen3 0.6B) |
| Qwen3.5 Adapter `.safetensors` | Qwen3.5 특징을 DiT에 전달 |
| `qwen35_tokenizer/` 디렉터리 | Qwen3.5 4B 전용 토크나이저 (출처: Qwen/Qwen3.5-4B) |
| Qwen3 0.6B 토크나이저 | 0.6B 전용, `qwen35_tokenizer/`와 공유하지 않음 (출처: Qwen/Qwen3-0.6B) |

어댑터는 구형(`anima_progressive_qwen35_cross_adapter_v1`)과
신형(`anima_qwen35_quality_anchored_semantic_connector_v2`)이 있으며,
구현은 **신형(v2) 기준**으로 한다.

## 2. 변형별 판정

### 2.1 Anima 2.9b — 조건부 가능

40블록 DiT라면 diffusers의 기존 DiT 계열 클래스 중 구조가 맞는 것이
있을 수 있다. 후보 (diffusers 0.41 기준, 설치된 venv에서 확인):

- `SD3Transformer2DModel` — `num_layers`를 config로 받음 (기본 24)
- `HunyuanDiT2DModel`, `PixArtTransformer2DModel`, `FluxTransformer2DModel`,
  `SanaTransformer2DModel`, `AuraFlowTransformer2DModel`,
  `HidreamImageTransformer2DModel`, `Lumina2Transformer2DModel` 등

**가능 경로 A (난이도 소~중):** 텐서 키 레이아웃이 위 클래스 중 하나와
일치하면, 해당 클래스를 `num_layers=40` config로 인스턴스화한 뒤
safetensors 가중치를 키 매핑하여 로드. 이후 기존 `Pipeline` 코드 경로를
그대로 사용 (AutoPipeline이 체크포인트별로 파이프라인을 자동 판별하므로
`generate.py` 수정이 최소화됨).

**경로 B (난이도 중):** 키 레이아웃이 독자적이면 3.8b와 같은 커스텀 DiT
로더를 40블록용으로 구현.

판정 확정 조건: `scripts/inspect_anima.py` 덤프의 **텐서 키 이름**.
키 이름 없이는 A/B를 가릴 수 없다.

### 2.2 Anima 3.8b — 커스텀 파이프라인 필요 (AutoPipeline 불가)

세 가지 이유:

1. **52블록 커스텀 DiT** — diffusers에 없는 아키텍처. 어떤 기존 클래스의
   config 매핑으로도 커버되지 않음.
2. **듀얼 텍스트 인코더** (Qwen3 0.6B + Qwen3.5 4B) — diffusers 파이프라인은
   고정된 TE 구성을 가정하므로 두 개의 서로 다른 TE + 커스텀 조합을
   표현할 수 없음.
3. **v2 커넥터** — Qwen3.5 특징을 DiT 컨디셔닝에 주입하는 커스텀 모듈.

긍정 요소: 설치된 transformers 5.19에 `Qwen3_5TextModel`,
`Qwen3Model`, `tokenization_qwen3_5`가 이미 들어있다 (venv에서 직접 확인).
즉 TE 로딩은 **새 의존성 없이** 기존 스택으로 가능하다.

## 3. 3.8b 파이프라인 설계

### 3.1 모듈 구조

`backend/atelier/engine/anima38.py` (신규, 기존 `pipeline.py` 컨벤션을 따름):

```
AnimaDiTConfig          # dataclass. 덤프에서 확정:
                        # num_layers=52, hidden_size, num_heads,
                        # mlp_ratio, patch_size, in_channels 등
AnimaDiT(nn.Module)     # patch_embed, timestep embed(adaLN),
                        # 52× AnimaDiTBlock, final_layer
QwenTextEncoders        # Qwen3Model(0.6B) + Qwen3_5TextModel(4B).
                        # transformers는 함수 내 lazy import.
                        # output_hidden_states=True, no_grad, fp16
Qwen35ConnectorV2       # anima_qwen35_quality_anchored_semantic_connector_v2.
                        # 내부 구조는 덤프 보고 확정 (선형 투영+어텐션 계열로 추정)
Anima38Pipeline         # load() / encode_prompt() / generate()
```

`load()` 순서: GPU 확인(`_default_device`, 없으면 RuntimeError→503)
→ DiT 가중치 (safetensors→키 remap→`load_state_dict`)
→ TE 2개 → `qwen35_tokenizer/` → v2 커넥터 → VAE → 스케줄러.

`encode_prompt(prompt, negative_prompt)`:
두 TE로 각각 인코딩한 뒤 Qwen3.5 측은 v2 커넥터를 통과시켜 DiT
컨디셔닝으로 변환. 네거티브 프롬프트도 동일 경로로 인코딩 (CFG용).

`generate(params)`:
노이즈 → DiT 샘플링 루프 (diffusers 스케줄러 + 기존
`samplers.apply_scheduler` 재사용. Anima 공식 50스텝·SDE 계열 권장값은
`settings.py` 기존 값 사용)
→ VAE 디코드 → 기존 `_save()` 패턴으로 저장.

컨벤션 준수:

- torch / transformers / diffusers는 함수·메서드 안에서 lazy import
  (모듈 import 시 GPU·torch 불필요 — `pipeline.py` 패턴).
- GPU가 없으면 `RuntimeError` → API 503 (기존 패턴).
- 새 설정 키가 필요하면 `settings.validate()`에 등록 (모르는 키 422 패턴).
  기본 prompt/steps/cfg 키는 재사용하고 새 키는 최소화.

### 3.2 TE / 어댑터 와이어링 (가정 — 덤프 후 확정)

```
prompt ──▶ qwen35_tokenizer/ ──▶ Qwen3.5-4B ──▶ hidden states
                                              ──▶ v2 connector ──▶ DiT cond 입력
prompt ──▶ Qwen3-0.6B 토크나이저 ──▶ Qwen3-0.6B ──▶ 네이티브 cond
```
(2026-10-15 세렌 확인: 두 TE의 토크나이저는 별개이며 공유하지 않음)

미확정 사항:

- 0.6B 출력 사용 방식: pooled(문장 벡터) vs sequence(토큰별). 덤프의
  DiT cond 관련 키 shape으로 판단.
- v1 어댑터는 구형이므로 미지원으로 명시 (v2만 구현).

### 3.3 VAE / 스케줄러

- VAE: base와 공유한다는 가정. 미확인 — 사용자에게 확인 필요.
  별도 파일이면 모델 팩에 포함.
- 스케줄러: diffusers 스케줄러를 그대로 사용.

### 3.4 모델 매니저 통합안

3.8b는 4개 파일 + 토크나이저 디렉터리가 한 세트이므로 "팩" 단위 인식이
필요하다. 제안:

```
models/anima38/<팩이름>/
  ├── Anima-3.8B-v1.1.safetensors  # DiT 본체 (실제 파일명, 2026-10-15 세렌 확인)
  ├── qwen35_4b.safetensors
  ├── qwen_3_06b_base.safetensors
  ├── Anima-3.8B-expanded_adapter.safetensors  # v2 커넥터 (실제 파일명, 2026-10-15 세렌 확인)
  ├── qwen35_tokenizer/          # Qwen3.5 4B 전용 (Qwen/Qwen3.5-4B)
  └── qwen3_06b_tokenizer/       # Qwen3 0.6B 전용 (Qwen/Qwen3-0.6B), 공유하지 않음
```

`models.py`의 `_scan()`을 확장해 이 레이아웃을 `kind="anima38pack"`으로
인식. 2.9b 단일 `.safetensors`는 기존 `checkpoint`로 인식하고 로드 시
DiT 분기로 처리.

`/api/generate`는 모델 kind에 따라 기존 `Pipeline`과
`Anima38Pipeline` 중 선택 (generate.py 분기 — 구현 단계에서).

## 4. 의존성

| 패키지 | 상태 | 용도 |
|---|---|---|
| transformers 5.19 | 이미 있음 | `Qwen3_5TextModel`, `Qwen3Model` (venv에서 확인) |
| tokenizers 0.23.3 | 이미 있음 | `qwen35_tokenizer/` 로드 |
| safetensors | 이미 있음 | DiT/어댑터 가중치 로드 |
| diffusers 0.41 | 이미 있음 | 스케줄러, VAE |
| accelerate | 없음 (선택) | CPU offload 시에만 고려 |

**신규 필수 의존성 없음.**

## 5. VRAM 예상

| 구성 요소 | fp16 기준 |
|---|---|
| DiT 3.8B | ~7.6 GB |
| Qwen3.5 4B TE | ~8.0 GB |
| Qwen3 0.6B TE | ~1.2 GB |
| VAE | ~0.2–0.8 GB |
| activation (1024px, 52L) | ~4–8 GB |
| **합계 (offload 없음)** | **약 21–25 GB** |

기존 결론(16GB 필요)과 비교: 8bit TE 양자화나 CPU offload를 쓰면
16GB에서도 가능할 수 있으나 속도가 크게 저하됨. 실측은 **24GB+ 환경
(RunPod 등)** 권장. 세렌 노트북(GTX 1050 4GB)에서는 실행 불가.

2.9b는 DiT 2.9B + TE 1개면 12–16GB대에서 가능할 수도 있음 (덤프 후 확정).

## 6. 테스트 계획 (모킹 기반, GPU 없이)

`backend/tests/test_anima38.py` (신규):

- DiT: tiny config(2 layers, hidden 32)로 forward shape 테스트.
- TE: transformers 모킹 — 고정된 fake hidden states 반환.
- 커넥터: 입출력 shape 테스트.
- 파이프라인: VAE·스케줄러 모킹 후 `generate()` 호출→저장까지.
- GPU 없음 → `RuntimeError` → 503 케이스.
- 설정 검증 422 케이스.
- `inspect_anima.py`: 가짜 safetensors 픽스처로 덤프 파싱 테스트 (선택).

## 7. 단계별 작업 분해

| 단계 | 내용 | 난이도 | 선행 조건 |
|---|---|---|---|
| P0 | `inspect_anima.py` 실행 결과 확보 (사용자) | — | **블로커** |
| P1 | 2.9b 덤프 분석 → diffusers 클래스 매핑 시도 | 소~중 | P0 |
| P2 | 3.8b DiT 아키텍처 확정 + `AnimaDiT` 구현 | 대 | P0 |
| P3 | safetensors → state_dict 키 매핑 | 중 | P2 |
| P4 | TE 로딩 (`Qwen3_5TextModel`/`Qwen3Model`) + 토크나이저 | 소~중 | P0 |
| P5 | v2 커넥터 구현 | 중~대 | P0 |
| P6 | `Anima38Pipeline` 통합 (샘플링 루프·VAE·스케줄러) | 중 | P2–P5 |
| P7 | 모델 매니저 팩 인식 + `/api/generate` 분기 + 프론트 | 중 | P6 |
| P8 | RunPod GPU 실측 | 중 | P7 |

P1이 성공하면 2.9b는 P2–P6 없이 조기 지원 가능.

## 8. 핵심 불확실성 (리스크)

1. **텐서 키 이름·shape** — 덤프(P0)로 해결. 없이는 매핑 확정 불가.
2. DiT 블록 내부 구조 (adaLN 방식, cond 주입 위치).
3. 0.6B 네이티브 TE 출력 사용 방식 및 전용 토크나이저 유무.
4. v2 커넥터 내부 구조.
5. VAE 종류 (base 공유 여부).
6. 라이선스: Qwen 가중치는 Apache 2.0 — MIT 프로젝트와 충돌 없음.
   단, 가중치 파일 자체는 저장소에 포함하지 않고 사용자가 직접 배치
   (ComfyUI와 동일한 방식).

## 9. 사용자에게 필요한 것

체크포인트가 있는 머신에서 실행 (Python 3.11+, 별도 pip 설치 불필요):

```sh
python3 scripts/inspect_anima.py <DiT>.safetensors qwen35_4b.safetensors \
  <qwen3-0.6b>.safetensors <adapter-v2>.safetensors \
  --tokenizer-dir qwen35_tokenizer/ --json-out anima38_dump.json
```

`anima38_dump.json` (또는 콘솔 출력 전체)을 전달하면 P1–P5의
매핑을 확정할 수 있다. (2026-10-15 확인됨: VAE는 전부 공유, 두 TE의 토크나이저는 별개로 공유하지 않음)
