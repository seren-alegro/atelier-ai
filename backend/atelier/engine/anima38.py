"""Anima 3.8b 커스텀 파이프라인 (DiT 52블록 + 듀얼 Qwen TE + v2 커넥터).

diffusers AutoPipeline으로 로드할 수 없는 커스텀 아키텍처이므로
torch 모듈을 직접 정의하고 safetensors 가중치를 키 매핑으로 로드한다.

클린룸: 아래 모듈은 2026-10-15 덤프(anima38_dump.json)의 텐서 키 이름·
shape에서 확인된 구조만으로 직접 작성했다. 외부 코드를 복사하지 않았다.

torch / transformers / safetensors / diffusers는 함수 안에서 lazy import
하므로 이 모듈 자체는 GPU·torch 없이 import 가능하다 (pipeline.py 패턴).
torch가 필요한 nn.Module 정의는 build_* 팩토리 함수 안에 둔다
(upscale.py의 _build_rrdbnet 패턴).

== v1.1 실측 사항 (2026-10-20 포드 헤더 덤프) ==
- DiT (anima38B_11.safetensors): `net.blocks` 0..51 (52블록),
  hidden 2048, MLP 8192 (layer1 [8192,2048] / layer2 [2048,8192]).
- 블록당 adaLN-Zero 3종: adaln_modulation_{self_attn,cross_attn,mlp} =
  Sequential(SiLU, Linear(2048,256), Linear(256,6144)) (bias 없음,
  6144 = shift/scale/gate × 2048).
- self_attn: q/k/v/output_proj [2048,2048], q_norm/k_norm [128] (per-head).
- cross_attn: q [2048,2048], k/v [2048,1024], output [2048,2048],
  q/k_norm [128] (16 q-heads / 8 kv-heads, GQA).
- x_embedder.proj = Sequential(?, Linear(68,2048)) (bias 없음).
  → 입력은 (B,T,68) 토큰. 68 = 16ch×2×2=64 + 4 (패딩, 추정).
- t_embedder = [파라미터 없음, MLP(linear_1 [2048,2048] /
  linear_2 [2048,6144])], t_embedding_norm [2048] (RMSNorm).
  → temb(2048)을 6144=3×2048로 만들어 각 adaln에 분배 (추정).
- pos_embedder: dim_spatial_range [21], dim_temporal_range [22], seq [256].
- final_layer: adaln Sequential(SiLU, 2048→256→4096) (4096=shift/scale×2048)
  + linear [64,2048] (16ch×2×2).
- llm_adapter: 6× 트랜스포머 블록
  (self/cross attn q/k/v/o [1024,1024]+q/k_norm [64],
   mlp.0 [4096,1024]+b / mlp.2 [1024,4096]+b, norm×3 [1024]),
  embed [32128,1024], norm [1024], out_proj [1024,1024]+b.
- v2 커넥터 (DiT 파일 내장 + 독립 어댑터 파일, 동일 구조):
  quality_anchor {query_norms [1024]×6, source_norms [2560]×6,
  semantic_attentions×6 (q [1024,1024] / k/v [1024,2560] /
  o [1024,1024], q/k_norm [64]), layer_mix_logits [6,4]},
  semantic_resampler {query_tokens [1,64,2048],
  layer_embeddings [4,1,2560], blocks×6
  (self attn q/k/v/o [2048,2048],
   cross attn q [2048,2048]/k/v [2048,2560]/o [2048,2048],
   mlp_in [11264,2048]/mlp_out [2048,5632] (gated),
   source_norm [2560] LayerNorm, time_modulation [12288,2048]+b),
  time_mlp = Sequential(Linear(2048,2048)+b, SiLU, Linear+b),
  output_norm [2048] LayerNorm, output_projection [1024,2048]},
  v2_attentions×6 (q/k/v/o [1024,1024], q/k_norm [64]),
  v2_query_norms/v2_semantic_norms [1024]×6.
- Qwen3.5 4B TE: 키 prefix 없음 (`embed_tokens.weight`, `layers.N.*`),
  32 레이어, hidden 2560, 245개 텐서가 FP8_E4M3.
- Qwen3 0.6B TE: 표준 Qwen3 레이아웃 (`model.layers` 28개, hidden 1024).
- 토크나이저 분리: 0.6B용 Qwen/Qwen3-0.6B, 4B용 qwen35_tokenizer/ (공유 안 함).
- VAE는 base와 공유.

== 미확인 사항 ==
- x_embedder 입력 68차원의 정확한 구성 (64+4 패딩으로 가정).
- temb 6144 분배 방식 (3×2048 chunk로 가정).
- v2 커넥터의 정확한 데이터플로우 (모듈 구조는 키 이름대로, 연산 순서는 추정).
- FP8 TE의 transformers 직접 로드 가능 여부 (폴백 구현됨).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 설정 (torch 불필요)
# ---------------------------------------------------------------------------

@dataclass
class AnimaDiTConfig:
    """DiT 설정. v1.1 체크포인트 실측 shape 기준 (2026-10-20 포드에서 헤더 덤프)."""

    num_layers: int = 52        # 실측: net.blocks 0..51
    hidden_size: int = 2048     # 실측: attn/MLP 주 차원
    mlp_dim: int = 8192        # 실측: mlp.layer1 [8192,2048]
    num_heads: int = 16         # 실측: 2048/128, q_norm [128]
    head_dim: int = 128         # 실측: q_norm/k_norm [128]
    cond_dim: int = 1024        # 실측: cross k/v [2048,1024]
    patch_size: int = 2         # 실측: final linear [64,2048]=16ch×2×2
    in_channels: int = 16       # 실측 추정: 출력 64=16×2×2
    # v2 커넥터
    connector_layers: int = 6   # 실측: 각 6개
    connector_dim: int = 1024   # 실측
    resampler_dim: int = 2048   # 실측
    resampler_mlp_dim: int = 5632  # 실측: 11264 = 2 × 5632
    te35_hidden: int = 2560     # 실측: Qwen3.5 4B hidden
    te06_hidden: int = 1024     # 실측: Qwen3 0.6B hidden


# 덤프에서 확인된 실제 키 이름 (테스트·매핑용)
DIT_PREFIX = "net."
CONFIRMED_DIT_KEYS = (
    "net.blocks",                       # 0..51 (indexed)
    "net.x_embedder.proj",
    "net.t_embedder",
    "net.final_layer.adaln_modulation",  # 2개
    "net.llm_adapter.blocks",           # 0..5 (indexed)
    "net.anima_v2_connector.quality_anchor.layer_mix_logits",
    "net.anima_v2_connector.quality_anchor.query_norms",      # 0..5
    "net.anima_v2_connector.quality_anchor.source_norms",     # 0..5
    "net.anima_v2_connector.quality_anchor.semantic_attentions",  # 0..5
    "net.anima_v2_connector.semantic_resampler.blocks",       # 0..5
    "net.anima_v2_connector.semantic_resampler.time_mlp",
    "net.anima_v2_connector.v2_attentions",                   # 0..5
    "net.anima_v2_connector.v2_query_norms",                  # 0..5
    "net.anima_v2_connector.v2_semantic_norms",               # 0..5
)
# semantic_attentions 블록의 덤프 실측 키 suffix
CONFIRMED_SA_SUFFIXES = (
    "q_norm.weight",   # [64]
    "q_proj.weight",   # [1024,1024]
    "k_norm.weight",   # [64]
    "k_proj.weight",   # [1024,2560]
    "v_proj.weight",   # [1024,2560]
    "o_proj.weight",   # [1024,1024]
)


# ---------------------------------------------------------------------------
# 팩(모델 폴더) 감지 — torch 불필요, models.py에서 사용
# ---------------------------------------------------------------------------

def _names_lower(dir_path: str) -> tuple[set[str], set[str]]:
    """(파일명 소문자 집합, 디렉토리명 소문자 집합)."""
    files, dirs = set(), set()
    try:
        with os.scandir(dir_path) as it:
            for e in it:
                n = e.name.lower()
                if e.is_file():
                    files.add(n)
                elif e.is_dir():
                    dirs.add(n)
    except OSError:
        pass
    return files, dirs


def looks_like_pack_dir(dir_path: str) -> bool:
    """팩 '후보' 디렉토리인지 판정 (파일이 일부 없어도 목록 표시용).

    조건: .safetensors가 1개 이상 있고, qwen35_4b.safetensors가 있거나
    디렉토리명에 '3.8'/'anima38'이 포함됨. (대소문자 무시)
    완전한 팩 판정은 is_anima38_pack_dir 사용.
    """
    if not os.path.isdir(dir_path):
        return False
    files, _ = _names_lower(dir_path)
    if not any(f.endswith(".safetensors") for f in files):
        return False
    name = os.path.basename(dir_path).lower()
    return ("qwen35_4b.safetensors" in files
            or "3.8" in name or "anima38" in name)


def _is_dit_filename(name: str) -> bool:
    """DiT 본체 파일명 판정.

    Civitai 공식 파일(anima38B_11.safetensors)은 '3.8'이 아닌 '38b'
    형태이므로 둘 다 인정한다. 어댑터·2.9b 파일은 제외.
    """
    n = name.lower()
    if not n.endswith(".safetensors"):
        return False
    if "adapter" in n:
        return False
    if "2.9" in n or "29b" in n:
        return False
    return "3.8" in n or "38b" in n


def is_anima_dit_file(path: str) -> bool:
    """단일 파일 Anima DiT 체크포인트인지 (2b/2.9b/3.8b).

    `net.blocks.N.` 키가 있으면 Anima DiT로 판정.
    """
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("net.blocks."):
                    return True
    except Exception:
        pass
    return False


def is_anima38_pack_dir(dir_path: str) -> bool:
    """3.8b 팩 디렉토리인지 판정.

    조건: qwen35_4b.safetensors 존재 + DiT 본체 파일 존재.
    (대소문자 무시)
    """
    if not os.path.isdir(dir_path):
        return False
    files, _ = _names_lower(dir_path)
    has_te35 = "qwen35_4b.safetensors" in files
    has_dit = any(_is_dit_filename(f) for f in files)
    return has_te35 and has_dit


def pack_status(dir_path: str) -> dict:
    """팩 구성 요소별 존재 여부. 파일이 없어도 목록 표시는 되게 한다."""
    files, dirs = _names_lower(dir_path)
    dit = sorted(f for f in files if _is_dit_filename(f))
    adapter = sorted(f for f in files
                     if f.endswith(".safetensors") and "adapter" in f)
    tok_dirs = sorted(d for d in dirs if "tokenizer" in d)
    return {
        "dit": dit[0] if dit else None,
        "qwen35_4b": "qwen35_4b.safetensors" in files,
        "qwen3_06b": any("qwen_3_06b" in f and f.endswith(".safetensors")
                         for f in files),
        "adapter": adapter[0] if adapter else None,
        "tokenizer_dirs": tok_dirs,
        "vae_dir": "vae" in dirs,
        "vae_file": next(
            (f for f in files
             if f.endswith(".safetensors") and "vae" in f),
            None,
        ),
    }


def find_dit_file(dir_path: str) -> str | None:
    """DiT 본체 파일의 절대 경로. 없으면 None."""
    st = pack_status(dir_path)
    if not st["dit"]:
        return None
    # 원래 대소문자로 복원
    for e in os.scandir(dir_path):
        if e.is_file() and e.name.lower() == st["dit"]:
            return e.path
    return None


def _fallback_te_path(role: str) -> str | None:
    """팩에 없는 TE/어댑터 파일을 models/text_encoders/에서 찾는다.

    role: "qwen35_4b" | "qwen3_06b" | "adapter".
    사용자가 선택한 파일(POST /api/models/text-encoder)이 역할에 맞으면
    우선하고, 없으면 파일명 패턴으로 첫 후보를 쓴다.
    """
    try:
        from atelier.api.routes import models as models_module
        te_dir = os.path.join(models_module.models_dir(), "text_encoders")
        selected = models_module.get_selected_text_encoder() or ""
    except Exception:  # noqa: BLE001 - 폴백은 조용히 포기
        return None
    if not os.path.isdir(te_dir):
        return None
    cands = sorted(
        f for f in os.listdir(te_dir)
        if f.lower().endswith(".safetensors") and not f.startswith(".")
    )

    def matches(f: str) -> bool:
        n = f.lower()
        if role == "qwen35_4b":
            return "qwen35" in n
        if role == "qwen3_06b":
            return "qwen_3_06b" in n or "qwen3_06b" in n
        return "adapter" in n

    sel_base = os.path.basename(selected).lower()
    for f in cands:  # 1) 사용자 선택 우선
        if f.lower() == sel_base and matches(f):
            return os.path.join(te_dir, f)
    for f in cands:  # 2) 패턴 매칭 첫 후보
        if matches(f):
            return os.path.join(te_dir, f)
    return None


def select_tokenizer_dirs(tokenizer_dirs: list[str]) -> tuple[str, str]:
    """(Qwen3.5 4B용, Qwen3 0.6B용) 토크나이저 디렉토리명 선택.

    두 TE의 토크나이저는 별개이며 공유하지 않는다 (2026-10-15 세렌 확인).
    둘 중 하나라도 없으면 RuntimeError (한국어).
    """
    tok35 = next((d for d in tokenizer_dirs if "35" in d), None)
    tok06 = next((d for d in tokenizer_dirs if d != tok35), None)
    if tok35 is None or tok06 is None:
        raise RuntimeError(
            "토크나이저 디렉토리가 2개 필요해요 "
            "(Qwen3.5 4B용 qwen35_tokenizer/ + Qwen3 0.6B용). "
            f"현재: {tokenizer_dirs or '없음'}")
    return tok35, tok06


# ---------------------------------------------------------------------------
# safetensors 로더 — torch는 함수 내 lazy import
# ---------------------------------------------------------------------------

def load_safetensors_mapped(path: str, key_map: dict[str, str] | None = None,
                            prefix_strip: str = "") -> dict:
    """safetensors 파일을 읽어 {모듈키: 텐서} dict로 반환.

    key_map: {파일키: 모듈키}. None이면 파일키 그대로(앞 prefix만 제거).
    torch는 여기서 import하므로 모듈 import 시점에는 불필요.
    """
    import torch
    from safetensors.torch import load_file

    raw = load_file(path, device="cpu")
    out: dict = {}
    for fkey, tensor in raw.items():
        mkey = key_map.get(fkey, fkey) if key_map else fkey
        if prefix_strip and mkey.startswith(prefix_strip):
            mkey = mkey[len(prefix_strip):]
        # FP8 텐서는 float32/bf16로 변환해 둔다 (TE 로더가 별도 처리하므로
        # 여기서는 안전하게 float32로)
        if str(tensor.dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
            tensor = tensor.to(torch.float32)
        out[mkey] = tensor
    return out


def apply_state_dict(module, state: dict, prefix: str = "") -> dict:
    """모듈에 state_dict를 적용하고 {missing, unexpected}를 보고한다.

    strict=False로 로드하므로, 덤프에 없는 키(미확인 구조)는 missing으로
    남아 보고서에 명시된다.
    """
    result = module.load_state_dict(state, strict=False)
    missing = [prefix + k for k in result.missing_keys]
    unexpected = [prefix + k for k in result.unexpected_keys]
    return {"missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------------------
# DiT 모듈 — torch 필요이므로 팩토리 안에서 정의 (upscale.py 패턴)
# ---------------------------------------------------------------------------

def build_anima_dit(config: AnimaDiTConfig | None = None):
    """AnimaDiT 모듈을 생성한다. torch는 여기서 import.

    v1.1 체크포인트 실측 shape 기준 (2026-10-20 포드 헤더 덤프):
    - x_embedder.proj: Sequential(Identity, Linear(68, 2048, bias=False))
    - t_embedder: ModuleList([Identity, TEmbedderMLP(linear_1/linear_2)])
    - t_embedding_norm: [2048] (RMSNorm)
    - pos_embedder: 버퍼 3개 (dim_spatial_range [21] 등)
    - 52× 블록: self_attn q/k/v/o [2048,2048] + q/k_norm [128],
      cross_attn q [2048,2048] + k/v [2048,1024] + o [2048,2048],
      mlp layer1 [8192,2048]/layer2 [2048,8192],
      adaln 3종 = Sequential(SiLU, Linear(2048,256), Linear(256,6144))
      (6144 = shift/scale/gate × 2048)
    - final_layer: adaln Sequential(SiLU, 2048→256→4096=shift/scale×2048)
      + linear [64,2048] (16ch×2×2)
    - llm_adapter: 6× 트랜스포머 블록 (1024-dim)
    - anima_v2_connector: 아래 build_connector_v2 참조
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    cfg = config or AnimaDiTConfig()
    H = cfg.hidden_size
    NH, HD = cfg.num_heads, cfg.head_dim
    CD = cfg.cond_dim

    class _RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), self.weight)

    class _WeightParam(nn.Module):
        """단일 weight 파라미터 (`*.weight` 키와 정확히 매칭)."""

        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    class SelfAttention(nn.Module):
        """실측: q/k/v/output_proj [2048,2048], q_norm/k_norm [128] (per-head)."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, H, bias=False)
            self.k_proj = nn.Linear(H, H, bias=False)
            self.v_proj = nn.Linear(H, H, bias=False)
            self.q_norm = _RMSNorm(HD)
            self.k_norm = _RMSNorm(HD)
            self.output_proj = nn.Linear(H, H, bias=False)

        def forward(self, x):
            b, t, _ = x.shape
            q = self.q_norm(self.q_proj(x).view(b, t, NH, HD))
            k = self.k_norm(self.k_proj(x).view(b, t, NH, HD))
            v = self.v_proj(x).view(b, t, NH, HD)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return self.output_proj(out.transpose(1, 2).reshape(b, t, H))

    class CrossAttention(nn.Module):
        """실측: q [2048,2048], k/v [2048,1024], o [2048,2048].

        16 q-heads × 8 kv-heads (GQA) — k/v를 repeat해서 맞춘다.
        """

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, H, bias=False)
            self.k_proj = nn.Linear(CD, H, bias=False)
            self.v_proj = nn.Linear(CD, H, bias=False)
            self.q_norm = _RMSNorm(HD)
            self.k_norm = _RMSNorm(HD)
            self.output_proj = nn.Linear(H, H, bias=False)

        def forward(self, x, cond):
            b, t, _ = x.shape
            n_kv = max(1, CD // HD)
            rep = max(1, NH // n_kv)
            q = self.q_norm(self.q_proj(x).view(b, t, NH, HD)).transpose(1, 2)
            k = self.k_norm(
                self.k_proj(cond).view(b, -1, n_kv, HD)).transpose(1, 2)
            v = self.v_proj(cond).view(b, -1, n_kv, HD).transpose(1, 2)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.output_proj(out.transpose(1, 2).reshape(b, t, H))

    class _Mlp(nn.Module):
        """실측: mlp.layer1 [8192,2048], mlp.layer2 [2048,8192]."""

        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(H, cfg.mlp_dim, bias=False)
            self.layer2 = nn.Linear(cfg.mlp_dim, H, bias=False)

        def forward(self, x):
            return self.layer2(F.gelu(self.layer1(x), approximate="tanh"))

    def _adaln_mlp(out_dim: int):
        # 실측: Sequential(SiLU, Linear(2048,256), Linear(256,out_dim)), bias 없음
        return nn.Sequential(
            nn.SiLU(), nn.Linear(H, 256, bias=False),
            nn.Linear(256, out_dim, bias=False))

    class AnimaDiTBlock(nn.Module):
        """DiT 블록 1개 — adaLN-Zero (경로별 shift/scale/gate)."""

        def __init__(self):
            super().__init__()
            # 블록 norm 가중치가 체크포인트에 없음 → affine 없는 LayerNorm
            self.norm1 = nn.LayerNorm(H, elementwise_affine=False)
            self.self_attn = SelfAttention()
            self.norm2 = nn.LayerNorm(H, elementwise_affine=False)
            self.cross_attn = CrossAttention()
            self.norm3 = nn.LayerNorm(H, elementwise_affine=False)
            self.mlp = _Mlp()
            self.adaln_modulation_self_attn = _adaln_mlp(3 * H)
            self.adaln_modulation_cross_attn = _adaln_mlp(3 * H)
            self.adaln_modulation_mlp = _adaln_mlp(3 * H)

        def forward(self, x, t_sa, t_xa, t_mlp, cond):
            b = x.shape[0]
            shift, scale, gate = self.adaln_modulation_self_attn(
                t_sa).view(b, 3, H).unbind(1)
            h = _modulate(self.norm1(x), shift.unsqueeze(1), scale.unsqueeze(1))
            x = x + gate.unsqueeze(1) * self.self_attn(h)
            if cond is not None:
                shift, scale, gate = self.adaln_modulation_cross_attn(
                    t_xa).view(b, 3, H).unbind(1)
                h = _modulate(self.norm2(x), shift.unsqueeze(1),
                              scale.unsqueeze(1))
                x = x + gate.unsqueeze(1) * self.cross_attn(h, cond)
            shift, scale, gate = self.adaln_modulation_mlp(
                t_mlp).view(b, 3, H).unbind(1)
            h = _modulate(self.norm3(x), shift.unsqueeze(1), scale.unsqueeze(1))
            return x + gate.unsqueeze(1) * self.mlp(h)

    class _Attn1024(nn.Module):
        """llm_adapter용 1024-dim attention (q/k/v/o [1024,1024], norm [64])."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(1024, 1024, bias=False)
            self.k_proj = nn.Linear(1024, 1024, bias=False)
            self.v_proj = nn.Linear(1024, 1024, bias=False)
            self.q_norm = _RMSNorm(64)
            self.k_norm = _RMSNorm(64)
            self.o_proj = nn.Linear(1024, 1024, bias=False)

        def forward(self, x, context=None):
            b, t, _ = x.shape
            kv = x if context is None else context
            s = kv.shape[1]
            nh = 1024 // 64
            q = self.q_norm(self.q_proj(x).view(b, t, nh, 64))
            k = self.k_norm(self.k_proj(kv).view(b, s, nh, 64))
            v = self.v_proj(kv).view(b, s, nh, 64)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return self.o_proj(out.transpose(1, 2).reshape(b, t, 1024))

    class LLMAdapterBlock(nn.Module):
        """llm_adapter 블록 실측: self/cross attn + mlp(0/2) + norm 3개."""

        def __init__(self):
            super().__init__()
            self.self_attn = _Attn1024()
            self.cross_attn = _Attn1024()
            self.mlp = nn.Sequential(
                nn.Linear(1024, 4096), nn.SiLU(), nn.Linear(4096, 1024))
            self.norm_self_attn = _RMSNorm(1024)
            self.norm_cross_attn = _RMSNorm(1024)
            self.norm_mlp = _RMSNorm(1024)

        def forward(self, x, context=None):
            x = x + self.self_attn(self.norm_self_attn(x))
            if context is not None:
                x = x + self.cross_attn(self.norm_cross_attn(x), context)
            return x + self.mlp(self.norm_mlp(x))

    class TEmbedderMLP(nn.Module):
        """t_embedder[1] 실측: linear_1 [2048,2048], linear_2 [2048,6144]."""

        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(H, H, bias=False)
            self.linear_2 = nn.Linear(H, 3 * H, bias=False)

        def forward(self, x):
            return self.linear_2(F.silu(self.linear_1(x)))

    class XEmbedder(nn.Module):
        """x_embedder 실측: proj = Sequential(?, Linear(68, 2048))."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Identity(), nn.Linear(68, H, bias=False))

        def forward(self, x):
            return self.proj(x)

    class AnimaDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.x_embedder = XEmbedder()
            # t_embedder[0]은 파라미터 없음, [1]이 linear_1/linear_2
            self.t_embedder = nn.ModuleList([nn.Identity(), TEmbedderMLP()])
            self.t_embedding_norm = _RMSNorm(H)
            self.pos_embedder = nn.Module()
            self.pos_embedder.register_buffer(
                "dim_spatial_range", torch.zeros(21))
            self.pos_embedder.register_buffer(
                "dim_temporal_range", torch.zeros(22))
            self.pos_embedder.register_buffer("seq", torch.zeros(256))
            self.blocks = nn.ModuleList([AnimaDiTBlock()
                                         for _ in range(cfg.num_layers)])
            # llm_adapter 실측 키: llm_adapter.blocks.0~5 / .embed / .norm / .out_proj
            _llm_adapter = nn.Module()
            _llm_adapter.blocks = nn.ModuleList([LLMAdapterBlock()
                                                 for _ in range(6)])
            _llm_adapter.embed = nn.Embedding(32128, 1024)
            _llm_adapter.norm = _RMSNorm(1024)
            _llm_adapter.out_proj = nn.Linear(1024, 1024)
            self.llm_adapter = _llm_adapter
            self.anima_v2_connector = build_connector_v2(cfg)
            self.final_layer = nn.ModuleDict({
                "adaln_modulation": _adaln_mlp(2 * H),
                "linear": nn.Linear(H, 64, bias=False),
            })

        def timestep_embedding(self, timesteps, dim=2048, max_period=10000):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(half, device=timesteps.device) / half)
            args = timesteps[:, None].float() * freqs[None]
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        def forward(self, x, timesteps, cond=None):
            # x: (B, C, h, w) latent → patchify(p=2) → (B, T, C*4) → 68로 맞춤
            b, c, h, w = x.shape
            p = cfg.patch_size
            gh, gw = h // p, w // p
            tok = x.unfold(2, p, p).unfold(3, p, p)
            tok = tok.permute(0, 2, 3, 1, 4, 5).reshape(b, gh * gw, c * p * p)
            if tok.shape[-1] < 68:
                tok = F.pad(tok, (0, 68 - tok.shape[-1]))
            elif tok.shape[-1] > 68:
                tok = tok[..., :68]
            h_tok = self.x_embedder(tok)  # (B, T, 2048)
            # temb 3종: t_embedder[1] 출력(6144)을 3×2048로 나눠 각 adaln에
            temb = self.t_embedding_norm(self.timestep_embedding(timesteps, dim=H))
            temb3 = self.t_embedder[1](temb)
            t_sa, t_xa, t_mlp = temb3.chunk(3, dim=-1)
            for blk in self.blocks:
                h_tok = blk(h_tok, t_sa, t_xa, t_mlp, cond)
            shift, scale = self.final_layer["adaln_modulation"](
                temb).view(b, 2, H).unbind(1)
            h_tok = _modulate(F.layer_norm(h_tok, (H,)),
                              shift.unsqueeze(1), scale.unsqueeze(1))
            out = self.final_layer["linear"](h_tok)  # (B, T, 64)
            return out, (b, gh, gw)

    return AnimaDiT()


# ---------------------------------------------------------------------------
# v2 커넥터 — torch 필요이므로 팩토리 안에서 정의
# ---------------------------------------------------------------------------

def build_connector_v2(config: AnimaDiTConfig | None = None):
    """anima_qwen35_quality_anchored_semantic_connector_v2.

    v1.1 체크포인트 실측 shape 기준 (2026-10-20 포드 헤더 덤프):
    - quality_anchor: query_norms [1024]×6, source_norms [2560]×6,
      semantic_attentions×6 (q/k/v/o [1024,1024] except k/v [1024,2560],
      q/k_norm [64]), layer_mix_logits [6,4]
    - semantic_resampler: query_tokens [1,64,2048],
      layer_embeddings [4,1,2560], blocks×6
      (self_attention q/k/v/o [2048,2048],
       cross_attention q [2048,2048]/k/v [2048,2560]/o [2048,2048],
       mlp_in [11264,2048]/mlp_out [2048,5632] (gated),
       source_norm [2560] (LayerNorm), time_modulation [12288,2048]+b),
      time_mlp = Sequential(Linear(2048,2048)+b, SiLU, Linear+b),
      output_norm [2048] (LayerNorm), output_projection [1024,2048]
    - v2_attentions×6: q/k/v/o [1024,1024], q/k_norm [64]
    - v2_query_norms/v2_semantic_norms [1024]×6

    연산 순서는 추정. 모듈 계층·파라미터 이름·shape은 실측과 일치.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = config or AnimaDiTConfig()
    D, T35 = cfg.connector_dim, cfg.te35_hidden
    L = cfg.connector_layers
    RD = cfg.resampler_dim

    class _RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), self.weight)

    class _WeightParam(nn.Module):
        """단일 weight 파라미터 (`*.weight` 키와 정확히 매칭)."""

        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    class SemanticAttention(nn.Module):
        """quality_anchor.semantic_attentions[i] (실측 shape)."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(D, D, bias=False)
            self.k_proj = nn.Linear(T35, D, bias=False)
            self.v_proj = nn.Linear(T35, D, bias=False)
            self.q_norm = _RMSNorm(64)
            self.k_norm = _RMSNorm(64)
            self.o_proj = nn.Linear(D, D, bias=False)

        def forward(self, queries, te_hidden):
            b, qn, _ = queries.shape
            t = te_hidden.shape[1]
            nh = max(1, D // 64)
            q = self.q_norm(self.q_proj(queries).view(b, qn, nh, 64))
            k = self.k_norm(self.k_proj(te_hidden).view(b, t, nh, 64))
            v = self.v_proj(te_hidden).view(b, t, nh, 64)
            attn = torch.softmax(
                q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
                / (64 ** 0.5), dim=-1)
            out = attn @ v.transpose(1, 2)
            return self.o_proj(out.transpose(1, 2).reshape(b, qn, D))

    class QualityAnchor(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_norms = nn.ModuleList(
                [_WeightParam(D) for _ in range(L)])
            self.source_norms = nn.ModuleList(
                [_WeightParam(T35) for _ in range(L)])
            self.semantic_attentions = nn.ModuleList(
                [SemanticAttention() for _ in range(L)])
            self.layer_mix_logits = nn.Parameter(torch.zeros(L, 4))

        def forward(self, queries, te_hidden):
            outs = []
            for i in range(L):
                sa = self.semantic_attentions[i]
                q = queries * self.query_norms[i].weight
                src = te_hidden * self.source_norms[i].weight
                outs.append(sa(q, src))
            mix = torch.softmax(self.layer_mix_logits, dim=-1)
            stacked = torch.stack(outs, dim=0)  # (L, B, Q, D)
            w = mix.mean(dim=-1)  # (L,) — 4개 레이어 가중치 평균 (추정)
            return (stacked * w.view(L, 1, 1, 1)).sum(dim=0)

    class ResamplerBlock(nn.Module):
        """semantic_resampler.blocks[i] (실측 shape)."""

        def __init__(self):
            super().__init__()
            self.self_attention = _SelfAttn2048()
            self.cross_attention = _CrossAttn2048()
            self.mlp_in = nn.Linear(RD, 2 * cfg.resampler_mlp_dim,
                                    bias=False)
            self.mlp_out = nn.Linear(cfg.resampler_mlp_dim, RD, bias=False)
            self.source_norm = nn.LayerNorm(T35)
            self.time_modulation = nn.Linear(RD, 6 * RD)  # bias 있음
            self.norm1 = nn.LayerNorm(RD, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(RD, elementwise_affine=False)
            self.norm3 = nn.LayerNorm(RD, elementwise_affine=False)

        def forward(self, x, source, temb):
            b = x.shape[0]
            mod = self.time_modulation(temb).view(b, 6, RD)
            s_sft, s_scl, s_gte = mod[:, 0], mod[:, 1], mod[:, 2]
            m_sft, m_scl, m_gte = mod[:, 3], mod[:, 4], mod[:, 5]
            h = _modulate(self.norm1(x), s_sft.unsqueeze(1), s_scl.unsqueeze(1))
            x = x + s_gte.unsqueeze(1) * self.self_attention(h)
            src = self.source_norm(source)
            x = x + self.cross_attention(self.norm2(x), src)
            h = _modulate(self.norm3(x), m_sft.unsqueeze(1), m_scl.unsqueeze(1))
            g = self.mlp_in(h)
            a, bb = g.chunk(2, dim=-1)
            h = self.mlp_out(a * F.silu(bb))
            return x + m_gte.unsqueeze(1) * h

    class _SelfAttn2048(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(RD, RD, bias=False)
            self.k_proj = nn.Linear(RD, RD, bias=False)
            self.v_proj = nn.Linear(RD, RD, bias=False)
            self.o_proj = nn.Linear(RD, RD, bias=False)

        def forward(self, x):
            b, t, _ = x.shape
            nh = max(1, RD // 128)
            q = self.q_proj(x).view(b, t, nh, 128).transpose(1, 2)
            k = self.k_proj(x).view(b, t, nh, 128).transpose(1, 2)
            v = self.v_proj(x).view(b, t, nh, 128).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.o_proj(out.transpose(1, 2).reshape(b, t, RD))

    class _CrossAttn2048(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(RD, RD, bias=False)
            self.k_proj = nn.Linear(T35, RD, bias=False)
            self.v_proj = nn.Linear(T35, RD, bias=False)
            self.o_proj = nn.Linear(RD, RD, bias=False)

        def forward(self, x, source):
            b, t, _ = x.shape
            s = source.shape[1]
            nh = max(1, RD // 128)
            q = self.q_proj(x).view(b, t, nh, 128).transpose(1, 2)
            k = self.k_proj(source).view(b, s, nh, 128).transpose(1, 2)
            v = self.v_proj(source).view(b, s, nh, 128).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.o_proj(out.transpose(1, 2).reshape(b, t, RD))

    class V2Attention(nn.Module):
        """v2_attentions[i] (실측: q/k/v/o [1024,1024], q/k_norm [64])."""

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(D, D, bias=False)
            self.k_proj = nn.Linear(D, D, bias=False)
            self.v_proj = nn.Linear(D, D, bias=False)
            self.q_norm = _RMSNorm(64)
            self.k_norm = _RMSNorm(64)
            self.o_proj = nn.Linear(D, D, bias=False)

        def forward(self, queries, semantics, q_scale=1.0, s_scale=1.0):
            b, qn, _ = queries.shape
            s = semantics.shape[1]
            nh = max(1, D // 64)
            q = self.q_norm(self.q_proj(queries).view(b, qn, nh, 64))
            k = self.k_norm(self.k_proj(semantics).view(b, s, nh, 64))
            v = self.v_proj(semantics).view(b, s, nh, 64)
            attn = torch.softmax(
                q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
                / (64 ** 0.5) * q_scale * s_scale, dim=-1)
            out = attn @ v.transpose(1, 2)
            return self.o_proj(out.transpose(1, 2).reshape(b, qn, D))

    class SemanticResampler(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([ResamplerBlock() for _ in range(L)])
            self.time_mlp = nn.Sequential(
                nn.Linear(RD, RD), nn.SiLU(), nn.Linear(RD, RD))
            self.query_tokens = nn.Parameter(torch.randn(1, 64, RD))
            self.layer_embeddings = nn.Parameter(torch.randn(4, 1, T35))
            self.output_norm = nn.LayerNorm(RD)
            self.output_projection = nn.Linear(RD, D, bias=False)

        def forward(self, te_hidden, temb):
            b = te_hidden.shape[0]
            x = self.query_tokens.expand(b, -1, -1)
            t = self.time_mlp(temb)
            for blk in self.blocks:
                x = blk(x, te_hidden, t)
            x = self.output_norm(x)
            return self.output_projection(x)  # (B, 64, 1024)

    class ConnectorV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.quality_anchor = QualityAnchor()
            self.semantic_resampler = SemanticResampler()
            self.v2_attentions = nn.ModuleList(
                [V2Attention() for _ in range(L)])
            self.v2_query_norms = nn.ModuleList(
                [_WeightParam(D) for _ in range(L)])
            self.v2_semantic_norms = nn.ModuleList(
                [_WeightParam(D) for _ in range(L)])
            self.latent_queries = nn.Parameter(torch.randn(32, D))

        def forward(self, te35_hidden, temb=None):
            b = te35_hidden.shape[0]
            dev = te35_hidden.device
            queries = self.latent_queries.unsqueeze(0).expand(b, -1, -1)
            anchored = self.quality_anchor(queries, te35_hidden)  # (B,32,1024)
            t = (temb if temb is not None
                 else torch.zeros(b, RD, device=dev))
            h = self.semantic_resampler(te35_hidden, t)  # (B,64,1024)
            for i, attn in enumerate(self.v2_attentions):
                h = h + attn(
                    h * self.v2_query_norms[i].weight,
                    anchored * self.v2_semantic_norms[i].weight,
                )
            return h  # (B, 64, 1024) — DiT cond 시퀀스

    return ConnectorV2()


# 어댑터 파일(최상위 키) → 모듈 키 remap.
# 어댑터 파일 키: layer_mix_logits, query_norms.i, source_norms.i,
#                semantic_attentions.i.*
# DiT 내장 키: net.anima_v2_connector.quality_anchor.*
REMAP_ADAPTER_TO_MODULE = {
    "layer_mix_logits": "quality_anchor.layer_mix_logits",
    "query_norms": "quality_anchor.query_norms",
    "source_norms": "quality_anchor.source_norms",
    "semantic_attentions": "quality_anchor.semantic_attentions",
}


def remap_adapter_keys(state: dict) -> dict:
    """어댑터 파일의 최상위 키를 모듈 키로 변환."""
    out = {}
    for k, v in state.items():
        for src, dst in REMAP_ADAPTER_TO_MODULE.items():
            if k == src or k.startswith(src + "."):
                k = dst + k[len(src):]
                break
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 텍스트 인코더 로딩 — transformers는 함수 내 lazy import
# ---------------------------------------------------------------------------

def _has_fp8_tensors(path: str) -> bool:
    """safetensors 헤더에 FP8 텐서가 있는지 확인 (가중치 로드 없음)."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return any(str(f.get_slice(k).get().dtype)
                   in ("torch.float8_e4m3fn", "torch.float8_e5m2")
                   for k in f.keys())


def dequantize_fp8_state_dict(path: str, target_dtype: str = "bfloat16") -> dict:
    """FP8 양자화 safetensors를 읽어 target dtype으로 변환한 state_dict 반환.

    transformers가 FP8 가중치를 직접 처리하지 못할 때의 폴백.
    """
    import torch
    from safetensors.torch import load_file

    raw = load_file(path, device="cpu")
    dtype = getattr(torch, target_dtype)
    out = {}
    for k, t in raw.items():
        if str(t.dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
            t = t.to(dtype)
        out[k] = t
    return out


def load_qwen3_te(path: str, dtype: str = "bfloat16"):
    """Qwen3 0.6B TE 로드 (표준 Qwen3 레이아웃 — 덤프 실측).

    체크포인트는 `model.` prefix가 있는 단독 .safetensors 파일이므로
    config를 직접 만들고 state_dict를 읽어 로드한다.
    (from_pretrained는 디렉토리/config.json를 요구해서 사용 불가)
    """
    import torch
    import transformers
    from safetensors.torch import load_file

    # Qwen3-0.6B 실측 (q_proj [2048,1024]=16×128, v_proj [1024,1024]=8×128)
    config = transformers.Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
    )
    model = transformers.Qwen3Model(config)
    state = load_file(path, device="cpu")
    # `model.` prefix 제거 (Qwen3Model은 prefix 없음)
    state = {k[6:] if k.startswith("model.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(dtype=getattr(torch, dtype))
    model.eval()
    return model


def load_qwen35_te(path: str, dtype: str = "bfloat16",
                   _model_cls=None, _config_cls=None):
    """Qwen3.5 4B TE 로드 (FP8 처리 포함).

    1차: transformers Qwen3_5TextModel.from_pretrained 직접 시도.
    실패 시 폴백: Qwen3_5TextConfig를 실측값으로 직접 생성 후
    FP8 → bf16 dequantize한 state_dict를 로드.
    (Qwen3_5Config는 vision+text 합성 config라 TextModel에 사용 불가)

    _model_cls/_config_cls: 테스트용 의존성 주입.
    """
    import torch
    import transformers
    from safetensors.torch import load_file

    model_cls = _model_cls or transformers.Qwen3_5TextModel
    config_cls = _config_cls or transformers.Qwen3_5TextConfig
    tdtype = getattr(torch, dtype)
    try:
        model = model_cls.from_pretrained(
            path, torch_dtype=tdtype, trust_remote_code=False,
        )
        model.eval()
        return model, {"fp8_fallback": False}
    except Exception as e:  # noqa: BLE001 — FP8 등 비표준 형식이면 폴백
        first_error = e

    # 폴백: Qwen3.5-4B text_config 실측값 (HF Qwen/Qwen3.5-4B config.json)
    try:
        config = config_cls(
            vocab_size=248320, hidden_size=2560, intermediate_size=9216,
            num_hidden_layers=32, num_attention_heads=16,
            num_key_value_heads=4, head_dim=256,
            linear_conv_kernel_dim=4, linear_key_head_dim=128,
            linear_num_key_heads=16, linear_value_head_dim=128,
            linear_num_value_heads=32, full_attention_interval=4,
            tie_word_embeddings=True, max_position_embeddings=262144,
            rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
        )
    except Exception:  # noqa: BLE001 — 테스트용 FakeConfig 등
        try:
            config = config_cls.from_pretrained(path, trust_remote_code=False)
        except Exception:  # noqa: BLE001
            config = config_cls()
    model = model_cls(config)
    state = dequantize_fp8_state_dict(path, dtype)
    # `model.` prefix 보정 (덤프: prefix 없음, Qwen3_5TextModel은 prefix 있음)
    if not any(k.startswith("model.") for k in state):
        state = {"model." + k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(tdtype)
    model.eval()
    return model, {"fp8_fallback": True, "first_error": str(first_error),
                   "missing": missing, "unexpected": unexpected}


def load_tokenizer(tokenizer_dir: str):
    """토크나이저 로드. 0.6B용과 4B용은 별개 디렉토리 (공유하지 않음)."""
    import transformers

    if not os.path.isdir(tokenizer_dir):
        raise RuntimeError(
            f"토크나이저 디렉토리를 찾을 수 없어요: {tokenizer_dir}")
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=False)


# ---------------------------------------------------------------------------
# Anima38Pipeline
# ---------------------------------------------------------------------------

class Anima38Pipeline:
    """Anima 3.8b 전용 파이프라인.

    generate.py의 Pipeline과 같은 인터페이스(load/txt2img/output_dir)를
    제공하므로 기존 생성 경로를 그대로 재사용한다.
    img2img/inpaint/pose는 미지원 (명확한 에러).
    """

    def __init__(self, pack_dir: str, output_dir: str = "outputs",
                 device: str | None = None, torch_dtype: str | None = None,
                 vae_path: str | None = None):
        self.pack_dir = pack_dir
        self.output_dir = output_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.vae_path = vae_path
        self._dit = None
        self._te06 = self._te35 = None
        self._tok06 = self._tok35 = None
        self._vae = None
        self._config = AnimaDiTConfig()

    # -- 로드 ----------------------------------------------------------

    def _require_pack_files(self) -> dict:
        st = pack_status(self.pack_dir)
        # DiT 본체는 팩에 필수. TE/어댑터는 없으면
        # models/text_encoders/ 폴백(_fallback_te_path)으로 메운다.
        missing = []
        if not st["dit"]:
            missing.append("DiT 본체(*3.8*.safetensors)")
        if missing:
            raise RuntimeError(
                "Anima 3.8b 팩 파일이 부족해요: " + ", ".join(missing)
                + f" — 팩 디렉토리: {self.pack_dir}")
        return st

    def load(self) -> "Anima38Pipeline":
        """팩 전체를 로드한다. 여러 번 호출해도 한 번만 로드."""
        if self._dit is not None:
            return self
        import torch
        from atelier.engine.pipeline import _default_device, _default_dtype

        device = self.device or _default_device()  # GPU 없으면 RuntimeError→503
        dtype_name = self.torch_dtype or _default_dtype(device)
        dtype = getattr(torch, dtype_name)
        st = self._require_pack_files()

        dit_path = find_dit_file(self.pack_dir)
        # TE 2개: 팩 우선, 없으면 models/text_encoders/ 폴백
        te35_path = (os.path.join(self.pack_dir, "qwen35_4b.safetensors")
                     if st["qwen35_4b"]
                     else _fallback_te_path("qwen35_4b"))
        if not te35_path or not os.path.isfile(te35_path):
            raise RuntimeError(
                "qwen35_4b.safetensors를 찾을 수 없어요. "
                "팩 디렉토리나 models/text_encoders/에 넣어주세요.")
        # qwen_3_06b 파일명 대소문자 복원
        te06_name = next(
            (f for f in os.listdir(self.pack_dir)
             if f.lower().startswith("qwen_3_06b")
             and f.lower().endswith(".safetensors")),
            None,
        )
        te06_path = (os.path.join(self.pack_dir, te06_name)
                     if te06_name
                     else _fallback_te_path("qwen3_06b"))
        if not te06_path or not os.path.isfile(te06_path):
            raise RuntimeError(
                "qwen_3_06b_base.safetensors를 찾을 수 없어요. "
                "팩 디렉토리나 models/text_encoders/에 넣어주세요.")

        # 1) DiT
        dit = build_anima_dit(self._config)
        dit_state = load_safetensors_mapped(dit_path, prefix_strip=DIT_PREFIX)
        report = apply_state_dict(dit, dit_state, prefix=DIT_PREFIX)
        self._dit_missing = report["missing"]
        self._dit_unexpected = report["unexpected"]
        dit.to(device=device, dtype=dtype)
        dit.eval()
        self._dit = dit

        # 2) TE 2개 (no_grad, eval)
        self._te06 = load_qwen3_te(te06_path, dtype_name)
        self._te06.to(device=device, dtype=dtype)
        self._te35, self._te35_info = load_qwen35_te(te35_path, dtype_name)
        self._te35.to(device=device, dtype=dtype)

        # 3) 토크나이저 (별개 — 공유하지 않음)
        tok35, tok06 = select_tokenizer_dirs(st["tokenizer_dirs"])
        self._tok35 = load_tokenizer(os.path.join(self.pack_dir, tok35))
        self._tok06 = load_tokenizer(os.path.join(self.pack_dir, tok06))

        # 4) v2 커넥터 가중치: 어댑터 파일 우선, 없으면 DiT 내장 사용
        self._load_connector_weights(st, device, dtype)

        # 5) VAE (base와 공유 — 팩 내 vae/ 또는 명시 경로)
        self._vae = self._load_vae(device, dtype)

        self.device = device
        self.torch_dtype = dtype_name
        return self

    def _load_connector_weights(self, st: dict, device, dtype) -> None:
        import torch
        from safetensors.torch import load_file

        connector = self._dit.anima_v2_connector
        adapter_name = st["adapter"]
        apath = None
        if adapter_name:
            apath = next(
                (os.path.join(self.pack_dir, f)
                 for f in os.listdir(self.pack_dir)
                 if f.lower() == adapter_name),
                None,
            )
        if apath is None:
            # 팩에 어댑터가 없으면 models/text_encoders/ 폴백
            apath = _fallback_te_path("adapter")
        if apath:
            raw = load_file(apath, device="cpu")
            state = remap_adapter_keys(
                {k: v.to(dtype) if str(v.dtype).startswith("torch.float")
                 else v for k, v in raw.items()})
            rep = apply_state_dict(connector, state,
                                   prefix="adapter:")
            self._connector_report = rep
        else:
            # DiT 파일 내장 커넥터 사용 (이미 dit_state에 포함됨)
            self._connector_report = {"missing": [], "unexpected": [],
                                      "note": "DiT 내장 커넥터 사용"}

    def _load_vae(self, device, dtype):
        from diffusers import AutoencoderKL

        candidates = []
        if self.vae_path:
            candidates.append(self.vae_path)
        # 사용자가 고른 VAE (모델 탭/퀵 패널) — 팩 내장보다 우선
        try:
            from atelier.api.routes import models as models_module
            sel_vae = models_module.get_selected_vae_path()
        except Exception:  # noqa: BLE001 - 폴백은 조용히 포기
            sel_vae = None
        if sel_vae:
            candidates.insert(0, sel_vae)
        pack_vae = os.path.join(self.pack_dir, "vae")
        if os.path.isdir(pack_vae):
            candidates.append(pack_vae)
        # 팩 루트에 바로 있는 VAE 파일 (예: qwenImageVAE.safetensors)
        try:
            for f in os.listdir(self.pack_dir):
                if (f.lower().endswith(".safetensors")
                        and "vae" in f.lower()
                        and not f.startswith(".")):
                    candidates.append(os.path.join(self.pack_dir, f))
                    break
        except OSError:
            pass
        for cand in candidates:
            try:
                if os.path.isdir(cand):
                    vae = AutoencoderKL.from_pretrained(
                        cand, torch_dtype=dtype, trust_remote_code=False)
                else:
                    vae = AutoencoderKL.from_single_file(
                        cand, torch_dtype=dtype, trust_remote_code=False)
                vae.to(device=device, dtype=dtype)
                vae.eval()
                return vae
            except Exception:
                continue
        raise RuntimeError(
            "VAE를 찾을 수 없어요. base 체크포인트에서 추출한 VAE를 "
            f"팩 디렉토리의 vae/ 폴더에 넣어주세요: {self.pack_dir}/vae/ "
            "(또는 extract_vae_from_checkpoint으로 생성)")

    # -- 프롬프트 인코딩 -------------------------------------------------

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """듀얼 TE 인코딩 + v2 커넥터 → DiT cond.

        반환: (cond, neg_cond) — cond는 (B, T, cond_dim) 시퀀스 (차원 추정).
        """
        import torch

        def _encode(tok, te, text):
            ids = tok(text, return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
            ids = {k: v.to(self.device) for k, v in ids.items()}
            with torch.no_grad():
                out = te(**ids)
            hidden = (out.last_hidden_state
                      if hasattr(out, "last_hidden_state") else out[0])
            return hidden

        with torch.no_grad():
            h35 = _encode(self._tok35, self._te35, prompt)
            h06 = _encode(self._tok06, self._te06, prompt)
            # llm_adapter는 0.6B 경로에 적용 (위치 추정 — 미확인)
            for blk in self._dit.llm_adapter.blocks:
                h06 = blk(h06)
            cond = self._dit.anima_v2_connector(h35)
            neg_cond = None
            if negative_prompt:
                nh35 = _encode(self._tok35, self._te35, negative_prompt)
                neg_cond = self._dit.anima_v2_connector(nh35)
        return cond, neg_cond

    # -- 생성 ----------------------------------------------------------

    def _common_kwargs(self, params: dict):
        import torch
        from atelier.settings import defaults, validate

        errors = validate(params)
        if errors:
            raise ValueError(f"파라미터 오류: {errors}")
        p = defaults()
        p.update(params)
        seed = p["seed"]
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return {
            "prompt": p["prompt"],
            "negative_prompt": p["negative_prompt"],
            "height": p["height"],
            "width": p["width"],
            "num_inference_steps": p["steps"],
            "guidance_scale": p["cfg_scale"],
            "generator": generator,
            "num_images_per_prompt": p["batch_size"],
        }, seed

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        """텍스트 → 이미지. 저장된 파일 경로 리스트 반환."""
        import torch
        from atelier.engine.samplers import build_scheduler
        from atelier.settings import defaults as _defaults

        if self._dit is None:
            raise RuntimeError(
                "파이프라인이 로드되지 않았어요. load()를 먼저 호출하세요.")
        if loras:
            # 3.8b 커스텀 DiT에는 diffusers LoRA를 적용할 수 없음
            print("[atelier] Anima 3.8b에서는 LoRA가 지원되지 않아 건너뜁니다.")

        kwargs, seed = self._common_kwargs(params)
        sampler_name = params.get("sampler", _defaults()["sampler"])
        noise_schedule = params.get("scheduler", _defaults()["scheduler"])
        from atelier.engine.samplers import _NoiseScheduleWrapper
        scheduler = build_scheduler(sampler_name)
        if noise_schedule not in (None, "auto", "normal"):
            scheduler = _NoiseScheduleWrapper(scheduler, noise_schedule)
        scheduler.set_timesteps(kwargs["num_inference_steps"])

        cond, neg_cond = self.encode_prompt(kwargs["prompt"],
                                            kwargs["negative_prompt"])
        # CFG: cond/new_cond concat (diffusers 관례)
        if neg_cond is not None and kwargs["guidance_scale"] > 1.0:
            cond_in = torch.cat([neg_cond, cond], dim=0)
        else:
            cond_in = cond

        h, w = kwargs["height"], kwargs["width"]
        lat_h, lat_w = h // 8, w // 8  # SDXL VAE 압축률 가정 (미확인)
        latents = torch.randn(
            (kwargs["num_images_per_prompt"], self._config.in_channels,
             lat_h, lat_w),
            generator=kwargs["generator"], device=self.device,
            dtype=getattr(torch, self.torch_dtype),
        )
        batch = latents.shape[0]
        if cond_in.shape[0] == 1 and batch > 1:
            cond_in = cond_in.expand(batch, -1, -1)

        dit = self._dit
        for t in scheduler.timesteps:
            t_batch = t.expand(batch).to(self.device)
            with torch.no_grad():
                if neg_cond is not None and kwargs["guidance_scale"] > 1.0:
                    lat_in = torch.cat([latents, latents], dim=0)
                    h_pred, _ = dit(lat_in, t_batch, cond_in)
                    # h_pred는 토큰 시퀀스 — unpatchify는 VAE 디코드 직전에
                    noise_pred = self._unpatchify(h_pred, lat_h, lat_w)
                    n_neg, n_pos = noise_pred.chunk(2, dim=0)
                    noise_pred = (n_neg + kwargs["guidance_scale"]
                                  * (n_pos - n_neg))
                else:
                    h_pred, _ = dit(latents, t_batch, cond_in)
                    noise_pred = self._unpatchify(h_pred, lat_h, lat_w)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = self._decode(latents)
        from atelier.pngmeta import build_meta_from_kwargs, build_pnginfo
        meta = build_meta_from_kwargs(kwargs, seed, sampler_name)
        return self._save(images, seed, "anima38-txt2img", meta)

    def _unpatchify(self, tokens, lat_h: int, lat_w: int):
        """토큰 시퀀스 → latent 이미지.

        실측: final linear 출력 64-dim = 16ch × 2×2 패치.
        """
        p = self._config.patch_size
        b, t, _ = tokens.shape
        ch = 16
        x = tokens.view(b, lat_h // p, lat_w // p, p, p, ch)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, ch, lat_h, lat_w)
        return x

    def _decode(self, latents):
        import torch
        from PIL import Image
        import numpy as np

        vae = self._vae
        scale = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)
        with torch.no_grad():
            imgs = vae.decode(latents / scale).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = (imgs.cpu().permute(0, 2, 3, 1).float().numpy() * 255
                ).round().astype(np.uint8)
        return [Image.fromarray(imgs[i]) for i in range(imgs.shape[0])]

    def _save(self, images, seed: int, tag: str,
              meta: dict | None = None) -> list[str]:
        import uuid
        from datetime import datetime

        from atelier.pngmeta import build_pnginfo

        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        uniq = uuid.uuid4().hex[:6]
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(
                self.output_dir, f"{stamp}-{tag}-s{seed}-{i}-{uniq}.png")
            if meta:
                img.save(path, pnginfo=build_pnginfo(meta))
            else:
                img.save(path)
            paths.append(path)
        return paths

    # -- 미지원 모드 -----------------------------------------------------

    def img2img(self, *a, **k):
        raise RuntimeError("Anima 3.8b에서는 img2img를 아직 지원하지 않아요.")

    def inpaint(self, *a, **k):
        raise RuntimeError("Anima 3.8b에서는 인페인트를 아직 지원하지 않아요.")

    def pose(self, *a, **k):
        raise RuntimeError("Anima 3.8b에서는 포즈 지정을 아직 지원하지 않아요.")


class Anima2BPipeline:
    """Anima 2b (base, 28블록) 전용 파이프라인.

    단일 .safetensors 파일에서 DiT를 로드.
    - DiT 28블록 (파일에서 감지)
    - Qwen3 0.6B TE → llm_adapter → cond (v2 커넥터 없음)
    - VAE (base와 공유)
    """

    def __init__(self, model_path: str, output_dir: str = "outputs",
                 device: str | None = None, torch_dtype: str | None = None,
                 vae_path: str | None = None, te_path: str | None = None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.vae_path = vae_path
        self.te_path = te_path
        self._dit = None
        self._te06 = None
        self._tok06 = None
        self._vae = None
        self._config = AnimaDiTConfig()

    def load(self) -> "Anima2BPipeline":
        """2b 전체를 로드한다."""
        if self._dit is not None:
            return self
        import torch
        from atelier.engine.pipeline import _default_device, _default_dtype
        from atelier.api.routes import models as models_module

        device = self.device or _default_device()
        dtype_name = self.torch_dtype or _default_dtype(device)
        dtype = getattr(torch, dtype_name)

        # DiT 레이어 수 감지
        num_layers = self._detect_layers(self.model_path)
        self._config.num_layers = num_layers

        # 1) DiT
        dit = build_anima_dit(self._config)
        dit_state = load_safetensors_mapped(self.model_path,
                                            prefix_strip=DIT_PREFIX)
        report = apply_state_dict(dit, dit_state, prefix=DIT_PREFIX)
        # v2 커넥터가 없으면 missing에 많이 나옴 — 2b는 정상
        dit.to(device=device, dtype=dtype)
        dit.eval()
        self._dit = dit

        # 2) Qwen3 0.6B TE
        te_path = self.te_path or self._find_te06()
        if not te_path:
            raise RuntimeError(
                "Qwen3 0.6B TE를 찾을 수 없어요. "
                "models/text_encoder/에 넣어주세요.")
        self._te06 = load_qwen3_te(te_path, dtype_name)
        self._te06.to(device=device, dtype=dtype)

        # 3) 토크나이저 (Qwen3-0.6B)
        self._tok06 = self._find_tokenizer()

        # 4) VAE
        vae_path = self.vae_path or models_module.get_selected_vae_path()
        if not vae_path:
            vae_path = self._find_vae()
        if not vae_path:
            raise RuntimeError("VAE를 찾을 수 없어요.")
        self._vae = self._load_vae(vae_path, device, dtype)

        self.device = device
        self.torch_dtype = dtype_name
        return self

    def _detect_layers(self, path: str) -> int:
        """safetensors 헤더에서 블록 수 감지."""
        import re
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            blocks = set()
            for k in f.keys():
                m = re.match(r"net\.blocks\.(\d+)\.", k)
                if m:
                    blocks.add(int(m.group(1)))
        return len(blocks) or 28

    def _find_te06(self) -> str | None:
        """Qwen3 0.6B TE 찾기."""
        import glob
        from atelier.api.routes import models as models_module
        # 선택된 TE 우선
        sel = models_module.get_selected_text_encoder()
        if sel:
            p = models_module.get_selected_text_encoder_path()
            if p and os.path.isfile(p):
                return p
        # models/text_encoders/ 또는 text_encoder/ 에서 찾기
        models_dir = models_module.models_dir()
        for d in ("text_encoders", "text_encoder"):
            for pat in (f"{d}/qwen_3_06b*.safetensors",
                        f"{d}/*06b*.safetensors"):
                for p in glob.glob(os.path.join(models_dir, pat)):
                    return p
        return None

    def _find_tokenizer(self):
        """Qwen3-0.6B 토크나이저 찾기."""
        from atelier.api.routes import models as models_module
        models_dir = models_module.models_dir()
        # 흔한 위치들
        for p in [
            os.path.join(models_dir, "qwen3_06b_tokenizer"),
            os.path.join(models_dir, "text_encoder", "qwen3_06b_tokenizer"),
            os.path.join(models_dir, "anima38pack", "qwen3_06b_tokenizer"),
        ]:
            if os.path.isdir(p):
                return load_tokenizer(p)
        # HF에서 다운로드
        import transformers
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def _find_vae(self) -> str | None:
        import glob
        from atelier.api.routes import models as models_module
        models_dir = models_module.models_dir()
        for p in glob.glob(os.path.join(models_dir, "vae", "*.safetensors")):
            return p
        return None

    def _load_vae(self, vae_path: str, device, dtype):
        import torch
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_single_file(
            vae_path, torch_dtype=dtype, trust_remote_code=False)
        vae.to(device=device, dtype=dtype)
        vae.eval()
        return vae

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """Qwen3 0.6B → llm_adapter → cond (1024-dim)."""
        import torch

        def _encode(text):
            ids = self._tok06(text, return_tensors="pt", padding=True,
                              truncation=True, max_length=512)
            ids = {k: v.to(self.device) for k, v in ids.items()}
            with torch.no_grad():
                out = self._te06(**ids)
            hidden = (out.last_hidden_state
                      if hasattr(out, "last_hidden_state") else out[0])
            # llm_adapter 블록으로 정제
            for blk in self._dit.llm_adapter.blocks:
                hidden = blk(hidden)
            return hidden

        with torch.no_grad():
            cond = _encode(prompt)
            neg_cond = _encode(negative_prompt) if negative_prompt else None
        return cond, neg_cond

    # -- 생성 (Anima38Pipeline.txt2img와 동일, 2b용) --

    def _common_kwargs(self, params: dict):
        # Anima38Pipeline의 것을 재사용
        tmp = Anima38Pipeline.__new__(Anima38Pipeline)
        tmp._config = self._config
        return Anima38Pipeline._common_kwargs(tmp, params)

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        import torch
        from atelier.engine.samplers import build_scheduler
        from atelier.settings import defaults as _defaults

        if self._dit is None:
            raise RuntimeError("파이프라인이 로드되지 않았어요.")
        if loras:
            print("[atelier] Anima 2b에서는 LoRA가 지원되지 않아 건너뜁니다.")

        kwargs, seed = self._common_kwargs(params)
        sampler_name = params.get("sampler", _defaults()["sampler"])
        noise_schedule = params.get("scheduler", _defaults()["scheduler"])
        from atelier.engine.samplers import _NoiseScheduleWrapper
        scheduler = build_scheduler(sampler_name)
        if noise_schedule not in (None, "auto", "normal"):
            scheduler = _NoiseScheduleWrapper(scheduler, noise_schedule)
        scheduler.set_timesteps(kwargs["num_inference_steps"])

        cond, neg_cond = self.encode_prompt(kwargs["prompt"],
                                            kwargs["negative_prompt"])
        if neg_cond is not None and kwargs["guidance_scale"] > 1.0:
            cond_in = torch.cat([neg_cond, cond], dim=0)
        else:
            cond_in = cond

        h, w = kwargs["height"], kwargs["width"]
        lat_h, lat_w = h // 8, w // 8
        latents = torch.randn(
            (kwargs["num_images_per_prompt"], self._config.in_channels,
             lat_h, lat_w),
            generator=kwargs["generator"], device=self.device,
            dtype=getattr(torch, self.torch_dtype),
        )
        batch = latents.shape[0]
        if cond_in.shape[0] == 1 and batch > 1:
            cond_in = cond_in.expand(batch, -1, -1)

        dit = self._dit
        for t in scheduler.timesteps:
            t_batch = t.expand(batch).to(self.device)
            with torch.no_grad():
                if neg_cond is not None and kwargs["guidance_scale"] > 1.0:
                    lat_in = torch.cat([latents, latents], dim=0)
                    h_pred, _ = dit(lat_in, t_batch, cond_in)
                    noise_pred = self._unpatchify(h_pred, lat_h, lat_w)
                    n_neg, n_pos = noise_pred.chunk(2, dim=0)
                    noise_pred = (n_neg + kwargs["guidance_scale"]
                                  * (n_pos - n_neg))
                else:
                    h_pred, _ = dit(latents, t_batch, cond_in)
                    noise_pred = self._unpatchify(h_pred, lat_h, lat_w)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = self._decode(latents)
        from atelier.pngmeta import build_meta_from_kwargs, build_pnginfo
        meta = build_meta_from_kwargs(kwargs, seed, sampler_name)
        return self._save(images, seed, "anima2b-txt2img", meta)

    def _unpatchify(self, tokens, lat_h: int, lat_w: int):
        p = self._config.patch_size
        b, t, _ = tokens.shape
        ch = 16
        x = tokens.view(b, lat_h // p, lat_w // p, p, p, ch)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, ch, lat_h, lat_w)
        return x

    def _decode(self, latents):
        import torch
        from PIL import Image
        import numpy as np
        vae = self._vae
        scale = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)
        with torch.no_grad():
            imgs = vae.decode(latents / scale).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = (imgs.cpu().permute(0, 2, 3, 1).float().numpy() * 255
                ).round().astype(np.uint8)
        return [Image.fromarray(imgs[i]) for i in range(imgs.shape[0])]

    def _save(self, images, seed: int, tag: str, meta) -> list[str]:
        import uuid
        from datetime import datetime
        from atelier.pngmeta import build_pnginfo
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        uniq = uuid.uuid4().hex[:6]
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(
                self.output_dir, f"{stamp}-{tag}-s{seed}-{i}-{uniq}.png")
            info = build_pnginfo(meta) if meta else None
            img.save(path, pnginfo=info)
            paths.append(path)
        return paths

    def img2img(self, *a, **k):
        raise RuntimeError("Anima 2b에서는 img2img가 지원되지 않아요.")

    def inpaint(self, *a, **k):
        raise RuntimeError("Anima 2b에서는 inpaint가 지원되지 않아요.")

    def pose(self, *a, **k):
        raise RuntimeError("Anima 2b에서는 pose가 지원되지 않아요.")


def extract_vae_from_checkpoint(ckpt_path: str, out_dir: str) -> str:
    """base 체크포인트에서 VAE만 추출해 out_dir에 저장.

    3.8b 팩의 vae/ 폴더를 준비할 때 사용한다. diffusers는 함수 내 lazy import.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_single_file(ckpt_path, trust_remote_code=False)
    os.makedirs(out_dir, exist_ok=True)
    vae.save_pretrained(out_dir)
    return out_dir
