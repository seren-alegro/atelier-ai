"""Anima 3.8b 커스텀 파이프라인 (DiT 52블록 + 듀얼 Qwen TE + v2 커넥터).

diffusers AutoPipeline으로 로드할 수 없는 커스텀 아키텍처이므로
torch 모듈을 직접 정의하고 safetensors 가중치를 키 매핑으로 로드한다.

클린룸: 아래 모듈은 2026-10-15 덤프(anima38_dump.json)의 텐서 키 이름·
shape에서 확인된 구조만으로 직접 작성했다. 외부 코드를 복사하지 않았다.

torch / transformers / safetensors / diffusers는 함수 안에서 lazy import
하므로 이 모듈 자체는 GPU·torch 없이 import 가능하다 (pipeline.py 패턴).
torch가 필요한 nn.Module 정의는 build_* 팩토리 함수 안에 둔다
(upscale.py의 _build_rrdbnet 패턴).

== 덤프에서 확정된 사항 ==
- DiT (Anima-3.8B-v1.1.safetensors): `net.blocks` 0..51 (52블록),
  hidden 2048, MLP 8192 (fc1 [8192,2048] / fc2 [2048,8192] 각 52개),
  블록당 [256,2048] 3개 + [6144,256] 3개 (rank-256 factorized QKV),
  블록당 [128] 4개, `net.x_embedder.proj`, `net.t_embedder`,
  `net.final_layer.adaln_modulation` (2개), `net.llm_adapter.blocks` 0..5.
- v2 커넥터 (DiT 파일 내장 + 독립 어댑터 파일, 동일 구조):
  quality_anchor {query_norms×6, source_norms×6, semantic_attentions×6,
  layer_mix_logits [6,4]}, semantic_resampler {blocks×6, time_mlp},
  v2_attentions×6, v2_query_norms×6, v2_semantic_norms×6.
  semantic_attentions 키 실측:
  `{...}.k_norm.weight` [64], `{...}.k_proj.weight` [1024,2560],
  `{...}.o_proj.weight` [1024,1024].
- Qwen3.5 4B TE: 키 prefix 없음 (`embed_tokens.weight`, `layers.N.*`),
  32 레이어, hidden 2560, 245개 텐서가 FP8_E4M3.
- Qwen3 0.6B TE: 표준 Qwen3 레이아웃 (`model.layers` 28개, hidden 1024).
- 토크나이저 분리: 0.6B용 Qwen/Qwen3-0.6B, 4B용 qwen35_tokenizer/ (공유 안 함).
- VAE는 base와 공유.

== 미확인 사항 (실측 필요, RunPod 24GB+) ==
- DiT 블록당 [2048,2048] 351개의 귀속 (cross-attn q/o·adaLN 등으로 추정).
- o_proj shape ([2048,6144]로 가정 — 6144=48heads×128).
- num_heads=48, head_dim=128 (6144/128, [128] norm에서 추정).
- adaLN 주입 방식 (adaLN-single 가정: 블록당 학습 테이블 + temb).
- x_embedder patch_size/in_channels, t_embedder 내부, final_layer 구조.
- llm_adapter / semantic_resampler 블록 내부 (shape 기반 추정 구현).
- v2 커넥터의 정확한 데이터플로우 (모듈 구조는 키 이름대로, 연산 순서는 추정).
- cross-attention cond 차원 (1024로 가정).
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
    """DiT 설정. 확정값은 덤프 실측, 나머지는 추정(미확인 표기)."""

    num_layers: int = 52        # 확정: net.blocks 0..51
    hidden_size: int = 2048     # 확정: 최다 shape
    mlp_dim: int = 8192        # 확정: [8192,2048]/[2048,8192] 각 52개
    qkv_rank: int = 256         # 확정: [256,2048] 3개/블록
    qkv_up_dim: int = 6144      # 확정: [6144,256] 3개/블록
    num_heads: int = 48         # 추정(미확인): 6144 = 48 × 128
    head_dim: int = 128         # 추정(미확인): [128] norm 4개/블록
    cond_dim: int = 1024        # 추정(미확인): 어댑터 [1024,*] 계열
    patch_size: int = 2         # 미확인
    in_channels: int = 4        # 미확인 (SDXL VAE latent 가정)
    # v2 커넥터
    connector_layers: int = 6   # 확정: query_norms/source_norms 등 각 6개
    connector_dim: int = 1024   # 확정: [1024,1024]/[1024,2560] 계열
    resampler_dim: int = 2048   # 추정: [11264,2048]/[2048,5632] 계열
    resampler_mlp_dim: int = 5632  # 추정: 11264 = 2 × 5632
    te35_hidden: int = 2560     # 확정: Qwen3.5 4B hidden
    te06_hidden: int = 1024     # 확정: Qwen3 0.6B hidden


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
    "k_norm.weight",   # [64]
    "k_proj.weight",   # [1024,2560]
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

    구조 (확정/추정 구분은 모듈 docstring 참조):
    - x_embedder: patchify conv (미확인: patch_size/in_channels)
    - t_embedder: timestep MLP (미확인: 내부 구조)
    - 52× AnimaDiTBlock: factorized self-attn (확정) + cross-attn (추정)
      + MLP (확정) + adaLN-single (추정)
    - final_layer: adaLN + unpatchify (미확인)
    - llm_adapter: 6× residual MLP 블록 (shape 기반 추정)
    - anima_v2_connector: 아래 build_connector_v2 참조
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    cfg = config or AnimaDiTConfig()
    H, R, U = cfg.hidden_size, cfg.qkv_rank, cfg.qkv_up_dim
    NH, HD = cfg.num_heads, cfg.head_dim
    assert U == NH * HD, "qkv_up_dim은 num_heads*head_dim이어야 함"

    class _RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), self.weight)

    class _WeightParam(nn.Module):
        """단일 weight 파라미터.

        덤프 키가 `query_norms.0.weight`처럼 `.weight`으로 끝나므로
        ParameterList 대신 이 모듈을 써서 키 이름을 정확히 맞춘다.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self):
            return self.weight

    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    class FactorizedSelfAttention(nn.Module):
        """rank-256 인수분해 self-attention (덤프 실측 shape).

        q/k/v 각각: down Linear(H→R) [R,H] + up Linear(R→U) [U,R].
        QK-norm은 head_dim=128 기준 per-head RMSNorm으로 추정 (미확인).
        o_proj [H,U] shape은 top-20 밖에 있어 미확인이나 6144 정합성으로 가정.
        """

        def __init__(self):
            super().__init__()
            self.q_down = nn.Linear(H, R)
            self.k_down = nn.Linear(H, R)
            self.v_down = nn.Linear(H, R)
            self.q_up = nn.Linear(R, U)
            self.k_up = nn.Linear(R, U)
            self.v_up = nn.Linear(R, U)
            self.q_norm = _RMSNorm(HD)  # [128] — 미확인(축 추정)
            self.k_norm = _RMSNorm(HD)  # [128] — 미확인(축 추정)
            self.o_proj = nn.Linear(U, H)  # [H,U] — 미확인(shape 가정)

        def forward(self, x):
            b, t, _ = x.shape
            q = self.q_up(self.q_down(x)).view(b, t, NH, HD)
            k = self.k_up(self.k_down(x)).view(b, t, NH, HD)
            v = self.v_up(self.v_down(x)).view(b, t, NH, HD)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(b, t, U)
            return self.o_proj(out)

    class CrossAttention(nn.Module):
        """cond(1024-dim, 추정)에 대한 cross-attention.

        k/v [H,cond] shape은 덤프에서 블록당 2개로 확정.
        q/o [H,H]는 미확인(가정).
        """

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, H)  # 미확인
            self.k_proj = nn.Linear(cfg.cond_dim, H)  # 확정 shape
            self.v_proj = nn.Linear(cfg.cond_dim, H)  # 확정 shape
            self.o_proj = nn.Linear(H, H)  # 미확인

        def forward(self, x, cond):
            b, t, _ = x.shape
            ch = max(1, H // HD)  # cross-attention head 수 (미확인)
            hd = H // ch
            q = self.q_proj(x).view(b, t, ch, hd).transpose(1, 2)
            k = self.k_proj(cond).view(b, -1, ch, hd).transpose(1, 2)
            v = self.v_proj(cond).view(b, -1, ch, hd).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(b, t, H)
            return self.o_proj(out)

    class AnimaDiTBlock(nn.Module):
        """DiT 블록 1개.

        adaLN 주입은 adaLN-single 방식(학습 테이블 + temb)으로 추정 —
        블록당 대형 modulation 선형층이 덤프에 없기 때문 (미확인).
        """

        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(H, elementwise_affine=False)
            self.attn = FactorizedSelfAttention()
            self.norm2 = nn.LayerNorm(H, elementwise_affine=False)
            self.cross_attn = CrossAttention()
            self.norm3 = nn.LayerNorm(H, elementwise_affine=False)
            self.mlp_fc1 = nn.Linear(H, cfg.mlp_dim)  # 확정 [8192,2048]
            self.mlp_fc2 = nn.Linear(cfg.mlp_dim, H)  # 확정 [2048,8192]
            # adaLN-single 테이블: shift/scale/gate × (attn, cross, mlp)
            # [6,H] — top-20 밖이라 미확인이나 부재와 정합
            self.scale_shift_table = nn.Parameter(torch.zeros(6, H))

        def forward(self, x, temb, cond):
            s = self.scale_shift_table + temb.unsqueeze(1)  # (B, 6, H)
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = (
                c.squeeze(1) for c in s.chunk(6, dim=1)
            )
            x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
            if cond is not None:
                # cross는 같은 gate를 재사용 (미확인)
                x = x + gate_msa * self.cross_attn(
                    _modulate(self.norm2(x), shift_msa, scale_msa), cond)
            h = _modulate(self.norm3(x), shift_mlp, scale_mlp)
            h = self.mlp_fc2(F.gelu(self.mlp_fc1(h), approximate="tanh"))
            return x + gate_mlp * h

    class LLMAdapterBlock(nn.Module):
        """llm_adapter 블록 (shape 기반 추정 — 미확인).

        덤프: 블록당 [4096,1024] + [1024,4096] + [4096] 각 1개.
        """

        def __init__(self):
            super().__init__()
            self.norm = _RMSNorm(4096)  # [4096] — 용도 추정
            self.fc1 = nn.Linear(1024, 4096)
            self.fc2 = nn.Linear(4096, 1024)

        def forward(self, x):
            return x + self.fc2(F.gelu(self.fc1(self.norm(x))))

    class AnimaDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            # 미확인: in_channels/patch_size (SDXL VAE latent 가정)
            self.x_embedder = nn.Conv2d(cfg.in_channels, H,
                                        kernel_size=cfg.patch_size,
                                        stride=cfg.patch_size)
            # t_embedder 내부 구조 미확인 — 2층 MLP로 가정
            self.t_embedder = nn.Sequential(
                nn.Linear(256, H), nn.SiLU(), nn.Linear(H, H),
            )
            self.blocks = nn.ModuleList([AnimaDiTBlock()
                                         for _ in range(cfg.num_layers)])
            self.llm_adapter = nn.ModuleList([LLMAdapterBlock()
                                              for _ in range(6)])
            self.anima_v2_connector = build_connector_v2(cfg)
            # final_layer: adaln_modulation 2개(덤프 실측, 용도 미확인) + 투영
            self.final_layer = nn.ModuleDict({
                "adaln_modulation": nn.ModuleList(
                    [nn.Linear(H, H), nn.Linear(H, H)]),
                "proj": nn.Linear(H, cfg.patch_size * cfg.patch_size
                                  * cfg.in_channels),
            })

        def timestep_embedding(self, timesteps, dim=256, max_period=10000):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(half, device=timesteps.device) / half)
            args = timesteps[:, None].float() * freqs[None]
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        def forward(self, x, timesteps, cond=None):
            # x: (B, C, h, w) latent
            temb = self.t_embedder(self.timestep_embedding(timesteps))
            h = self.x_embedder(x)  # (B, H, gh, gw)
            b, _, gh, gw = h.shape
            h = h.flatten(2).transpose(1, 2)  # (B, T, H)
            for blk in self.blocks:
                h = blk(h, temb, cond)
            return h, (b, gh, gw)

    return AnimaDiT()


# ---------------------------------------------------------------------------
# v2 커넥터 — torch 필요이므로 팩토리 안에서 정의
# ---------------------------------------------------------------------------

def build_connector_v2(config: AnimaDiTConfig | None = None):
    """anima_qwen35_quality_anchored_semantic_connector_v2.

    덤프 실측 키 이름 그대로 사용:
    - quality_anchor.query_norms.{i}.weight (6)
    - quality_anchor.source_norms.{i}.weight (6)
    - quality_anchor.semantic_attentions.{i}.{k_norm,k_proj,o_proj}.weight (6)
    - quality_anchor.layer_mix_logits [6,4]
    - semantic_resampler.blocks.{i} (6), semantic_resampler.time_mlp
    - v2_attentions.{i} (6), v2_query_norms.{i} (6),
      v2_semantic_norms.{i} (6)

    연산 순서는 추정(미확인). 모듈 계층·파라미터 이름·shape은 덤프와 일치.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = config or AnimaDiTConfig()
    D, T35 = cfg.connector_dim, cfg.te35_hidden
    L = cfg.connector_layers

    class _WeightParam(nn.Module):
        """단일 weight 파라미터 (덤프 키 `*.weight`와 정확히 매칭)."""

        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

    class SemanticAttention(nn.Module):
        """quality_anchor.semantic_attentions[i].

        확정 키: k_norm.weight [64], k_proj.weight [1024,2560],
        o_proj.weight [1024,1024].
        """

        def __init__(self):
            super().__init__()
            self.k_norm = _WeightParam(64)  # [64] — 축 미확인
            self.k_proj = nn.Linear(T35, D)  # [1024,2560] 확정
            self.o_proj = nn.Linear(D, D)  # [1024,1024] 확정

        def forward(self, queries, te_hidden):
            # queries: (B, Q, D), te_hidden: (B, T, 2560)
            k = self.k_proj(te_hidden)
            # k_norm [64]: head-dim 기준 추정 — 여기서는 단순 스케일로 적용
            # (정확한 축은 실측 필요)
            k = k * (self.k_norm.weight.mean())
            attn = torch.softmax(
                queries @ k.transpose(-2, -1) / (D ** 0.5), dim=-1)
            return self.o_proj(attn @ te_hidden[:, :, :D]
                               if te_hidden.shape[-1] == D else attn @ k)

    class QualityAnchor(nn.Module):
        def __init__(self):
            super().__init__()
            # 덤프 키: query_norms.{i}.weight (ModuleList+_WeightParam으로 맞춤)
            self.query_norms = nn.ModuleList(
                [_WeightParam(64) for _ in range(L)])
            self.source_norms = nn.ModuleList(
                [_WeightParam(64) for _ in range(L)])
            self.semantic_attentions = nn.ModuleList(
                [SemanticAttention() for _ in range(L)])
            self.layer_mix_logits = nn.Parameter(torch.zeros(L, 4))

        def forward(self, queries, te_hidden):
            outs = []
            for i in range(L):
                sa = self.semantic_attentions[i]
                q = queries * self.query_norms[i].weight.mean()
                outs.append(sa(q, te_hidden))
            mix = torch.softmax(self.layer_mix_logits, dim=-1)
            stacked = torch.stack(outs, dim=0)  # (L, B, Q, D)
            # [6,4] 가중치로 레이어 혼합 (정확한 의미는 실측 필요)
            w = mix.mean(dim=-1)  # (L,)
            return (stacked * w.view(L, 1, 1, 1)).sum(dim=0)

    class ResamplerBlock(nn.Module):
        """semantic_resampler.blocks[i] (shape 기반 추정 — 미확인).

        덤프: [11264,2048]=2048→2×5632 (gated MLP),
        [2048,5632], [12288]+[12288,2048] (adaLN-Zero modulation).
        """

        def __init__(self):
            super().__init__()
            rd, rm = cfg.resampler_dim, cfg.resampler_mlp_dim
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(rd, 6 * rd))
            self.gate_proj = nn.Linear(rd, 2 * rm)
            self.down_proj = nn.Linear(rm, rd)

        def forward(self, x, temb):
            # temb: (B, resampler_dim) — time_mlp 출력으로 가정
            mod = self.modulation(temb).view(
                x.shape[0], 6, -1)
            shift = mod[:, 0]
            # gated MLP (추정)
            g = self.gate_proj(x)
            a, b = g.chunk(2, dim=-1)
            h = self.down_proj(a * F.silu(b))
            return x + h * torch.sigmoid(shift).unsqueeze(1)

    class V2Attention(nn.Module):
        """v2_attentions[i] (구조 추정).

        덤프에서 v2_query_norms / v2_semantic_norms는 별도 indexed 그룹이므로
        norm은 어텐션 밖에 둔다.
        """

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(D, D)
            self.k_proj = nn.Linear(D, D)
            self.v_proj = nn.Linear(D, D)
            self.o_proj = nn.Linear(D, D)

        def forward(self, queries, semantics, q_scale=1.0, s_scale=1.0):
            q = self.q_proj(queries) * q_scale
            k = self.k_proj(semantics) * s_scale
            v = self.v_proj(semantics)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (D ** 0.5), -1)
            return self.o_proj(attn @ v)

    class SemanticResampler(nn.Module):
        """semantic_resampler (덤프 키 이름 그대로).

        time_mlp: Sequential(Linear, SiLU, Linear) → 가중치 2개
        (time_mlp.0.weight / time_mlp.2.weight, 덤프 count=2와 일치).
        """

        def __init__(self):
            super().__init__()
            rd = cfg.resampler_dim
            self.blocks = nn.ModuleList([ResamplerBlock() for _ in range(L)])
            self.time_mlp = nn.Sequential(
                nn.Linear(rd, rd), nn.SiLU(), nn.Linear(rd, rd))

    class ConnectorV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.quality_anchor = QualityAnchor()
            self.semantic_resampler = SemanticResampler()
            self.v2_attentions = nn.ModuleList(
                [V2Attention() for _ in range(L)])
            # 덤프 키: v2_query_norms.{i}.weight / v2_semantic_norms.{i}.weight
            self.v2_query_norms = nn.ModuleList(
                [_WeightParam(64) for _ in range(L)])
            self.v2_semantic_norms = nn.ModuleList(
                [_WeightParam(64) for _ in range(L)])
            # 학습 쿼리 (개수 미확인 — 32로 가정)
            self.latent_queries = nn.Parameter(torch.randn(32, D))
            # 차원 연결 투영 (미확인 — 덤프에 대응 키 미확인)
            self.anchor_to_resampler = nn.Linear(D, cfg.resampler_dim)
            self.resampler_to_cond = nn.Linear(cfg.resampler_dim, D)

        def forward(self, te35_hidden, temb=None):
            b = te35_hidden.shape[0]
            queries = self.latent_queries.unsqueeze(0).expand(b, -1, -1)
            anchored = self.quality_anchor(queries, te35_hidden)  # (B,Q,1024)
            # resampler 차원(2048, 추정)으로 투영 — 미확인
            h = self.anchor_to_resampler(anchored)
            t = self.semantic_resampler.time_mlp(
                temb if temb is not None
                else torch.zeros(b, cfg.resampler_dim, device=h.device))
            for blk in self.semantic_resampler.blocks:
                h = blk(h, t)
            # DiT cond 차원(1024, 추정)으로 복귀 — 미확인
            h = self.resampler_to_cond(h)
            for i, attn in enumerate(self.v2_attentions):
                h = h + attn(
                    h, anchored,
                    q_scale=self.v2_query_norms[i].weight.mean(),
                    s_scale=self.v2_semantic_norms[i].weight.mean(),
                )
            return h  # (B, 32, 1024) — DiT cond 시퀀스 (차원 추정)

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

    transformers Qwen3Model로 직접 로드한다.
    NOTE: `import transformers` + 속성 접근을 쓴다.
    `from transformers import X` 형태는 transformers의 lazy 모듈에서
    monkeypatch가 안정적으로 동작하지 않는 경우가 있어 테스트를 위해
    이 형태를 사용한다.
    """
    import torch
    import transformers

    model = transformers.Qwen3Model.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), trust_remote_code=False,
    )
    model.eval()
    return model


def load_qwen35_te(path: str, dtype: str = "bfloat16",
                   _model_cls=None, _config_cls=None):
    """Qwen3.5 4B TE 로드 (FP8 처리 포함).

    1차: transformers Qwen3_5TextModel.from_pretrained 직접 시도.
    실패 시 폴백: FP8 → bf16 dequantize 후 빈 모델에 load_state_dict.
    덤프 실측: 키에 `model.` prefix가 없으므로 필요시 추가한다.

    _model_cls/_config_cls: 테스트용 의존성 주입 (기본값은 transformers).
    """
    import torch
    import transformers

    model_cls = _model_cls or transformers.Qwen3_5TextModel
    config_cls = _config_cls or transformers.Qwen3_5Config
    tdtype = getattr(torch, dtype)
    try:
        model = model_cls.from_pretrained(
            path, torch_dtype=tdtype, trust_remote_code=False,
        )
        model.eval()
        return model, {"fp8_fallback": False}
    except Exception as e:  # noqa: BLE001 — FP8 등 비표준 형식이면 폴백
        first_error = e

    # 폴백: config는 파일에서 읽고(없으면 기본), 가중치는 dequantize
    try:
        config = config_cls.from_pretrained(path, trust_remote_code=False)
    except Exception:  # noqa: BLE001
        config = config_cls()
    model = model_cls(config)
    state = dequantize_fp8_state_dict(path, dtype)
    # `model.` prefix 보정 (덤프: prefix 없음)
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
            for blk in self._dit.llm_adapter:
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
        """토큰 시퀀스 → latent 이미지 (x_embedder 역연산, 구조 추정)."""
        import torch
        cfg = self._config
        p = cfg.patch_size
        b, t, _ = tokens.shape
        ch = cfg.in_channels
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


def extract_vae_from_checkpoint(ckpt_path: str, out_dir: str) -> str:
    """base 체크포인트에서 VAE만 추출해 out_dir에 저장.

    3.8b 팩의 vae/ 폴더를 준비할 때 사용한다. diffusers는 함수 내 lazy import.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_single_file(ckpt_path, trust_remote_code=False)
    os.makedirs(out_dir, exist_ok=True)
    vae.save_pretrained(out_dir)
    return out_dir
