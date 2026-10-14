"""LoRA 학습 엔진 (diffusers + peft 직접 구현).

- SDXL(Anima 등) 체크포인트 대상. UNet LoRA가 기본, 텍스트 인코더 LoRA는 옵션.
- 데이터셋: 이미지 + 동명 .txt 캡션 (kohya식). 트리거 워드는 캡션 맨 앞에 자동 삽입.
- torch / diffusers / peft는 함수 안에서 lazy import하므로,
  이 모듈 자체는 GPU·torch 없이 import 가능하다.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable

from atelier.engine.pipeline import _default_device


@dataclass
class TrainConfig:
    """LoRA 학습 설정. UI의 한국어 설명·추천값과 1:1로 매칭된다."""

    model_path: str                      # 학습시킬 베이스 체크포인트 (.safetensors)
    dataset_dir: str                     # 이미지 + .txt 캡션 폴더
    output_dir: str                      # LoRA 저장 폴더
    trigger_word: str = ""               # 캡션 맨 앞에 삽입 (예: sks girl)
    rank: int = 16                       # LoRA rank (dim)
    lr: float = 1e-4                     # 학습률 (AdamW)
    max_train_steps: int = 1500          # 총 스텝
    resolution: int = 1024               # 학습 해상도 (정사각)
    train_text_encoder: bool = False     # 텍스트 인코더도 LoRA 학습
    gradient_checkpointing: bool = True  # VRAM 절약
    checkpoint_every: int = 500          # 중간 저장 간격 (스텝)
    seed: int | None = None
    # 진행 콜백: on_step(step, total_steps, loss)
    on_step: Callable[[int, int, float], None] | None = field(
        default=None, repr=False)


def _list_pairs(dataset_dir: str) -> list[tuple[str, str]]:
    """이미지 파일과 동명 .txt 캡션을 짝지어 반환한다."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    pairs = []
    for name in sorted(os.listdir(dataset_dir)):
        low = name.lower()
        if not low.endswith(exts):
            continue
        base = os.path.splitext(name)[0]
        txt = os.path.join(dataset_dir, base + ".txt")
        pairs.append((os.path.join(dataset_dir, name),
                      txt if os.path.isfile(txt) else ""))
    if not pairs:
        raise RuntimeError(
            f"데이터셋 폴더에 이미지가 없어요: {dataset_dir}")
    return pairs


def _read_caption(txt_path: str, trigger_word: str) -> str:
    caption = ""
    if txt_path:
        with open(txt_path, encoding="utf-8") as f:
            caption = f.read().strip().replace("\n", " ")
    if trigger_word:
        caption = f"{trigger_word}, {caption}" if caption else trigger_word
    return caption


def train_lora(cfg: TrainConfig) -> str:
    """LoRA를 학습하고 저장한 .safetensors 경로를 반환한다."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from diffusers.utils import check_min_version
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    from PIL import Image
    from torchvision import transforms

    check_min_version("0.30.0")

    device = _default_device()  # GPU 없으면 여기서 RuntimeError
    weight_dtype = torch.float16

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    # -- 베이스 모델 로드 -------------------------------------------
    pipe = StableDiffusionXLPipeline.from_single_file(
        cfg.model_path, torch_dtype=weight_dtype)
    unet, vae = pipe.unet, pipe.vae
    text_enc1, text_enc2 = pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe

    vae.requires_grad_(False)
    text_enc1.requires_grad_(False)
    text_enc2.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_enc1.to(device, dtype=weight_dtype)
    text_enc2.to(device, dtype=weight_dtype)

    # -- LoRA 주입 ---------------------------------------------------
    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if cfg.train_text_encoder:
        text_enc1 = get_peft_model(text_enc1, lora_cfg)
        text_enc2 = get_peft_model(text_enc2, lora_cfg)

    # -- 데이터셋 ----------------------------------------------------
    pairs = _list_pairs(cfg.dataset_dir)

    train_tf = transforms.Compose([
        transforms.Resize(cfg.resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class CapDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            img_path, txt_path = pairs[i]
            img = Image.open(img_path).convert("RGB")
            return {
                "pixel_values": train_tf(img),
                "caption": _read_caption(txt_path, cfg.trigger_word),
            }

    loader = DataLoader(CapDataset(), batch_size=1, shuffle=True)

    # -- 옵티마이저 --------------------------------------------------
    params = [p for p in unet.parameters() if p.requires_grad]
    if cfg.train_text_encoder:
        params += [p for p in text_enc1.parameters() if p.requires_grad]
        params += [p for p in text_enc2.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    # -- SDXL 조건 인코딩 헬퍼 ---------------------------------------
    def encode_prompt(caption: str):
        t1 = tok1(caption, padding="max_length",
                  max_length=tok1.model_max_length,
                  truncation=True, return_tensors="pt")
        t2 = tok2(caption, padding="max_length",
                  max_length=tok2.model_max_length,
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            e1 = text_enc1(t1.input_ids.to(device))[0]
            enc2_out = text_enc2(t2.input_ids.to(device))
            e2 = enc2_out[0]
            pooled = enc2_out[1]  # pooler 출력 → SDXL text_embeds
        prompt_embeds = torch.concat([e1, e2], dim=-1)
        add_text_embeds = pooled
        add_time_ids = torch.tensor(
            [[cfg.resolution, cfg.resolution, 0, 0,
              cfg.resolution, cfg.resolution]],
            dtype=weight_dtype, device=device).repeat(1, 1)
        return prompt_embeds, add_text_embeds, add_time_ids

    # -- 학습 루프 ---------------------------------------------------
    os.makedirs(cfg.output_dir, exist_ok=True)
    unet.to(device)
    unet.train()

    data_iter = iter(loader)
    for step in range(1, cfg.max_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,),
            device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, add_text_embeds, add_time_ids = encode_prompt(
            batch["caption"][0])
        added_cond = {"text_embeds": add_text_embeds,
                      "time_ids": add_time_ids}

        model_pred = unet(noisy_latents, timesteps, prompt_embeds,
                          added_cond_kwargs=added_cond).sample
        target = noise
        loss = torch.nn.functional.mse_loss(model_pred.float(),
                                            target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if cfg.on_step is not None:
            cfg.on_step(step, cfg.max_train_steps, float(loss.detach()))

        if step % cfg.checkpoint_every == 0 or step == cfg.max_train_steps:
            _save(unet, text_enc1, text_enc2, cfg, step,
                  get_peft_model_state_dict, save_file)

    final = os.path.join(cfg.output_dir, "lora.safetensors")
    return final


def _save(unet, text_enc1, text_enc2, cfg: TrainConfig, step: int,
          get_peft_model_state_dict, save_file) -> str:
    """LoRA 가중치를 .safetensors로 저장한다."""
    import torch

    state = get_peft_model_state_dict(unet)
    if cfg.train_text_encoder:
        for i, enc in enumerate((text_enc1, text_enc2), start=1):
            for k, v in get_peft_model_state_dict(enc).items():
                state[f"text_encoder_{i}.{k}"] = v
    name = (f"lora-step{step:06d}.safetensors"
            if step < cfg.max_train_steps else "lora.safetensors")
    path = os.path.join(cfg.output_dir, name)
    save_file(state, path)
    return path
