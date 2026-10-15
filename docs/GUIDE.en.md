# Atelier AI User Guide

[한국어](GUIDE.md) · [日本語](GUIDE.ja.md)

## What is this tool?

Atelier AI is a **desktop app dedicated to image generation**. It runs entirely on your own computer, not as a web service, so your generated images never leave your machine. Completely free and open source (MIT).

It's a good fit if you:

- Want to generate anime-style images yourself
- Prefer **Danbooru-style tags** (`1girl, long hair`) over English sentences for prompts
- Use Anima-family models
- Want something easy enough for beginners

## Screen layout

Six menus on the left — that's everything.

| Menu | What it does |
|---|---|
| Create image | Generate images from text prompts. Batch queue, X/Y plot, and Hires Fix live here |
| Edit image | Inpainting (redraw part of an image), outpainting (expand the canvas), pose-guided generation |
| Models & Training | Manage checkpoints, LoRA, and ControlNet; train LoRA on your own images |
| Library | A place to collect your favorite results |
| History | Past generations and the settings used |
| Settings | Generation defaults, language (한국어/English/日本語), updates |

## First-time flow

1. **Add a model**: Put a checkpoint (`.safetensors`) in the models folder. You can check the folder location in Settings. The app opens even without a model.
2. **Write a prompt**: In Create image, describe what you want with Danbooru-style tags. If you're stuck, open the **Prompt Helper**. It has recommended tag, style, and composition chips, and the 🔍 search tab finds English tags from Korean keywords. (`긴 머리` → `long hair`)
3. **Generate**: Start with single images while adjusting steps, CFG, and sampler. Every setting shows an explanation and a recommended value.

## Features worth knowing

- **Batch queue**: Queue up to 200 prompts and generate them one by one. If one fails, it moves on to the next.
- **X/Y plot**: Compare sampler, CFG, steps, and seeds in a grid. (Up to 36 cells)
- **Hires Fix**: Upscale and refine a small generation into high resolution. (Single generation only)
- **Face fix**: Click `✨ 얼굴 보정` (Fix face) in the gallery to detect and redraw faces.
- **Upscale**: Enlarge with RealESRGAN. A slider controls how much of the original to blend in.
- **Outpainting**: Click `🖼️ 확장` (Expand) in the gallery to extend the canvas up, down, left, or right.
- **LoRA training**: Train character or style LoRA from 15–30 of your own images. Captions are generated automatically.

## Good to know

- **A GPU is required.** You need an NVIDIA GPU (CUDA) or Apple Silicon (Metal). Generation doesn't work on CPU alone.
- **Tag descriptions download on first run.** The app downloads the Danbooru wiki Korean translation data once, for the Korean tag descriptions in the Prompt Helper. (That's the only time it needs the internet.)
- **Switch language in the header.** The `한 / EN / 日` switch at the top right changes everything, including setting descriptions.
- **Updates are announced automatically.** When a new version is out, you can download, install, and restart from the Settings tab.

## FAQ

**Q. Is it free?**
A. Yes. It's MIT-licensed open source — free to use and modify.

**Q. Can I publish the images I make?**
A. The app itself is MIT so you're free, but please check the license of the model you used. Commercial-use terms differ per model.

**Q. Does it work on Windows and macOS?**
A. Yes. Linux, macOS, and Windows are all supported.
