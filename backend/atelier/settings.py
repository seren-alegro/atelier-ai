"""생성 설정 정의.

Atelier AI의 핵심 차별점: 모든 설정에 한국어 설명과 추천값을 붙인다.
추천값은 Anima 계열 모델 기준 초안이며, 실측 후 조정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Setting:
    key: str
    label: str            # UI 표시 이름 (한국어)
    description: str      # 초보자용 한국어 설명
    recommended: object   # 추천값 (Anima 기준)
    kind: str = "number"  # number | choice | text | toggle
    min: float | None = None
    max: float | None = None
    options: list[str] = field(default_factory=list)
    unit: str = ""
    label_en: str = ""        # 영어 UI 표시 이름
    description_en: str = ""  # 영어 설명
    label_ja: str = ""        # 일본어 UI 표시 이름
    description_ja: str = ""  # 일본어 설명


SETTINGS: list[Setting] = [
    Setting(
        key="prompt",
        label="프롬프트",
        description="그리고 싶은 내용을 Danbooru식 태그로 적어주세요. "
                    "쉼표로 구분하고, 앞에 적을수록 강하게 반영돼요.",
        recommended="1girl, solo",
        kind="text",
        label_en="Prompt",
        description_en="Describe what you want to draw using Danbooru-style tags. "
                       "Separate with commas — tags written earlier have stronger influence.",
        label_ja="プロンプト",
        description_ja="描きたい内容をDanbooru式タグで書いてください。"
                       "カンマで区切り、先に書くほど強く反映されます。",
    ),
    Setting(
        key="negative_prompt",
        label="네거티브 프롬프트",
        description="그리지 않았으면 하는 것을 적어주세요. "
                    "손가락이 이상하게 나오는 게 싫으면 여기에 적어요.",
        recommended="worst quality, low quality, bad anatomy",
        kind="text",
        label_en="Negative prompt",
        description_en="Write what you don't want drawn. "
                       "If you hate weird-looking fingers, put it here.",
        label_ja="ネガティブプロンプト",
        description_ja="描いてほしくないものを書いてください。"
                       "指が変になるのが嫌ならここに書きます。",
    ),
    Setting(
        key="steps",
        label="스텝 수",
        description="그림을 다듬는 반복 횟수예요. Anima 공식 사양은 50 스텝입니다.",
        recommended=50,
        min=20,
        max=50,
        label_en="Steps",
        description_en="How many refinement iterations to run. "
                       "The official Anima spec is 50 steps.",
        label_ja="ステップ数",
        description_ja="画像を整える反復回数です。Anima公式仕様は50ステップです。",
    ),
    Setting(
        key="cfg_scale",
        label="CFG 스케일",
        description="프롬프트를 얼마나 빡빡하게 따를지 정해요. "
                    "높을수록 프롬프트 충실, 낮을수록 AI 마음대로 자유로워져요. "
                    "너무 높이면 색이 타고 깨져요.",
        recommended=5.0,
        min=1.0,
        max=9.0,
        label_en="CFG scale",
        description_en="How strictly to follow the prompt. "
                       "Higher sticks closer to the prompt, lower lets the AI roam free. "
                       "Too high burns colors and breaks the image.",
        label_ja="CFGスケール",
        description_ja="プロンプトにどれだけ忠実に従うかを決めます。"
                       "高くなるほどプロンプト通り、低くなるほどAIの自由になります。"
                       "高すぎると色が飛んで壊れます。",
    ),
    Setting(
        key="sampler",
        label="샘플러",
        description="그림을 그려나가는 방식이에요. Anima는 SDE 계열이 잘 맞아요. "
                    "er_sde가 기본 추천이에요. 실험실 항목은 궁합을 직접 돌려보셔야 해요 "
                    "(restart는 느리지만 깔끔하게, lcm/tcd는 몇 스텝 만에 뽑아요).",
        recommended="er_sde",
        kind="choice",
        options=["er_sde",
                 "euler", "euler_ancestral",
                 "dpmpp_2m", "dpmpp_2m_karras",
                 "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_karras",
                 "dpmpp_sde", "dpmpp_sde_karras",
                 "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_karras",
                 "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_karras",
                 "dpmpp_3m_sde", "dpmpp_3m_sde_karras",
                 "dpm2", "dpm2_ancestral", "dpm_fast", "dpm_adaptive",
                 "lms", "heun", "ddim", "plms", "unipc",
                 "restart", "sa_solver", "deis", "lcm", "tcd"],
        label_en="Sampler",
        description_en="The drawing method. Anima works well with the SDE family, "
                       "and er_sde is the default recommendation. Lab entries need "
                       "hands-on testing (restart is slow but clean, lcm/tcd finish "
                       "in a few steps).",
        label_ja="サンプラー",
        description_ja="画像を描き進める方式です。AnimaにはSDE系がよく合います。"
                       "er_sdeが基本のおすすめです。実験室の項目は相性を自分で試す"
                       "必要があります（restartは遅いが綺麗に、lcm/tcdは数ステップで描きます）。",
    ),
    Setting(
        key="width",
        label="가로",
        description="그림의 가로 크기예요. 클수록 VRAM을 많이 써요.",
        recommended=832,
        min=256,
        max=2048,
        label_en="Width",
        description_en="Image width in pixels. Larger uses more VRAM.",
        label_ja="横幅",
        description_ja="画像の横サイズです。大きいほどVRAMを多く使います。",
    ),
    Setting(
        key="height",
        label="세로",
        description="그림의 세로 크기예요. 세로 그림엔 832x1216이 무난해요.",
        recommended=1216,
        min=256,
        max=2048,
        label_en="Height",
        description_en="Image height in pixels. 832x1216 is a safe pick for portraits.",
        label_ja="縦幅",
        description_ja="画像の縦サイズです。縦長の絵には832x1216が無難です。",
    ),
    Setting(
        key="seed",
        label="시드",
        description="그림의 랜덤 씨앗이에요. -1이면 매번 바뀌고, "
                    "숫자를 고정하면 같은 구도를 다시 뽑을 수 있어요.",
        recommended=-1,
        min=-1,
        max=2**32 - 1,  # torch.Generator.manual_seed 오버플로우 방지
        label_en="Seed",
        description_en="The random seed. -1 picks a new one every time; "
                       "fixing a number lets you redraw the same composition.",
        label_ja="シード",
        description_ja="画像のランダムシードです。-1なら毎回変わり、"
                       "数字を固定すると同じ構図を再描画できます。",
    ),
    Setting(
        key="denoise_strength",
        label="디노이즈 강도",
        description="img2img·인페인트에서 원본을 얼마나 바꿀지 정해요. "
                    "높을수록 원본에서 멀어지고, 낮을수록 원본을 유지해요.",
        recommended=0.75,
        min=0.0,
        max=1.0,
        label_en="Denoise strength",
        description_en="How much to change the original in img2img/inpainting. "
                       "Higher drifts further from the original, lower preserves it.",
        label_ja="デノイズ強度",
        description_ja="img2img・インペイントで元画像をどれだけ変えるかを決めます。"
                       "高くなるほど元画像から離れ、低くなるほど維持します。",
    ),
    Setting(
        key="hires_fix_enabled",
        label="Hires Fix 사용",
        description="생성 후 고해상도 보정을 켜요. 작은 해상도로 먼저 뽑고 "
                    "키운 뒤 살짝만 다시 그려서 디테일을 살려요. "
                    "시간이 약 2배 걸리지만 얼굴·손 같은 잔붕괴가 줄어요.",
        recommended=False,
        kind="toggle",
        label_en="Use Hires Fix",
        description_en="Enables high-resolution fix after generation. Draws small first, "
                       "upscales, then lightly redraws for detail. Takes about 2x longer "
                       "but reduces broken faces and hands.",
        label_ja="Hires Fixを使用",
        description_ja="生成後の高解像度補正をオンにします。小さい解像度で先に描き、"
                       "拡大してから軽く描き直してディテールを整えます。"
                       "時間は約2倍かかりますが、顔・手などの崩れが減ります。",
    ),
    Setting(
        key="hires_fix_scale",
        label="Hires Fix 배율",
        description="Hires Fix에서 몇 배로 키울지 정해요. "
                    "2.0이면 가로·세로가 각각 2배가 돼요.",
        recommended=2.0,
        min=1.25,
        max=3.0,
        label_en="Hires Fix scale",
        description_en="How much to upscale in Hires Fix. "
                       "2.0 doubles width and height each.",
        label_ja="Hires Fix倍率",
        description_ja="Hires Fixで何倍に拡大するかを決めます。"
                       "2.0なら縦・横がそれぞれ2倍になります。",
    ),
    Setting(
        key="hires_fix_denoise",
        label="Hires Fix 디노이즈",
        description="Hires Fix 2단계에서 얼마나 다시 그릴지 정해요. "
                    "높을수록 원본 구도에서 멀어지고, 낮을수록 원본을 유지해요.",
        recommended=0.35,
        min=0.1,
        max=0.7,
        label_en="Hires Fix denoise",
        description_en="How much to redraw in the second Hires Fix pass. "
                       "Higher drifts from the original composition, lower preserves it.",
        label_ja="Hires Fixデノイズ",
        description_ja="Hires Fixの第2段階でどれだけ描き直すかを決めます。"
                       "高くなるほど元の構図から離れ、低くなるほど維持します。",
    ),
    Setting(
        key="batch_size",
        label="한 번에 뽑는 장수",
        description="한 번에 몇 장을 뽑을지 정해요. 많을수록 VRAM을 많이 써요.",
        recommended=1,
        min=1,
        max=8,
        label_en="Batch size",
        description_en="How many images to generate at once. More uses more VRAM.",
        label_ja="同時生成枚数",
        description_ja="一度に何枚描くかを決めます。多いほどVRAMを多く使います。",
    ),
    Setting(
        key="controlnet_model",
        label="ControlNet 모델",
        description="포즈 지정에 쓸 ControlNet 모델이에요. models/controlnet/ 폴더에 "
                    ".safetensors 파일을 넣으면 목록에서 고를 수 있고, "
                    "HuggingFace repo id를 직접 적어도 돼요. "
                    "비워두면 기본값(xinsir/controlnet-openpose-sdxl-1.0)을 받아와요.",
        recommended="",
        kind="text",
        label_en="ControlNet model",
        description_en="The ControlNet model for pose control. Drop .safetensors files "
                       "into models/controlnet/ to pick from the list, or type a "
                       "HuggingFace repo id directly. Leave empty to fetch the default "
                       "(xinsir/controlnet-openpose-sdxl-1.0).",
        label_ja="ControlNetモデル",
        description_ja="ポーズ指定に使うControlNetモデルです。models/controlnet/フォルダに"
                       ".safetensorsファイルを入れると一覧から選べ、HuggingFaceのrepo idを"
                       "直接書いても構いません。空ならデフォルト"
                       "（xinsir/controlnet-openpose-sdxl-1.0）を取得します。",
    ),
    Setting(
        key="controlnet_strength",
        label="ControlNet 강도",
        description="포즈를 얼마나 따를지 정해요. 높을수록 포즈맵을 빡빡하게 따르고, "
                    "낮을수록 포즈에서 자유로워져요.",
        recommended=0.8,
        min=0.0,
        max=1.0,
        label_en="ControlNet strength",
        description_en="How strictly to follow the pose. Higher follows the pose map "
                       "tightly, lower gives more freedom.",
        label_ja="ControlNet強度",
        description_ja="ポーズにどれだけ従うかを決めます。高くなるほどポーズマップに"
                       "厳密に従い、低くなるほど自由になります。",
    ),
]


def get_setting(key: str) -> Setting | None:
    for s in SETTINGS:
        if s.key == key:
            return s
    return None


def defaults() -> dict:
    """모든 설정의 추천값 딕셔너리."""
    return {s.key: s.recommended for s in SETTINGS}


def validate(params: dict) -> dict[str, str]:
    """params를 SETTINGS 기준으로 검증한다.

    반환: {설정키: 에러메시지}. 빈 dict면 유효함.
    """
    errors: dict[str, str] = {}
    for key, value in params.items():
        s = get_setting(key)
        if s is None:
            errors[key] = f"알 수 없는 설정입니다: {key}"
            continue
        if s.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors[key] = f"'{s.label}'은(는) 숫자여야 해요."
            elif ((s.min is not None and value < s.min)
                    or (s.max is not None and value > s.max)):
                lo = "" if s.min is None else str(s.min)
                hi = "" if s.max is None else str(s.max)
                errors[key] = f"'{s.label}'은(는) {lo}~{hi} 범위여야 해요."
        elif s.kind == "choice":
            if value not in s.options:
                errors[key] = f"'{s.label}'은(는) 다음 중 하나여야 해요: {', '.join(s.options)}"
        elif s.kind == "text":
            if not isinstance(value, str):
                errors[key] = f"'{s.label}'은(는) 문자열이어야 해요."
        elif s.kind == "toggle":
            if not isinstance(value, bool):
                errors[key] = f"'{s.label}'은(는) 참/거짓이어야 해요."
    return errors
