"""생성 엔진 (diffusers 기반). MVP에서는 스텁."""

from __future__ import annotations


class Pipeline:
    """txt2img/img2img/inpaint 파이프라인. diffusers 연동 예정."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._pipe = None

    def load(self) -> None:
        raise NotImplementedError("diffusers 연동 후 구현")

    def txt2img(self, params: dict) -> list[str]:
        raise NotImplementedError("diffusers 연동 후 구현")

    def img2img(self, params: dict, image: str) -> list[str]:
        raise NotImplementedError("diffusers 연동 후 구현")

    def inpaint(self, params: dict, image: str, mask: str) -> list[str]:
        raise NotImplementedError("diffusers 연동 후 구현")
