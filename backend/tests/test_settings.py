"""settings.py 테스트: 기본값·범위·validate()·다국어 필드."""

from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.settings import SETTINGS, defaults, get_setting, validate


def test_defaults_keys_match_settings():
    assert set(defaults()) == {s.key for s in SETTINGS}


def test_recommended_within_bounds():
    for s in SETTINGS:
        if s.kind != "number":
            continue
        v = s.recommended
        assert isinstance(v, (int, float)) and not isinstance(v, bool)
        if s.min is not None:
            assert v >= s.min, s.key
        if s.max is not None:
            assert v <= s.max, s.key


def test_sampler_options_expected():
    s = get_setting("sampler")
    assert s.options == ["er_sde",
                         "euler", "euler_ancestral",
                         "dpmpp_2m", "dpmpp_2m_karras",
                         "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_karras",
                         "dpmpp_sde", "dpmpp_sde_karras",
                         "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_karras",
                         "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_karras",
                         "dpmpp_3m_sde", "dpmpp_3m_sde_karras",
                         "dpm2", "dpm2_ancestral", "dpm_fast", "dpm_adaptive",
                         "lms", "heun", "ddim", "plms", "unipc",
                         "restart", "sa_solver", "deis", "lcm", "tcd"]
    assert s.recommended == "er_sde"


def test_steps_bounds():
    s = get_setting("steps")
    assert (s.min, s.max, s.recommended) == (20, 50, 50)


def test_cfg_scale_max():
    s = get_setting("cfg_scale")
    assert s.max == 9.0


def test_validate_defaults_ok():
    assert validate(defaults()) == {}


def test_validate_out_of_range():
    assert "steps" in validate({"steps": 10})       # min 20
    assert "steps" in validate({"steps": 60})       # max 50
    assert "cfg_scale" in validate({"cfg_scale": 10})  # max 9.0
    assert "width" in validate({"width": 100})


def test_validate_bad_choice():
    assert "sampler" in validate({"sampler": "nope"})


def test_validate_unknown_key():
    assert "strength" in validate({"strength": 0.75})


def test_validate_wrong_type():
    assert "steps" in validate({"steps": "50"})
    assert "prompt" in validate({"prompt": 123})


def test_validate_seed_allows_any_nonneg():
    assert validate({"seed": -1}) == {}
    assert validate({"seed": 0}) == {}
    assert validate({"seed": 999999}) == {}
    assert "seed" in validate({"seed": -2})


def test_all_settings_have_translations():
    for s in SETTINGS:
        assert s.label_en.strip(), s.key
        assert s.description_en.strip(), s.key
        assert s.label_ja.strip(), s.key
        assert s.description_ja.strip(), s.key


def test_api_settings_includes_language_fields():
    client = TestClient(app_module.create_app())
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == len(SETTINGS)
    for entry in body:
        for f in ("label_en", "description_en", "label_ja", "description_ja"):
            assert entry[f].strip(), (entry["key"], f)
    by_key = {e["key"]: e for e in body}
    assert by_key["steps"]["label_en"] == "Steps"
    assert by_key["steps"]["label_ja"] == "ステップ数"
