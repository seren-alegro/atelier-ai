"""settings.py 테스트: 기본값·범위·validate()."""

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
    assert s.options == ["er_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde",
                        "euler_ancestral", "dpmpp_2m_karras", "ddim"]
    assert s.recommended == "er_sde"


def test_experimental_sampler_options_expected():
    s = get_setting("experimental_sampler")
    assert s.options == ["restart", "sa_solver", "deis", "lcm", "tcd"]
    assert s.recommended == "restart"


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
    assert "experimental_sampler" in validate({"experimental_sampler": "nope"})


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
