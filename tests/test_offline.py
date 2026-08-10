from offline.static_guidance import OFFLINE_GUIDANCE
from pipeline.offline_fallback import get_static_guidance


def test_all_12_combinations_present():
    expected_keys = {
        (cls, stage)
        for cls in ("HEALTHY", "MSV", "MLN")
        for stage in (0, 1, 2, 3)
    }
    # HEALTHY only ever maps to stage 0 in practice, but 1/2/3 are stubbed
    # to prevent KeyErrors — all 12 combinations must resolve.
    assert expected_keys.issubset(OFFLINE_GUIDANCE.keys())


def test_get_static_guidance_returns_dict_with_required_keys():
    guidance = get_static_guidance("MSV", 2)
    for key in (
        "justification", "key_fact", "immediate_actions", "management",
        "spread", "distances", "prevention", "precautions", "detection",
        "control", "protocol_title", "protocol_steps", "tagalog",
    ):
        assert key in guidance, f"missing key: {key}"


def test_get_static_guidance_unknown_key_falls_back_to_healthy():
    guidance = get_static_guidance("MSV", 99)  # invalid stage, should never happen
    assert guidance == OFFLINE_GUIDANCE[("HEALTHY", 0)]


def test_trace_level_msv_mln_not_confused_with_healthy():
    # severity_to_stage() returns 0 for severity_pct < 1.0% even when the
    # classification is a genuine MSV/MLN positive, not just for HEALTHY.
    # These must NOT silently show "no disease detected" messaging.
    msv_stage0 = get_static_guidance("MSV", 0)
    mln_stage0 = get_static_guidance("MLN", 0)
    healthy = get_static_guidance("HEALTHY", 0)
    assert msv_stage0["justification"] != healthy["justification"]
    assert mln_stage0["justification"] != healthy["justification"]
    assert "no disease" not in msv_stage0["justification"].lower()
    assert "no disease" not in mln_stage0["justification"].lower()


def test_tagalog_present_for_every_entry():
    for key, guidance in OFFLINE_GUIDANCE.items():
        assert "tagalog" in guidance, f"missing tagalog for {key}"
        assert guidance["tagalog"]["justification"], f"empty tagalog justification for {key}"
