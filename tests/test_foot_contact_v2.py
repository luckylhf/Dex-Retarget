import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dex_frax.v8.foot_contact_v2 import DEFAULTS, orientation_targets, support_labels


def test_tilt_crossing_does_not_choose_a_pivot():
    count = 30
    raw = Rotation.from_euler("x", np.linspace(-16, -21, count)[:, None], degrees=True).as_matrix()
    heel = np.tile([0.0, 0.0, 0.005], (count, 1))
    toe = np.tile([0.2, 0.0, 0.025], (count, 1))
    stable = np.ones(count, dtype=bool)
    speed = np.full(count, 0.04)
    labels = support_labels(heel, toe, stable, stable, 5, DEFAULTS)
    cfg = dict(v2=DEFAULTS, height_spread_m=0.05, height_tolerance_m=0.035, transition_s=0.1)
    r, _, side, weight = orientation_targets(
        raw, labels, dict(heel=heel, toe=toe), stable, stable, speed, speed, 0.0, 50.0, cfg,
    )
    np.testing.assert_array_equal(labels, np.ones(count))
    np.testing.assert_allclose(r, np.broadcast_to(np.eye(3), r.shape))
    np.testing.assert_array_equal(side, np.zeros(count))
    np.testing.assert_allclose(weight, np.ones(count))


def test_single_end_requires_sustained_opposite_lift():
    heel = np.zeros((20, 3))
    toe = np.zeros_like(heel)
    heel[2:4, 2] = 0.04
    heel[6:12, 2] = 0.04
    toe[13:20, 2] = 0.04
    stable_heel = heel[:, 2] == 0
    stable_toe = toe[:, 2] == 0
    labels = support_labels(heel, toe, stable_heel, stable_toe, 5, DEFAULTS)
    np.testing.assert_array_equal(labels[:6], np.ones(6))
    np.testing.assert_array_equal(labels[6:12], np.full(6, 2))
    np.testing.assert_array_equal(labels[13:], np.full(7, 3))


def test_sole_edge_evidence_only_releases_roll():
    n = 12
    raw = Rotation.from_euler("xyz", [[-20, -7, 30]] * n, degrees=True).as_matrix()
    heel = np.tile([0.0, 0.0, 0.01], (n, 1))
    toe = np.tile([0.2, 0.0, 0.03], (n, 1))
    stable = np.ones(n, dtype=bool)
    labels = support_labels(heel, toe, stable, stable, 5, DEFAULTS)
    corners = np.tile([[0, 0.02, 0.001], [0, -0.02, 0.041],
                       [0.2, 0.02, 0.002], [0.2, -0.02, 0.042]], (n, 1, 1))
    cfg = dict(v2=DEFAULTS, height_spread_m=0.05, height_tolerance_m=0.035, transition_s=0.1)
    args = (stable, stable, np.zeros(n), np.zeros(n), 0.0, 50.0, cfg)
    plain, _, _, _ = orientation_targets(raw, labels, dict(heel=heel, toe=toe), *args)
    edged, _, side, _ = orientation_targets(raw, labels, dict(heel=heel, toe=toe, sole_corners=corners), *args)
    plain = Rotation.from_matrix(plain).as_euler("xyz", degrees=True)
    edged = Rotation.from_matrix(edged).as_euler("xyz", degrees=True)
    np.testing.assert_allclose(side, np.ones(n))
    np.testing.assert_allclose(edged[:, 0], -13)
    np.testing.assert_allclose(edged[:, 1:], plain[:, 1:])


@pytest.mark.parametrize("command", ["convert", "batch"])
@pytest.mark.parametrize("policy", [None, "v1", "v2"])
def test_cli_foot_contact_policy_reaches_dispatch(monkeypatch, tmp_path, command, policy):
    from dex_frax import cli, managed_batch, resources
    from dex_frax.contract import conversion_contract

    output = tmp_path / "output"
    argv = ["dex-frax", command, "input", str(output)]
    if policy is not None:
        argv.extend(["--foot-contact", policy])
    monkeypatch.setattr(sys, "argv", argv)
    seen = []

    def dispatch(args):
        seen.append((args.foot_contact, conversion_contract(args).get("foot_contact", "v1")))
        if command == "batch":
            output.mkdir()
            (output / "batch.json").write_text('{"status": "complete"}')
            return []

    monkeypatch.setattr(cli, "single_auto", dispatch)
    monkeypatch.setattr(managed_batch, "run", dispatch)
    monkeypatch.setattr(resources, "contain_linux", lambda _: None)
    cli.main()
    expected = "v2" if policy is None else policy
    assert seen == [(expected, expected)]


def test_repair_checkpoint_accepts_numpy_diagnostic_scalars(tmp_path):
    from dex_frax.cli import atomic_json

    path = tmp_path / "refine_state.json"
    diagnostics = {"attempts": [{"feasible": np.bool_(False), "iterations": np.int32(3),
                                "deficit": np.float32(0.125)}]}
    atomic_json(path, diagnostics)
    assert json.loads(path.read_text()) == {
        "attempts": [{"feasible": False, "iterations": 3, "deficit": 0.125}],
    }
