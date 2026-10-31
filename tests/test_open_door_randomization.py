"""Per-episode door physics, actual solver effects, and replay context consistency."""

import pytest
import torch

from supermimic.tasks.open_door.assets import DEFAULT_MOTION_DIR
from supermimic.tasks.open_door.events import DoorRandomizationCfg


PARAMETERS = (("mass", 39), ("damping", 40), ("hinge_friction", 41),
              ("robot_friction", 43), ("stiffness", 44))


@pytest.mark.parametrize("name,bounds", [
    ("mass", (0., 1.)), ("mass", (2., 1.)), ("mass", (1., float("inf"))),
    ("damping", (-1., 1.)), ("damping", (float("nan"), 1.)),
    ("hinge_friction", (1., .5)), ("stiffness", (-1., 1.)),
    ("robot_friction", (0., float("inf"))), ("robot_friction", (1., 0.)),
])
def test_invalid_physical_ranges_are_rejected(name, bounds):
    with pytest.raises(ValueError):
        DoorRandomizationCfg(**{name: bounds})


def test_fixed_ranges_and_zero_resistance_are_valid():
    cfg = DoorRandomizationCfg(mass=(8., 8.), damping=(0., 0.),
                              hinge_friction=(0., 0.), stiffness=(0., 0.),
                              robot_friction=(0., 0.))
    assert cfg.enabled and cfg.mass == (8., 8.)


def test_door_randomization_is_a_reset_event_in_training_and_play():
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg
    from supermimic.tasks.open_door.events import randomize_door_physics
    for play in (False, True):
        cfg = make_env_cfg(play)
        events = [term for term in cfg.events.values() if term.func is randomize_door_physics]
        assert len(events) == 1 and events[0].mode == "reset"
        assert cfg.commands["motion"].door_randomization_cfg.enabled


@pytest.fixture(scope="module")
def randomized_door(cuda_device):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.rl.vecenv import ReplayEnvAdapter
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg
    if not list(DEFAULT_MOTION_DIR.glob("*.npz")):
        pytest.skip("Convert the supplied OpenDoor motion first")
    cfg = make_env_cfg(play=False)
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.auto_reset = 4, 2.5, False
    cfg.commands["motion"].motion_dir = str(DEFAULT_MOTION_DIR)
    cfg.commands["motion"].reset_noise = False
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        yield ReplayEnvAdapter(env)
    finally:
        env.close()


def _reset(replay, frames=(0, 10, 20, 30), ids=None):
    return replay.reset(ids, torch.tensor(frames, device=replay.device))


def _door_indices(replay):
    model, door = replay.env.sim.mj_model, replay.env.scene["door"]
    joint = int(door.indexing.joint_ids[0])
    panel = next(int(i) for i in door.indexing.geom_ids
                 if model.geom(int(i)).name.endswith("door_panel"))
    return joint, int(model.jnt_bodyid[joint]), int(model.jnt_dofadr[joint]), panel


def _physics(replay):
    sim = replay.env.sim
    joint, body, dof, panel = _door_indices(replay)
    return torch.stack((sim.model.body_mass[:, body], sim.model.dof_damping[:, dof],
                        sim.model.dof_frictionloss[:, dof], sim.model.geom_friction[:, panel, 0],
                        sim.model.jnt_stiffness[:, joint]), -1)


def _assert_context_matches_model(replay):
    columns = [index for _, index in PARAMETERS]
    torch.testing.assert_close(_physics(replay), replay.xi[:, columns], atol=0, rtol=0)
    sim = replay.env.sim
    _, body, _, _ = _door_indices(replay)
    default_mass = sim.get_default_field("body_mass")[body]
    default_inertia = sim.get_default_field("body_inertia")[body]
    expected = default_inertia[None] * (replay.xi[:, 39] / default_mass)[:, None]
    torch.testing.assert_close(sim.model.body_inertia[:, body], expected)
    torch.testing.assert_close(sim.model.body_subtreemass[:, body], replay.xi[:, 39])


def test_each_environment_samples_physics_and_holds_it_until_reset(randomized_door):
    replay = randomized_door
    observations, _ = _reset(replay)
    assert observations["context"].shape == (4, 46)
    for name, index in PARAMETERS:
        low, high = getattr(replay.command.cfg.door_randomization_cfg, name)
        value = replay.xi[:, index]
        assert ((low <= value) & (value <= high)).all() and value.std() > 0
    _assert_context_matches_model(replay)
    before_context, before_physics = replay.xi.clone(), _physics(replay).clone()
    replay.env.step(torch.zeros(4, 29, device=replay.device))
    torch.testing.assert_close(replay.xi, before_context, atol=0, rtol=0)
    torch.testing.assert_close(_physics(replay), before_physics, atol=0, rtol=0)


def test_partial_resets_preserve_other_worlds_and_do_not_compound_inertia(randomized_door):
    replay = randomized_door
    _reset(replay)
    before_context, before_physics = replay.xi.clone(), _physics(replay).clone()
    before_state = replay.dynamics.pack(replay.state()).clone()
    ids = torch.tensor([0, 2], device=replay.device)
    for _ in range(3):
        _reset(replay, (40, 60), ids)
        _assert_context_matches_model(replay)
        torch.testing.assert_close(replay.xi[[1, 3]], before_context[[1, 3]], atol=0, rtol=0)
        torch.testing.assert_close(_physics(replay)[[1, 3]], before_physics[[1, 3]], atol=0, rtol=0)
        torch.testing.assert_close(replay.dynamics.pack(replay.state())[[1, 3]],
                                   before_state[[1, 3]], atol=0, rtol=0)
        assert (_physics(replay)[ids] != before_physics[ids]).all()
        # Scene geometry, spring reference, and startup robot physics stay fixed.
        torch.testing.assert_close(replay.xi[:, 33:39], before_context[:, 33:39], atol=0, rtol=0)
        torch.testing.assert_close(replay.xi[:, [0, 30, 31, 32, 42, 45]],
                                   before_context[:, [0, 30, 31, 32, 42, 45]], atol=0, rtol=0)


@pytest.mark.parametrize("setting", ["play", "disabled"])
def test_evaluation_or_disabled_randomization_restores_configured_values(randomized_door, monkeypatch, setting):
    replay, cfg = randomized_door, randomized_door.command.cfg
    if setting == "play":
        monkeypatch.setattr(cfg, "play", True)
    else:
        monkeypatch.setattr(cfg.door_randomization_cfg, "enabled", False)
    values = dict(door_mass=9.5, door_damping=3.25, door_friction=.35,
                  robot_door_friction=.8, door_stiffness=14.)
    for name, value in values.items():
        monkeypatch.setattr(cfg, name, value)
    for _ in range(2):
        _reset(replay)
        expected = replay.xi.new_tensor(list(values.values())).expand(4, -1)
        torch.testing.assert_close(_physics(replay), expected)
        _assert_context_matches_model(replay)


def test_terminal_transition_keeps_physics_from_before_reset(randomized_door):
    from supermimic.rl.storage import ReplayBuffer
    replay = randomized_door
    _reset(replay)
    _reset(replay, (int(replay.motion.clip_end[0]) - 1,),
           torch.tensor([0], device=replay.device))
    before = replay.xi.clone()
    transition = replay.step(torch.zeros(4, 29, device=replay.device))
    assert transition["done"][0] and replay.ep_step[0] == 0
    torch.testing.assert_close(transition["xi"], before, atol=0, rtol=0)
    columns = [index for _, index in PARAMETERS]
    assert (replay.xi[0, columns] != before[0, columns]).all()
    buffer = ReplayBuffer(2, 4, replay.device, state_dim=73, action_dim=29, context_dim=46)
    buffer.add(transition)
    _reset(replay)
    torch.testing.assert_close(transition["xi"], before, atol=0, rtol=0)
    torch.testing.assert_close(buffer.data["xi"][0], before, atol=0, rtol=0)


def _isolated_hinge(replay, angle=.4, speed=.7):
    """Remove contact so measured hinge forces have an independent analytic oracle."""
    robot, door = replay.command.robot, replay.env.scene["door"]
    pose = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w), -1).clone()
    pose[:, 0] += 20
    robot.write_root_link_pose_to_sim(pose)
    door.write_joint_state_to_sim(torch.full((4, 1), angle, device=replay.device),
                                 torch.full((4, 1), speed, device=replay.device))
    replay.env.sim.forward()


def test_sampled_stiffness_and_damping_change_solver_passive_torque(randomized_door):
    replay = randomized_door
    _reset(replay)
    _isolated_hinge(replay)
    _, _, dof, _ = _door_indices(replay)
    expected = -replay.xi[:, 44] * (.4 - replay.xi[:, 45]) - replay.xi[:, 40] * .7
    torch.testing.assert_close(replay.env.sim.data.qfrc_passive[:, dof], expected,
                               atol=2e-5, rtol=2e-5)
    assert expected.std() > 0


def test_mass_and_hinge_friction_change_actual_acceleration(randomized_door, monkeypatch):
    replay, cfg = randomized_door, randomized_door.command.cfg.door_randomization_cfg
    # Three otherwise identical doors: double mass halves acceleration; larger
    # dry friction reduces its magnitude. The fourth duplicates the baseline.
    for i, (mass, friction) in enumerate(((8., .5), (16., .5), (8., 2.), (8., .5))):
        for name, value in dict(mass=mass, damping=5., hinge_friction=friction,
                                robot_friction=.7, stiffness=20.).items():
            monkeypatch.setattr(cfg, name, (value, value))
        _reset(replay, (0,), torch.tensor([i], device=replay.device))
    _isolated_hinge(replay, speed=0.)
    _, _, dof, _ = _door_indices(replay)
    acceleration = replay.env.sim.data.qacc[:, dof]
    assert (acceleration < 0).all()
    torch.testing.assert_close(acceleration[1], acceleration[0] / 2, atol=2e-3, rtol=2e-3)
    assert acceleration[0].abs() > acceleration[2].abs()
    torch.testing.assert_close(acceleration[3], acceleration[0], atol=2e-3, rtol=2e-3)


def test_robot_door_contacts_use_sampled_door_friction(randomized_door):
    replay = randomized_door
    robot_geoms = replay.command.robot.indexing.geom_ids
    _, _, _, panel = _door_indices(replay)
    found = False
    for frame in (30, 80, 110, 140):
        _reset(replay, (frame,) * 4)
        replay.env.sim.forward()
        data = replay.env.sim.data
        count = int(data.nacon[0])
        assert count < replay.env.sim.wp_data.naconmax
        contact = data.contact
        geoms = contact.geom[:count]
        selected = (geoms == panel).any(-1) & torch.isin(geoms, robot_geoms).any(-1)
        if selected.any():
            worlds = contact.worldid[:count][selected].long()
            expected = replay.xi[worlds, 43:44].expand(-1, 2)
            torch.testing.assert_close(contact.friction[:count][selected, :2], expected,
                                       atol=1e-6, rtol=1e-6)
            found = True
    assert found, "Reference frames must exercise real robot-door contact"
