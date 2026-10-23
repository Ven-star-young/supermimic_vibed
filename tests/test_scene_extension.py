"""Native scene composition without pretending the G1 world model models objects."""

import mujoco
import pytest
import torch


def box_cfg():
    from mjlab.entity import EntityCfg

    def make_spec():
        return mujoco.MjSpec.from_string("""
            <mujoco model="test_box">
              <worldbody>
                <body name="box">
                  <freejoint name="box_free"/>
                  <geom name="box_collision" type="box" size="0.1 0.1 0.1" mass="1"/>
                </body>
              </worldbody>
            </mujoco>
        """)

    return EntityCfg(
        spec_fn=make_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(2.0, 0.0, 0.09)),
    )


def test_configured_scene_integrates_object_contacts_and_shared_reward(gpu_components):
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    from supermimic.envs.tracking_env import TrackingSim
    from supermimic.mdp import MDP, RewardCfg, unflatten_state
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    kin, motion, baseline_mdp = gpu_components
    cfg = make_env_cfg(play=True)
    cfg.scene.entities["box"] = box_cfg()
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="box_contact", primary=ContactMatch(mode="body", pattern="box", entity="box"),
            fields=("found", "force"), reduce="netforce", history_length=3,
        ),
    )
    cfg.rewards = RewardCfg(w_body_pos=2.0)
    mdp = MDP(kin, motion, baseline_mdp.default_q, cfg=cfg.rewards)
    with TrackingSim(2, motion, mdp, device="cuda:0", eval_mode=True, cfg=cfg) as env:
        ids = torch.arange(2, device=env.device)
        env.reset(ids, torch.tensor([3, 29], device=env.device))
        initial_box = env.scene["box"].data.root_link_pos_w.clone()
        action = torch.zeros(2, 29, device=env.device)
        for _ in range(8):
            result = env.step(action)
        box = env.scene["box"].data.root_link_pos_w
        contact = env.scene["box_contact"].data
        assert torch.isfinite(box).all()
        assert not torch.allclose(box, initial_box)
        assert contact.found is not None and (contact.found > 0).all()
        assert contact.force is not None and torch.isfinite(contact.force).all()
        assert contact.force.abs().sum() > 0
        predicted_state = unflatten_state(result["s1"])
        reward = mdp.reward(predicted_state, result["f"] + 1, action, result["a_prev"])
        baseline = baseline_mdp.reward(predicted_state, result["f"] + 1, action, result["a_prev"])
        torch.testing.assert_close(result["r"], reward)
        assert (result["r"] > baseline).all()
        # A partial reset must clear only that world's sensor history. Advancing
        # all sensors by even dt=0 would still roll the untouched world's buffer.
        history = contact.force_history.clone()
        assert history[1].abs().sum() > 0
        env.reset(torch.tensor([0], device=env.device), torch.tensor([3], device=env.device))
        after = env.scene["box_contact"].data.force_history
        torch.testing.assert_close(after[1], history[1], atol=0, rtol=0)
        torch.testing.assert_close(after[0], torch.zeros_like(after[0]), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["step", "interval"])
def test_event_state_changes_are_captured_in_transition(gpu_components, mode):
    from mjlab.managers.event_manager import EventTermCfg

    from supermimic.mdp import flatten_state, unflatten_state
    from supermimic.tasks import load_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    kin, motion, _ = gpu_components
    cfg = make_env_cfg(play=True)
    velocity = (0.8, -0.4, 0.2, 0.1, 0.2, -0.1)

    def write_velocity(env, env_ids):
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        value = torch.tensor(velocity, device=env.device).expand(len(ids), -1)
        env.robot.write_root_link_velocity_to_sim(value, env_ids=ids)

    cfg.events["write_velocity"] = EventTermCfg(
        func=write_velocity, mode=mode, interval_range_s=(0.02, 0.02) if mode == "interval" else None,
    )
    task = load_task("g1_tracking", device="cuda:0", play=True, motion=motion, kin=kin, cfg=cfg)
    mdp = task.mdp
    with task.make_env(2, device="cuda:0") as env:
        assert env.eval_mode
        assert not env.reset_noise
        env.reset(torch.arange(2, device=env.device), torch.tensor([3, 29], device=env.device))
        action = torch.zeros(2, 29, device=env.device)
        first = env.step(action)
        state = unflatten_state(first["s1"])
        expected = torch.tensor(velocity, device=env.device).expand(2, -1)
        torch.testing.assert_close(state[2], expected[:, :3], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(state[3], expected[:, 3:], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(first["s1"], flatten_state(env.state()))
        torch.testing.assert_close(first["r"], mdp.reward(state, first["f"] + 1, action, first["a_prev"]))
        second = env.step(action)
        torch.testing.assert_close(second["s"], first["s1"])
