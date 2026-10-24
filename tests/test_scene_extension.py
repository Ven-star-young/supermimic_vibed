"""Native scene composition with object sensors and manager event timing."""

import mujoco
import pytest
import torch


def box_cfg():
    from mjlab.entity import EntityCfg

    def make_spec():
        return mujoco.MjSpec.from_string("""
            <mujoco model="test_box">
              <worldbody><body name="box"><freejoint name="box_free"/>
                <geom name="box_collision" type="box" size="0.1 0.1 0.1" mass="1"/>
              </body></worldbody>
            </mujoco>
        """)

    return EntityCfg(spec_fn=make_spec, init_state=EntityCfg.InitialStateCfg(pos=(2.0, 0.0, 0.09)))


def test_configured_scene_integrates_object_contacts_and_shared_reward(cuda_device, motion_dir):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from supermimic.mdp import MDP, RewardCfg
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg
    from tools._smoke import reset_frames

    cfg = make_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.commands["motion"].motion_dir = motion_dir
    cfg.commands["motion"].reward_cfg = RewardCfg(w_body_pos=2.0)
    cfg.scene.entities["box"] = box_cfg()
    cfg.scene.sensors = (ContactSensorCfg(
        name="box_contact", primary=ContactMatch(mode="body", pattern="box", entity="box"),
        fields=("found", "force"), reduce="netforce", history_length=3,
    ),)
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        cmd = env.command_manager.get_term("motion")
        baseline = MDP(cmd.mdp.kin, cmd.motion, cmd.mdp.default_q)
        reset_frames(env, torch.tensor([3, 29], device=cuda_device))
        initial_box = env.scene["box"].data.root_link_pos_w.clone()
        action = torch.zeros(2, 29, device=cuda_device)
        for _ in range(8):
            _, reward, _, _, _ = env.step(action)
        box, contact = env.scene["box"].data.root_link_pos_w, env.scene["box_contact"].data
        assert torch.isfinite(box).all() and not torch.allclose(box, initial_box)
        assert contact.found is not None and (contact.found > 0).all()
        assert contact.force is not None and torch.isfinite(contact.force).all()
        assert contact.force.abs().sum() > 0
        state = cmd.state()
        expected = cmd.mdp.reward(state, cmd.time_steps, action, action)
        torch.testing.assert_close(reward, expected)
        assert (reward > baseline.reward(state, cmd.time_steps, action, action)).all()
        history = contact.force_history.clone()
        assert history[1].abs().sum() > 0
        reset_frames(env, torch.tensor([3], device=cuda_device), torch.tensor([0], device=cuda_device))
        after = env.scene["box_contact"].data.force_history
        torch.testing.assert_close(after[1], history[1], atol=0, rtol=0)
        torch.testing.assert_close(after[0], torch.zeros_like(after[0]), atol=0, rtol=0)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["step", "interval"])
def test_event_state_changes_are_captured_in_transition(cuda_device, motion_dir, mode):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from supermimic.mdp import flatten_state, unflatten_state
    from supermimic.rl.vecenv import ReplayEnvAdapter
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg(play=True)
    cfg.scene.num_envs, cfg.auto_reset = 2, False
    cfg.commands["motion"].motion_dir = motion_dir
    velocity = (0.8, -0.4, 0.2, 0.1, 0.2, -0.1)

    def write_velocity(env, env_ids):
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        value = torch.tensor(velocity, device=env.device).expand(len(ids), -1)
        env.scene["robot"].write_root_link_velocity_to_sim(value, env_ids=ids)

    cfg.events["write_velocity"] = EventTermCfg(
        func=write_velocity, mode=mode, interval_range_s=(0.02, 0.02) if mode == "interval" else None,
    )
    native = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        env = ReplayEnvAdapter(native)
        env.reset(torch.arange(2, device=cuda_device), torch.tensor([3, 29], device=cuda_device))
        action = torch.zeros(2, 29, device=cuda_device)
        first = env.step(action)
        state = unflatten_state(first["s1"])
        expected = torch.tensor(velocity, device=cuda_device).expand(2, -1)
        torch.testing.assert_close(state[2], expected[:, :3], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(state[3], expected[:, 3:], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(first["s1"], flatten_state(env.state()))
        # Stock mjlab computes reward/termination before step/interval events.
        # Replay must still capture the post-event state observed by the next action.
        second = env.step(action)
        torch.testing.assert_close(second["s"], first["s1"])
    finally:
        native.close()
