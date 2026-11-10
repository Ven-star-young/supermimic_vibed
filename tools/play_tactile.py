"""Deploy a Sitting/OpenDoor policy with live G1 points and 112 tactile channels.

Run from the project root:
  .venv/bin/python tools/play_tactile.py --task sitting
  .venv/bin/python tools/play_tactile.py --task open_door
  .venv/bin/python tools/play_tactile.py --task open_door_side
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer

from motion_data.chair_split import load_split
from supermimic.rl.runner import SuperMimicRunner
from supermimic.scripts import play_policy
from supermimic.scripts.evaluate import _policy_config, _restore_open_door_config
from supermimic.tactile import G1TactileSensorCfg


ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "sitting": ("logs/cuhk_sitting_deployment/model_10000.pt", None, None),
    "open_door": ("logs/wm_transfer/back_model_10000.pt",
        "Mjlab-SuperMimic-OpenDoor-Unitree-G1",
        "data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz"),
    "open_door_side": ("logs/side_policy_playback/2026-11-01_01-14-42_fresh_ac_model10000/model_10000_actor.pt",
        "Mjlab-SuperMimic-OpenDoorSide-Unitree-G1",
        "data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz"),
}
ZONE_LABELS = {"chest": "Chest", "back": "Back", "left_arm": "Left arm",
               "right_arm": "Right arm", "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=tuple(PRESETS), default="sitting")
    parser.add_argument("--checkpoint", type=Path, help="Override the retained final policy")
    parser.add_argument("--reference", type=Path, help="One task-compatible motion NPZ")
    parser.add_argument("--split", type=Path, help="Sitting training split; defaults to the checkpoint's sibling")
    parser.add_argument("--data-root", type=Path, help="Relocated Sitting motion data")
    parser.add_argument("--partition", choices=("train", "test", "all"), default="all")
    parser.add_argument("--index", type=int, default=0, help="Sitting reference index within the sorted partition")
    parser.add_argument("--layout", choices=("original_outer", "inner_arms_v1"), default="original_outer")
    parser.add_argument("--counts-full-scale-mm", type=float, default=10.,
                        help="Effective depth giving 255 synthetic counts (uncalibrated)")
    parser.add_argument("--counts-exponent", type=float, default=1., help="Positive synthetic-count response exponent")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--paused", action="store_true", help="Open at the initial state; press Play to start")
    return parser


def load_deployment(args):
    """Reuse the existing deployment contracts; allocate only the frozen Actor."""
    response_cfg = G1TactileSensorCfg(name="tactile",
        counts_full_scale_depth_m=args.counts_full_scale_mm / 1000., counts_exponent=args.counts_exponent)
    if args.task != "sitting" and (args.split or args.data_root or args.partition != "all" or args.index):
        raise ValueError("--split/--data-root/--partition/--index are Sitting options; use --reference for a door motion")
    default_checkpoint, task_id, default_reference = PRESETS[args.task]
    checkpoint_path = (args.checkpoint or ROOT / default_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = None
    if args.task == "sitting":
        options = play_policy.build_parser().parse_args(["--checkpoint", str(checkpoint_path)])
        for key in ("split", "data_root", "partition", "index", "reference", "seed", "device"):
            setattr(options, key, getattr(args, key))
        manifest, root = load_split(options.split or checkpoint_path.parent / "split.json", options.data_root)
        paths = play_policy.resolve_references(options, manifest, root)
        if not 0 <= args.index < len(paths):
            raise ValueError(f"Reference index {args.index} out of range; found {len(paths)}")
        reference = paths[args.index]
        play_policy.reference_identity(checkpoint, manifest, reference, options)
        cfg = play_policy.make_deployment_cfg(checkpoint, reference, options)
        train_cfg = checkpoint["train_cfg"]
    else:
        reference = (args.reference or ROOT / default_reference).expanduser().resolve()
        cfg = load_env_cfg(task_id, play=True)
        _restore_open_door_config(cfg, checkpoint)
        command = cfg.commands["motion"]
        command.motion_dir = str(reference)
        command.motion_files = ()
        command.door_xml = str(reference.with_suffix(".xml"))
        for field in ("history_length", "goal_frames"):
            setattr(command, field, _policy_config(checkpoint)[field])
        train_cfg = asdict(load_rl_cfg(task_id))
    cfg.scene.num_envs, cfg.auto_reset, cfg.seed = 1, False, args.seed
    for sensor in cfg.scene.sensors:
        if sensor.name == "tactile":
            sensor.layout = args.layout
            sensor.counts_full_scale_depth_m = response_cfg.counts_full_scale_depth_m
            sensor.counts_exponent = response_cfg.counts_exponent
    native = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env = RslRlVecEnvWrapper(native)
        runner = SuperMimicRunner(env, train_cfg | {"amp": "none", "no_compile": True}, device=args.device)
        if manifest is not None:
            play_policy.load_actor(runner, checkpoint, manifest)
        else:
            runner.load(str(checkpoint_path), load_cfg={"actor": True}, map_location=args.device)
        runner.pi.eval().requires_grad_(False)
        return env, runner, checkpoint_path, reference
    except BaseException:
        native.close()
        raise


def tactile_colors(distance_mm, contact, valid, mode, limit_mm, counts):
    """One fixed display scale shared by point markers and channel cells."""
    if mode == "Counts (0-255)":
        strength = counts.astype(float) / 255.
    elif mode == "Contact":
        strength = contact.astype(float)
    else:
        strength = np.clip(1. - np.nan_to_num(distance_mm, nan=limit_mm, posinf=limit_mm) / limit_mm, 0., 1.)
    cold, hot = np.array([34, 65, 90]), np.array([255, 115, 45])
    colors = (cold + strength[..., None] * (hot - cold)).astype(np.uint8)
    colors[~valid] = (110, 110, 110)
    return colors


def display_channel_indices(zones):
    """Map body-view cells to the unchanged 112-vector (not packet indices).

    The saved ABI uses the unfolded garment's 206-first back order. A rear
    view is rotated 180 degrees: raw 58 starts the upper left (PDF page 17).
    """
    result, offset = {}, 0
    for zone in zones:
        count = zone["channel_count"]
        grid = np.arange(offset, offset + count).reshape(zone["display_grid_shape"])
        result[zone["name"]] = grid[::-1, ::-1] if zone["name"] == "back" else grid
        offset += count
    return result


def channel_image(distance_mm, contact, valid, zones, mode, limit_mm, counts, *,
                  show_raw_ids=False, selected_channel=None):
    """Draw anatomical views without reordering the sensor output or mapping."""
    image = Image.new("RGB", (448, 650), (17, 24, 35))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    title = ImageFont.load_default(size=15)
    caption = ImageFont.load_default(size=10)
    colors = tactile_colors(distance_mm, contact, valid, mode, limit_mm, counts)
    grids = display_channel_indices(zones)
    raw_indices = np.array([raw for zone in zones for raw in zone["raw_indices_1based"]])
    # Torso grids use full width; paired arms and shoulders sit side by side.
    origins = [(16, 38), (16, 258), (16, 450), (232, 450), (16, 548), (232, 548)]
    for zone, (x, y) in zip(zones, origins, strict=True):
        grid = grids[zone["name"]]
        rows, cols = grid.shape
        active = int(contact[grid].sum())
        draw.text((x, y - 30), f"{ZONE_LABELS[zone['name']]}  {active}/{grid.size}", font=title, fill=(225, 231, 240))
        orientation = ("FRONT: robot R -> L | top -> waist" if zone["name"] == "chest" else
                       "REAR: robot L -> R | top -> waist" if zone["name"] == "back" else
                       "neck -> arm" if "shoulder" in zone["name"] else
                       "shoulder -> hand")
        draw.text((x, y - 12), orientation, font=caption, fill=(170, 190, 210))
        for row in range(rows):
            for col in range(cols):
                i = int(grid[row, col])
                left, top = x + col * 52, y + row * 30
                outline = ((70, 225, 255) if i == selected_channel else
                           (255, 245, 210) if contact[i] and valid[i] else None)
                draw.rounded_rectangle((left, top, left + 48, top + 26), radius=3,
                    fill=tuple(colors[i]), outline=outline, width=2)
                if show_raw_ids:
                    value = f"#{raw_indices[i]}"
                elif not valid[i]:
                    value = "--"
                else:
                    value = (f"{distance_mm[i]:.1f}" if mode == "Distance (mm)" else
                             str(int(counts[i] if mode == "Counts (0-255)" else contact[i])))
                draw.text((left + 24, top + 13), value, font=font, fill="white", anchor="mm")
    draw.text((16, 609), "White: contact | Cyan: selected channel | Gray: invalid", font=font, fill=(190, 203, 218))
    legend = ("Labels: raw packet indices (1-based); colors: output" if show_raw_ids else
              "Synthetic counts / uncalibrated: 0 to 255" if mode == "Counts (0-255)" else
              f"Distance colors: 0 to {limit_mm} mm" if mode == "Distance (mm)" else
              "Contact: 0 = inactive, 1 = active")
    draw.text((16, 630), legend, font=font, fill=(190, 203, 218))
    return np.asarray(image)


class TactileViewer(ViserPlayViewer):
    def __init__(self, env, policy, server, checkpoint, reference, *, paused=False):
        super().__init__(env, policy, viser_server=server)
        self.cfg = replace(self.cfg, elevation=20., azimuth=120., distance=2.8)
        self.checkpoint, self.reference = checkpoint, reference
        self._is_paused, self._restart_at, self._episode = paused, None, 1
        abi = ROOT / "supermimic/assets/g1/tactile/upper-body-channel-layout.json"
        self.zones = json.loads(abi.read_text())["zones"]

    def setup(self):
        super().setup()
        gui = self._server.gui
        tactile_cfg = self.env.unwrapped.scene["tactile"].cfg
        # Keep tactile output above the stock tabs, visible alongside playback controls.
        with gui.add_folder("G1 Tactile", order=-1):
            self._summary = gui.add_markdown("")
            playback = gui.add_button_group("Playback", ("Play / Pause", "Step", "Reset"))

            @playback.on_click
            def control(event):
                {"Play / Pause": self.request_toggle_pause,
                 "Step": self.request_single_step, "Reset": self.request_reset}[event.target.value]()

            self._heatmap = gui.add_image(np.zeros((650, 448, 3), dtype=np.uint8), format="png")
            self._counts_description = gui.add_markdown("")
            with gui.add_folder("Channel inspection", expand_by_default=False):
                self._raw_ids = gui.add_checkbox("Show raw channel IDs", initial_value=False)
                layout = self.env.unwrapped.scene["tactile"].layout
                self._channel_options = ["None"] + [
                    f"{name.removeprefix('upper_body.')} | raw {raw + 1} | vector {i}"
                    for i, (name, raw) in enumerate(zip(layout.channel_names, layout.raw_indices, strict=True))]
                self._channel = gui.add_dropdown("Highlight channel", self._channel_options)
                self._channel_details = gui.add_markdown("Select a channel to highlight its member points in cyan.")
            with gui.add_folder("Display", expand_by_default=False):
                self._mode = gui.add_dropdown("Output", ("Counts (0-255)", "Distance (mm)", "Contact"))
                self._full_scale = gui.add_number("Counts full-scale depth (mm)",
                    initial_value=tactile_cfg.counts_full_scale_depth_m * 1000., min=0., step=.1,
                    hint="Smaller positive depths give higher synthetic counts.")
                self._exponent = gui.add_number("Counts exponent", initial_value=tactile_cfg.counts_exponent,
                    min=0., step=.1, hint="Below 1 increases low responses; above 1 reduces them.")
                self._limit = gui.add_slider("Color range (mm)", min=4., max=100., step=1., initial_value=20.)
                self._points = gui.add_checkbox("Show 464 points", initial_value=True)
                self._loop = gui.add_checkbox("Loop policy", initial_value=True)
                gui.add_markdown(f"**Policy:** {html.escape(self.checkpoint.name)}\n\n"
                                 f"**Reference:** {html.escape(self.reference.name)}\n\n"
                                 "Open surfaces use proximity only. Contact threshold: 4 mm.")
            for control in (self._mode, self._limit, self._points, self._full_scale, self._exponent,
                            self._raw_ids, self._channel):
                control.on_update(lambda _: self.request_action("CUSTOM", "tactile_redraw"))
        self._scene.request_update()

    def _handle_custom_action(self, action, payload):
        if payload == "tactile_redraw":
            self._scene.needs_update = True
            return True
        return super()._handle_custom_action(action, payload)

    def _execute_step(self):
        # Single-step at a terminal state starts the next episode cleanly.
        if self._restart_at is not None:
            self.reset_environment()
        success = super()._execute_step()
        if success:
            native = self.env.unwrapped
            command = native.command_manager.get_term("motion")
            if bool(native.sim.data.overflow.any()) or not bool(torch.isfinite(command.dynamics.pack(command.state())).all()):
                raise RuntimeError("Invalid physics state during policy playback")
            if bool(native.reset_buf[0]) or int(command.time_steps[0]) >= int(command.motion.clip_end[0]):
                self._restart_at = time.monotonic() + 1.0
                self._sim_budget = 0.0
                if not self._loop.value:
                    self.pause()
                    self._sync_ui_state()
        return success

    def _step_physics(self, dt):
        if self._restart_at is not None:
            self._sim_budget = 0.0
            if not self._loop.value:
                self.pause()
                self._sync_ui_state()
                return
            if time.monotonic() < self._restart_at:
                return
            self.reset_environment()
        super()._step_physics(dt)

    def reset_environment(self):
        self._restart_at = None
        super().reset_environment()
        self._episode += 1
        self._scene.request_update()

    def resume(self):
        if self._restart_at is not None and not self._loop.value:
            self.reset_environment()
        super().resume()

    def _handle_gui_reset(self, all_envs=False):
        self.reset_environment()

    def _submit_scene_update_if_needed(self, sim, has_pending_updates):
        if not self._should_submit_scene_update(self._counter, self._is_paused, has_pending_updates):
            return
        # Keep the robot, skin markers and heatmap in one frame. Rendering is
        # synchronous here, so the background viewer cannot sample a later pose.
        with self._sim_lock, self._server.atomic():
            self._update_tactile()
            self._scene.update(sim.data)
            self._server.flush()
        self._pending_update_reasons.clear()
        self._scene.needs_update = False

    def _update_tactile(self):
        native = self.env.unwrapped
        sensor = native.scene["tactile"]
        # UI callbacks only request a redraw. Apply mapping edits on this main thread.
        full_scale_mm, exponent = self._full_scale.value, self._exponent.value
        valid_mapping = True
        try:
            response_cfg = replace(sensor.cfg, counts_full_scale_depth_m=full_scale_mm / 1000.,
                                   counts_exponent=exponent)
        except ValueError:
            valid_mapping = False
            self._full_scale.value = full_scale_mm = sensor.cfg.counts_full_scale_depth_m * 1000.
            self._exponent.value = exponent = sensor.cfg.counts_exponent
        else:
            sensor.cfg.counts_full_scale_depth_m = response_cfg.counts_full_scale_depth_m
            sensor.cfg.counts_exponent = response_cfg.counts_exponent
        self._counts_description.content = (
            f"**Synthetic counts / uncalibrated** · Full-scale depth **{full_scale_mm:g} mm** · "
            f"Exponent **{exponent:g}**\n\n"
            "Smaller full-scale depth raises sensitivity. Color scale stays at 0–255."
            + ("\n\nMapping parameters are outside the supported positive range; previous values restored." if not valid_mapping else ""))
        sensor.update(0.)
        reading = sensor.data
        fields = ("point_positions_w", "point_normals_w", "surface_distance_m", "geometric_contact", "valid",
                  "channel_surface_distance_m", "channel_geometric_contact", "channel_valid",
                  "point_counts", "channel_counts")
        values = {name: getattr(reading, name)[0].detach().cpu().numpy().copy() for name in fields}
        channel_mm = values["channel_surface_distance_m"] * 1000
        mode, limit = self._mode.value, self._limit.value
        self._limit.visible = mode == "Distance (mm)"
        selected = self._channel_options.index(self._channel.value) - 1
        selected_points = np.zeros(len(sensor.layout.point_ids), dtype=bool)
        if selected >= 0:
            members = sensor.layout.channel_members[selected]
            selected_points[members] = True
            links = ", ".join(dict.fromkeys(sensor.layout.link_names[i] for i in members))
            self._channel_details.content = (
                f"**{sensor.layout.channel_names[selected]}** · raw **{sensor.layout.raw_indices[selected] + 1}** "
                f"(1-based) · vector **{selected}** (0-based)\n\n"
                f"**{len(members)} member points** on {links}\n\n"
                f"Counts **{int(values['channel_counts'][selected])}** · "
                f"Distance **{channel_mm[selected]:.2f} mm** · Valid **{bool(values['channel_valid'][selected])}**")
        else:
            self._channel_details.content = "Select a channel to highlight its member points in cyan."
        self._heatmap.image = channel_image(channel_mm, values["channel_geometric_contact"],
            values["channel_valid"], self.zones, mode, limit, values["channel_counts"],
            show_raw_ids=self._raw_ids.value, selected_channel=selected if selected >= 0 else None)
        command = native.command_manager.get_term("motion")
        frame = int(command.time_steps[0] - command.motion.clip_start[0])
        self._summary.content = (f"**Episode {self._episode} · Frame {frame}/{int(command.motion.clip_len[0]) - 1} · "
            f"{frame * native.step_dt:.2f} s**\n\n"
            f"Geometry contact: points **{values['geometric_contact'].sum()}/464** · "
            f"channels **{values['channel_geometric_contact'].sum()}/112**\n\n"
            f"Nonzero channels **{np.count_nonzero((values['channel_counts'] > 0) & values['channel_valid'])}/112** · "
            f"Peak **{int(values['channel_counts'].max())}/255**")
        if self._points.value or selected >= 0:
            colors = tactile_colors(values["surface_distance_m"] * 1000,
                values["geometric_contact"], values["valid"], mode, limit, values["point_counts"]) / 255.
            # Display-only lift avoids z-fighting; measured points stay unchanged.
            positions = values["point_positions_w"] + .001 * values["point_normals_w"]
            for i, (point, color) in enumerate(zip(positions, colors, strict=True)):
                if selected_points[i]:
                    color = np.array([70, 225, 255]) / 255.
                elif not self._points.value:
                    continue
                if np.isfinite(point).all():
                    self._scene.add_sphere(point, .006 if selected_points[i] else .004, (*color, 1.))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535 or args.index < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid port, reference index or seed")
    env, runner, checkpoint, reference = load_deployment(args)
    server = None
    before = play_policy.policy_digest(runner.pi)
    try:
        server = viser.ViserServer(host="127.0.0.1", port=args.port, label="G1 Policy + Tactile")
        viewer = TactileViewer(env, runner.get_inference_policy(), server, checkpoint, reference, paused=args.paused)
        print(f"[{args.task}] {checkpoint}\nReference: {reference}\nOpen http://127.0.0.1:{server.get_port()}", flush=True)
        viewer.run()
        if viewer._last_error:
            raise RuntimeError(viewer._last_error)
        if play_policy.policy_digest(runner.pi) != before:
            raise RuntimeError("Policy parameters changed during deployment")
        assert runner.wm is runner.critic is runner.buffer is None
    finally:
        if server is not None:
            server.stop()
        env.close()


if __name__ == "__main__":
    main()
