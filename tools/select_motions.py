"""Browser-based motion review, with a geometric back-to-chair proximity cue.

uv run python tools/select_motions.py
uv run python tools/select_motions.py data/motions/sitting --port 8086

Only forward kinematics is used. Selection never changes the source motions.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np
import viser

from motion_data import load_motion
from motion_data.back_proximity import BackProximity, summary
from motion_data.selection import SelectionStore


ROOT = Path(__file__).resolve().parents[1]
STATUS = {"全部": None, "未筛选": "unreviewed", "保留": "keep", "排除": "reject", "待定": "later"}
TAGS = {"未标注": [], "背部接触": ["back_contact"], "接触不确定": ["unclear"]}
SORTS = ("文件名", "背部贴近占比优先", "持续贴近优先")


def timeline_image(distances, threshold, frame):
    """A green/orange proximity strip; the white cursor marks the current frame."""
    indices = np.minimum(np.arange(600) * len(distances) // 600, len(distances) - 1)
    near = distances[indices] <= threshold
    image = np.empty((32, 600, 3), dtype=np.uint8)
    image[:] = np.where(near[:, None], [40, 183, 137], [93, 102, 119])
    cursor = min(599, int((frame + .5) * 600 / len(distances)))
    image[:, max(0, cursor - 1):min(600, cursor + 2)] = 255
    return image


class MotionSelector:
    """All scene/store mutations run on one loop; callbacks enqueue user actions."""

    def __init__(self, args):
        self.args = args
        self.store = SelectionStore(args.motion, args.selection)
        if not self.store.files:
            raise ValueError(f"No NPZ motions in {args.motion}")
        self.model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        self.data = mujoco.MjData(self.model)
        roots = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(roots) != 1:
            raise ValueError("Expected one floating robot root")
        self.root_adr = int(self.model.jnt_qposadr[roots[0]])
        self.proximity = BackProximity(self.model)
        self.events = queue.SimpleQueue()
        self.stop_scan = threading.Event()
        self.scan_thread = None
        self.current = None
        self.clip = None
        self.analysis = None
        self.frame = 0
        self.playhead = 0.0
        self.cache = OrderedDict()
        self.scores = {}
        self.scan_errors = {}
        self.filtered = list(self.store.files)
        self.labels = {}
        self.body_handles = {}
        self.chair = None
        self.last_time = time.monotonic()
        self.last_info_time = 0.0
        self._read_scores()
        self.server = viser.ViserServer(host=args.host, port=args.port,
            label=getattr(args, "viewer_label", "Sitting · 背部接触筛选"))
        self.server.gui.configure_theme(control_layout="fixed", control_width="large",
                                        dark_mode=True, show_logo=False, show_share_button=False,
                                        brand_color=(55, 186, 153))
        self.server.scene.set_up_direction("+z")
        self._build_scene()
        self._build_gui()
        async def connected(client):
            self.events.put(("camera", client))
        self.server.on_client_connect(connected)
        self.refresh_list()
        if self.current is None:
            self.load_clip(self.store.files[0])

    def _score_signature(self):
        digest = hashlib.sha256()
        # Includes named-joint mapping, axes and default pose as well as geometry.
        compiled = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=compiled)
        digest.update(compiled.tobytes())
        digest.update((ROOT / "motion_data/back_proximity.py").read_bytes())
        digest.update(str(self.store.source_dir).encode())
        return digest.hexdigest()

    def _read_scores(self):
        self.score_path = self.args.selection.with_name(self.args.selection.stem + "_proximity.json")
        self.score_signature = self._score_signature()
        try:
            saved = json.loads(self.score_path.read_text())
            if saved.get("signature") != self.score_signature:
                return
            for name, item in saved["scores"].items():
                if name not in self.store.files:
                    continue
                stat = (self.store.source_dir / name).stat()
                distances = np.asarray(item["distances"], dtype=float)
                if (item["stamp"] == [stat.st_size, stat.st_mtime_ns]
                        and distances.ndim == 1 and len(distances)
                        and np.isfinite(distances).all() and (distances >= 0).all()
                        and np.isfinite(item["fps"]) and item["fps"] > 0):
                    self.scores[name] = item
        except (OSError, ValueError, KeyError, TypeError):
            self.scores = {}

    def _save_scores(self):
        self.score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.score_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(signature=self.score_signature, scores=self.scores)))
        temporary.replace(self.score_path)

    def _add_score(self, name, distances, fps):
        stat = (self.store.source_dir / name).stat()
        self.scores[name] = dict(distances=np.asarray(distances).tolist(), fps=float(fps),
                                 stamp=[stat.st_size, stat.st_mtime_ns])

    def _build_scene(self):
        scene = self.server.scene
        scene.add_grid("/floor", width=12, height=12, cell_size=.25, section_size=1,
                       cell_color=(64, 75, 89), section_color=(88, 101, 119),
                       plane_color=(28, 36, 48), plane_opacity=.9)
        for gid in range(self.model.ngeom):
            if self.model.geom_group[gid] != 1 or self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            bid = int(self.model.geom_bodyid[gid])
            body_name = self.model.body(bid).name
            prefix = f"/robot/{body_name}"
            if bid not in self.body_handles:
                self.body_handles[bid] = scene.add_frame(prefix, show_axes=False)
            mesh = int(self.model.geom_dataid[gid])
            va, vn = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
            fa, fn = int(self.model.mesh_faceadr[mesh]), int(self.model.mesh_facenum[mesh])
            color = (101, 163, 211) if body_name == "torso_link" else tuple(
                (self.model.geom_rgba[gid, :3] * 255).astype(int))
            scene.add_mesh_simple(f"{prefix}/geom{gid}", self.model.mesh_vert[va:va + vn],
                                  self.model.mesh_face[fa:fa + fn], color=color,
                                  position=self.model.geom_pos[gid], wxyz=self.model.geom_quat[gid])
        self.back_patch = scene.add_point_cloud("/robot/torso_link/back_patch", self.proximity.back_points,
                                                colors=(38, 223, 231), point_size=.007)
        self.contact_line = scene.add_line_segments("/back_distance", np.zeros((1, 2, 3)),
                                                    colors=(250, 182, 70), thickness=.008)
        self.contact_points = scene.add_point_cloud("/back_points", np.zeros((2, 3)),
                                                    colors=(250, 182, 70), point_size=.024)

    def _input(self, handle, action):
        async def update(event):
            if event.client_id is not None:
                self.events.put((action, (self.current, event.target.value)))
        handle.on_update(update)
        return handle

    def _button(self, label, action, value=None, **kwargs):
        handle = self.server.gui.add_button(label, **kwargs)
        async def clicked(_):
            self.events.put((action, (self.current, value)))
        handle.on_click(clicked)
        return handle

    def _group(self, label, options, action):
        handle = self.server.gui.add_button_group(label, options)
        async def clicked(event):
            self.events.put((action, (self.current, event.target.value)))
        handle.on_click(clicked)
        return handle

    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# Sitting / 背部接触\n旋转视角检查靠背动作，标记后自动保存。")
        self.counts = gui.add_markdown("")
        with gui.add_folder("01 · 动作列表"):
            self.search = self._input(gui.add_text("搜索椅子 / 文件名", ""), "filter")
            self.status_filter = self._input(gui.add_dropdown("显示", tuple(STATUS)), "filter")
            self.sort = self._input(gui.add_dropdown("排序", SORTS), "filter")
            self.min_fraction = self._input(gui.add_slider("最少贴近占比 %", 0, 100, 5, 0,
                hint="大于 0 时只显示已分析且达到阈值的片段；可先扫描全部。"), "filter")
            self.picker = self._input(gui.add_dropdown("选择动作", ["加载中"]), "choose")
            self._group("切换", ["上一段", "下一段", "下一未筛选"], "navigate")
            self.file_info = gui.add_markdown("")
        with gui.add_folder("02 · 播放 / 视角"):
            self.playing = self._input(gui.add_checkbox("播放", True), "play")
            self.speed = gui.add_dropdown("速度", ["0.25×", "0.5×", "1×", "2×"], initial_value="1×")
            self.slider = self._input(gui.add_slider("帧", 0, 124, 1, 0), "frame")
            self._group("定位", ["前一帧", "后一帧", "最贴近帧", "重播"], "step")
            self._group("视角", ["斜后方", "侧面", "正面"], "view")
            self.opacity = self._input(gui.add_slider("椅子透明度", .15, 1., .05, .85), "opacity")
            self.show_back = self._input(gui.add_checkbox("显示背部采样点与距离线", True), "overlay")
        with gui.add_folder("03 · 背部贴近提示"):
            self.threshold = self._input(gui.add_slider("距离阈值 cm", .5, 10., .5, 3.), "threshold")
            self.proximity_info = gui.add_markdown("")
            self.timeline = gui.add_image(np.zeros((32, 600, 3), dtype=np.uint8), label="绿色：贴近 · 白线：当前帧")
            gui.add_markdown("青色点只取**躯干背侧**，不含臀部。距离是采样表面近似值，"
                             "不能判断支撑力或穿透；请旋转视角确认是否靠住椅背。")
            self.scan_button = self._button("扫描全部并排序", "scan", color="teal")
            self.cancel_button = self._button("停止扫描", "stop_scan", disabled=True)
            self.scan_progress = gui.add_progress_bar(0)
            self.scan_info = gui.add_markdown(f"已分析 {len(self.scores)} / {len(self.store.files)}")
        with gui.add_folder("04 · 筛选 / 自动保存"):
            self.auto_next = gui.add_checkbox("标记后自动下一段", True)
            self.tag = self._input(gui.add_dropdown("接触标注", tuple(TAGS)), "tag")
            self.notes = self._input(gui.add_text("备注", "", multiline=True), "notes")
            self._button("保留 ✓", "decision", "keep", color="teal")
            self._button("排除 ✕", "decision", "reject", color="red")
            self._button("待定", "decision", "later", color="yellow")
            self._button("撤销本段标记", "decision", "unreviewed")
            self.saved_info = gui.add_markdown("")
        with gui.add_folder("05 · 导出已保留动作", expand_by_default=False):
            self.export_mode = gui.add_dropdown("方式", ["软链接（节省空间）", "复制文件"])
            self._button("导出保留子集", "export", color="teal")
            gui.add_markdown(f"每次生成独立子目录，可直接用作训练 motion-dir。\n\n保存记录：`{self.args.selection}`")
        self.message = gui.add_markdown("就绪。鼠标左键旋转、滚轮缩放、右键平移。")

    def update_counts(self):
        counts = self.store.summary()
        self.counts.content = (f"**保留 {counts['keep']}** · 排除 {counts['reject']} · 待定 {counts['later']}"
                               f" · 未筛选 {counts['unreviewed']} / {counts['total']}")

    def _metrics(self, name):
        item = self.scores.get(name)
        return summary(np.asarray(item["distances"]), item["fps"], self.threshold.value / 100) if item else None

    def refresh_list(self, preferred=None):
        needle = self.search.value.strip().lower()
        status = STATUS[self.status_filter.value]
        names = [name for name in self.store.files if needle in name.lower()
                 and (status is None or self.store.get(name)["status"] == status)]
        metrics = {name: self._metrics(name) for name in names}
        if self.min_fraction.value:
            names = [name for name in names if metrics[name] and
                     metrics[name]["fraction"] >= self.min_fraction.value / 100]
        if self.sort.value != SORTS[0]:
            key = "fraction" if self.sort.value == SORTS[1] else "longest_seconds"
            names.sort(key=lambda name: (-(metrics[name][key] if metrics[name] else -1), name))
        self.filtered = names
        self.labels = {}
        for i, name in enumerate(names):
            metric = metrics[name]
            score = f"{metric['fraction']:.0%}" if metric else "未分析"
            label = f"{i + 1:04d} · {score} · {name.removesuffix('.npz').removeprefix('sitting__')}"
            self.labels[label] = name
        self.picker.options = tuple(self.labels) or ("无匹配动作",)
        self.picker.disabled = not names
        self.update_counts()
        selected = preferred if preferred in names else self.current if self.current in names else next(iter(names), None)
        if selected:
            self.picker.value = next(label for label, name in self.labels.items() if name == selected)
            if selected != self.current:
                self.load_clip(selected)
        else:
            # Keep the current preview for undoing a decision; make the empty filter explicit.
            self.message.content = "当前筛选条件没有匹配动作。下方仍显示上一段，可调整筛选条件。"

    def load_clip(self, name):
        self.message.content = "正在加载动作…"
        path = self.store.source_dir / name
        try:
            if name in self.cache:
                clip, analysis = self.cache.pop(name)
            else:
                clip = load_motion(path)
                analysis = self.proximity.analyze(clip)
            addresses = [int(self.model.joint(str(joint)).qposadr[0]) for joint in clip["joint_names"]]
        except Exception:
            # A failed choice must never label the previous preview as the new file.
            self._sync_picker()
            raise
        self.cache[name] = (clip, analysis)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        self.current, self.clip, self.analysis = name, clip, analysis
        self.addresses = addresses
        self._add_score(name, analysis["distances"], clip["fps"])
        self.scan_info.content = f"已分析 {len(self.scores)} / {len(self.store.files)}"
        self.scan_progress.value = len(self.scores) / len(self.store.files) * 100
        if self.chair:
            self.chair.remove()
        self.chair = self.server.scene.add_mesh_simple("/chair", clip["object_vertices"] * clip["object_scale"],
            clip["object_faces"], color=(184, 148, 102), opacity=self.opacity.value, side="double")
        self.slider.max = max(1, len(clip["root_pos"]) - 1)
        self.slider.disabled = len(clip["root_pos"]) == 1
        self.frame, self.playhead = 0, 0.0
        self.slider.value = 0
        self.last_time = time.monotonic()
        description = name.removesuffix(".npz").removeprefix("sitting__").replace("_", " ")
        self.file_info.content = (f"**{description}**\n\n{len(clip['root_pos'])} 帧 · {float(clip['fps']):g} Hz · "
                                  f"{(len(clip['root_pos']) - 1) / float(clip['fps']):.2f} 秒（首末帧）")
        self._show_record()
        self.render_frame()
        self.update_proximity()
        self._sync_picker()
        self.set_camera()
        self.message.content = "动作已加载。青色区域为背部；绿色距离线表示低于阈值。"

    def _sync_picker(self):
        if self.current is None:
            return
        labels = [label for label, name in self.labels.items() if name == self.current]
        metric = self._metrics(self.current)
        score = f"{metric['fraction']:.0%}" if metric else "未分析"
        prefix = labels[0].split(" · ")[0] if labels else "当前预览"
        label = f"{prefix} · {score} · {self.current.removesuffix('.npz').removeprefix('sitting__')}"
        if labels:
            self.labels = {label if item == self.current else key: item for key, item in self.labels.items()}
        else:
            self.labels[label] = self.current
        self.picker.options = tuple(self.labels)
        self.picker.value = label

    def _show_record(self, *, load_fields=True):
        record = self.store.get(self.current)
        if load_fields:
            self.notes.value = record["notes"]
            self.tag.value = next((key for key, value in TAGS.items() if value == record["tags"]), "未标注")
        status = next(key for key, value in STATUS.items() if value == record["status"])
        self.saved_info.content = f"当前：**{status}** · 修改即时保存"
        self.update_counts()

    def set_camera(self, client=None, view="斜后方"):
        if self.clip is None:
            return
        # Orient relative to the robot at its lowest root height (the sitting phase).
        index = int(np.argmin(self.clip["root_pos"][:, 2]))
        center = self.clip["root_pos"][index].copy()
        center[2] += .25
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.clip["root_quat"][index])
        direction = {"斜后方": [-2.4, -2.1, 1.0], "侧面": [0, -3.0, .45], "正面": [3., 0, .65]}[view]
        offset = matrix.reshape(3, 3) @ np.asarray(direction)
        offset[2] = direction[2]
        for target in ([client] if client else self.server.get_clients().values()):
            target.camera.fov = np.deg2rad(45)
            target.camera.up_direction = (0., 0., 1.)
            target.camera.position = center + offset
            target.camera.look_at = center

    def render_frame(self):
        clip, frame = self.clip, self.frame
        adr = self.root_adr
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[adr:adr + 3] = clip["root_pos"][frame]
        self.data.qpos[adr + 3:adr + 7] = clip["root_quat"][frame]
        self.data.qpos[self.addresses] = clip["joint_pos"][frame]
        mujoco.mj_kinematics(self.model, self.data)
        with self.server.atomic():
            for bid, handle in self.body_handles.items():
                handle.position = self.data.xpos[bid]
                handle.wxyz = self.data.xquat[bid]
            self.chair.position = clip["object_pos"][frame]
            self.chair.wxyz = clip["object_quat"][frame]
            pair = self.analysis["points_world"][frame]
            self.contact_line.points = pair[None]
            self.contact_points.points = pair
            near = self.analysis["distances"][frame] <= self.threshold.value / 100
            color = (40, 215, 150) if near else (250, 182, 70)
            self.contact_line.colors = np.asarray(color, dtype=np.uint8)
            self.contact_points.colors = np.asarray(color, dtype=np.uint8)
            self.slider.value = frame
        if time.monotonic() - self.last_info_time > .1 or not self.playing.value:
            self.update_proximity()

    def update_proximity(self):
        distances = self.analysis["distances"]
        threshold = self.threshold.value / 100
        metrics = summary(distances, float(self.clip["fps"]), threshold)
        self.proximity_info.content = (
            f"当前 **{distances[self.frame] * 100:.1f} cm** · {self.frame / float(self.clip['fps']):.2f} s\n\n"
            f"整段贴近 **{metrics['fraction']:.0%}** · 最长连续 **{metrics['longest_seconds']:.2f} s**"
            f" · 最小 {distances.min() * 100:.1f} cm")
        self.timeline.image = timeline_image(distances, threshold, self.frame)
        self.last_info_time = time.monotonic()

    def navigate(self, kind):
        names = self.filtered
        if not names:
            return
        index = names.index(self.current) if self.current in names else -1
        if kind == "下一未筛选":
            candidates = names[index + 1:] + names[:index + 1]
            name = next((name for name in candidates if self.store.get(name)["status"] == "unreviewed"), None)
            if name is None:
                self.message.content = "当前列表没有未筛选动作。"
                return
        else:
            name = names[(index + (-1 if kind == "上一段" else 1)) % len(names)]
        self.load_clip(name)
        self.picker.value = next(label for label, item in self.labels.items() if item == name)

    def start_scan(self):
        if self.scan_thread and self.scan_thread.is_alive():
            return
        pending = [name for name in self.store.files if name not in self.scores]
        self.stop_scan.clear()
        self.scan_button.disabled, self.cancel_button.disabled = True, False
        self.scan_errors = {}

        def scan():
            try:
                analyzer = BackProximity(self.model)
                for name in pending:
                    if self.stop_scan.is_set():
                        break
                    try:
                        clip = load_motion(self.store.source_dir / name)
                        result = analyzer.analyze(clip)
                        self.events.put(("score", (name, result["distances"], float(clip["fps"]))))
                    except Exception as exc:
                        self.events.put(("scan_error", (name, str(exc))))
            finally:
                self.events.put(("scan_done", None))
        self.scan_thread = threading.Thread(target=scan, daemon=True, name="back-proximity-scan")
        self.scan_thread.start()

    def process_event(self, action, payload):
        if action == "camera":
            self.set_camera(payload)
            return
        if action == "score":
            self._add_score(*payload)
            done = len(self.scores)
            self.scan_progress.value = done / len(self.store.files) * 100
            self.scan_info.content = f"已分析 {done} / {len(self.store.files)} · 失败 {len(self.scan_errors)}"
            if done % 50 == 0:
                self._save_scores()
            return
        if action == "scan_error":
            self.scan_errors[payload[0]] = payload[1]
            return
        if action == "scan_done":
            self.scan_button.disabled, self.cancel_button.disabled = False, True
            self._save_scores()
            self.sort.value = SORTS[1]
            self.refresh_list()
            self.scan_info.content = f"已分析 {len(self.scores)} / {len(self.store.files)} · 失败 {len(self.scan_errors)}"
            self.message.content = ("扫描已停止，可继续扫描。" if self.stop_scan.is_set() else "扫描完成，已按背部贴近占比排序。")
            if self.scan_errors:
                self.message.content += f" 首个失败：{next(iter(self.scan_errors.items()))}"
            return
        name, value = payload
        if action == "filter":
            self.refresh_list()
        elif action == "choose":
            if value in self.labels:
                self.load_clip(self.labels[value])
        elif action == "navigate":
            self.navigate(value)
        elif action == "view":
            self.set_camera(view=value)
        elif action == "opacity":
            self.chair.opacity = value
        elif action == "overlay":
            self.back_patch.visible = self.contact_line.visible = self.contact_points.visible = value
        elif action == "threshold":
            self.render_frame()
            self.update_proximity()
            self.refresh_list()
        elif action == "play":
            self.last_time = time.monotonic()
        elif action in ("frame", "step") and name == self.current:
            if action == "frame":
                self.frame = min(int(value), len(self.clip["root_pos"]) - 1)
            else:
                self.frame = {"前一帧": max(0, self.frame - 1),
                              "后一帧": min(len(self.clip["root_pos"]) - 1, self.frame + 1),
                              "最贴近帧": int(np.argmin(self.analysis["distances"])), "重播": 0}[value]
            self.playhead = float(self.frame)
            self.playing.value = value == "重播"
            self.render_frame()
        elif action in ("decision", "notes", "tag"):
            record = self.store.get(name)
            if action == "decision":
                self.store.set(name, value)
            elif action == "notes":
                self.store.set(name, record["status"], notes=value)
            else:
                self.store.set(name, record["status"], tags=TAGS[value])
            if name == self.current:
                self._show_record(load_fields=False)
                if action == "decision":
                    preferred = None
                    if self.auto_next.value and value != "unreviewed" and name in self.filtered:
                        index = self.filtered.index(name)
                        candidates = self.filtered[index + 1:] + self.filtered[:index]
                        preferred = next(iter(candidates), None)
                    self.refresh_list(preferred)
        elif action == "export":
            mode = "copy" if self.export_mode.value == "复制文件" else "symlink"
            exported = self.store.export(self.args.export_dir, mode=mode)
            self.message.content = f"已导出 **{self.store.summary()['keep']}** 段：`{exported}`"
        elif action == "scan":
            self.start_scan()
        elif action == "stop_scan":
            self.stop_scan.set()

    def tick(self):
        self._did_loop = False
        # Bound work per tick so a fast scanner cannot starve playback.
        for _ in range(20):
            try:
                action, payload = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                self.process_event(action, payload)
            except Exception as exc:
                self.message.content = f"操作失败：{exc}"
                print(f"{action}: {exc}", flush=True)
        now = time.monotonic()
        elapsed, self.last_time = now - self.last_time, now
        if self.clip is not None and self.playing.value:
            advanced = self.playhead + elapsed * float(self.clip["fps"]) * float(self.speed.value.rstrip("×"))
            self._did_loop = advanced >= len(self.clip["root_pos"])
            self.playhead = advanced % len(self.clip["root_pos"])
            frame = int(self.playhead)
            if frame != self.frame:
                self.frame = frame
                self.render_frame()

    def close(self):
        self.stop_scan.set()
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        self._save_scores()
        self.server.stop()

    def run(self):
        print(f"Review: http://{self.args.host}:{self.server.get_port()}\n"
              f"Selection: {self.args.selection}\nCtrl+C to stop.", flush=True)
        try:
            while True:
                self.tick()
                time.sleep(.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, nargs="?", default=ROOT / "data/motions/sitting")
    parser.add_argument("--model", type=Path, default=ROOT / "data/g1_description/g1_29dof.xml")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/selections/sitting_back.json")
    parser.add_argument("--export-dir", type=Path, default=ROOT / "data/motions/sitting_back_selected")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = MotionSelector(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot start motion selector: {exc}\n")
    app.run()


if __name__ == "__main__":
    main()
