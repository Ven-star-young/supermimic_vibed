"""Replay manually kept motions by chair and choose one reference per chair.

uv run python -m tools.review_chairs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from motion_data.references import ChairReferences, chair_id
from motion_data.selection import SelectionStore
from tools.select_motions import MotionSelector, ROOT


CHAIR_FILTERS = ("待选的多候选椅子", "全部已保留椅子", "已确定 reference", "只有一个候选")


class ChairReviewer(MotionSelector):
    def __init__(self, args):
        args.viewer_label = "Sitting · 每椅一个 reference"
        self.current_chair = None
        self.chair_labels = {}
        self.visible_chairs = []
        self._preserve_camera = False
        self._restart_clip = False
        # Validate before allocating a server/scene.
        selection = SelectionStore(args.motion, args.selection)
        self.references = ChairReferences(selection, args.references)
        if not self.references.groups:
            raise ValueError("筛选记录中还没有标为保留的动作。")
        super().__init__(args)

    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# 按椅子挑选 reference\n只回放已保留的动作，每种椅子选一个。")
        self.counts = gui.add_markdown("")
        with gui.add_folder("01 · 椅子 / 候选动作"):
            self.search = self._input(gui.add_text("搜索椅子", ""), "filter")
            self.chair_filter = self._input(gui.add_dropdown("椅子范围", CHAIR_FILTERS), "filter")
            self.chair_picker = self._input(gui.add_dropdown("椅子", ["加载中"]), "chair")
            self._group("切换椅子", ["上一种椅子", "下一种椅子", "下一待选椅子"], "chair_nav")
            self.chair_info = gui.add_markdown("")
            self.picker = self._input(gui.add_dropdown("当前椅子的 motion", ["加载中"]), "choose")
            self._group("比较候选", ["上一段", "下一段"], "navigate")
            self.reference_info = gui.add_markdown("")
            self.pick_reference = self._button("选当前 motion 为这把椅子的 reference", "reference", color="teal")
            self.auto_next_chair = gui.add_checkbox("选定后转到下一待选椅子", True)
            self._button("撤销这把椅子的 reference", "clear_reference")
        with gui.add_folder("02 · 回放 / 对比"):
            self.playing = self._input(gui.add_checkbox("播放", True), "play")
            self.sequence = gui.add_checkbox("同椅子的候选依次回放", True,
                hint="播放完一段后，自动回放同一把椅子的下一段；不会自动跨椅子。")
            self.sync_time = gui.add_checkbox("手动切换候选时保持时间位置", True)
            self.keep_view = gui.add_checkbox("同椅子切换时保持相机视角", True)
            self.speed = gui.add_dropdown("速度", ["0.25×", "0.5×", "1×", "2×"], initial_value="1×")
            self.slider = self._input(gui.add_slider("帧", 0, 124, 1, 0), "frame")
            self._group("定位", ["前一帧", "后一帧", "最贴近帧", "重播"], "step")
            self._group("视角", ["斜后方", "侧面", "正面"], "view")
            self.opacity = self._input(gui.add_slider("椅子透明度", .15, 1., .05, .85), "opacity")
            self.show_back = self._input(gui.add_checkbox("显示背部提示", True), "overlay")
            self.file_info = gui.add_markdown("")
        with gui.add_folder("03 · 当前椅子的候选统计", expand_by_default=False):
            self.threshold = self._input(gui.add_slider("距离阈值 cm", .5, 10., .5, 3.), "threshold")
            self.candidate_table = gui.add_markdown("")
            self.proximity_info = gui.add_markdown("")
            self.timeline = gui.add_image(np.zeros((32, 600, 3), dtype=np.uint8), label="绿色：背部贴近")
            gui.add_markdown("贴近距离仅辅助判断；请观察是否真正靠背、有无穿透。")
        with gui.add_folder("04 · 保存 / 导出", expand_by_default=False):
            gui.add_markdown("只有一个候选的椅子自动采用该候选。多候选椅子手动选定，选择即时保存。")
            self._button("重新读取最新保留列表", "reload_selection")
            self.export_mode = gui.add_dropdown("导出方式", ["软链接（节省空间）", "复制文件"])
            self._button("导出每椅一个 reference", "export_references", color="teal")
            gui.add_markdown(f"reference 记录：`{self.args.references}`")
        # The shared renderer updates these; no full-dataset scan in chair review.
        self.scan_info = gui.add_markdown("", visible=False)
        self.scan_progress = gui.add_progress_bar(0, visible=False)
        self.message = gui.add_markdown("选择一把椅子，比较其已保留的候选动作。")

    def _save_scores(self):
        # The original selector may still be open and owns its score cache.
        pass

    def update_counts(self):
        stats = self.references.summary()
        self.counts.content = (f"**{stats['candidates']} 段候选 / {stats['chairs']} 种椅子**\n\n"
            f"已确定 **{stats['resolved']}**（含 {stats['single']} 种唯一候选） · 待选 **{stats['pending']}**")

    def refresh_list(self, preferred=None):
        needle = self.search.value.strip().lower().replace(" ", "_")
        mode = self.chair_filter.value
        groups = self.references.groups
        chairs = [chair for chair in groups if needle in chair.lower()]
        if mode == CHAIR_FILTERS[0]:
            chairs = [chair for chair in chairs if self.references.resolved(chair) is None]
        elif mode == CHAIR_FILTERS[2]:
            chairs = [chair for chair in chairs if self.references.resolved(chair) is not None]
        elif mode == CHAIR_FILTERS[3]:
            chairs = [chair for chair in chairs if len(groups[chair]) == 1]
        chairs.sort(key=lambda chair: (self.references.resolved(chair) is not None, -len(groups[chair]), chair))
        self.visible_chairs = chairs
        self.chair_labels = {f"{i+1:03d} · {len(groups[chair])} 段 · "
            f"{'已定' if self.references.resolved(chair) else '待选'} · {chair}": chair
            for i, chair in enumerate(chairs)}
        self.chair_picker.options = tuple(self.chair_labels) or ("当前范围没有椅子",)
        self.chair_picker.disabled = not chairs
        self.update_counts()
        if preferred in chairs:
            target = preferred
        else:
            target = self.current_chair if self.current_chair in chairs else next(iter(chairs), None)
        if target:
            self.load_chair(target)
        elif self.current is None:
            # A completed review still opens on a kept reference, never raw data.
            self.chair_filter.value = CHAIR_FILTERS[1]
            self.refresh_list()
        else:
            self.message.content = "当前范围没有椅子。可切换到“全部已保留椅子”；预览保留在上一把。"
            self._show_record()

    def load_chair(self, chair):
        previous = self.current_chair
        self.current_chair = chair
        self.filtered = list(self.references.groups[chair])
        self.labels = {self.candidate_label(name): name for name in self.filtered}
        self.picker.options = tuple(self.labels)
        target = self.current if self.current in self.filtered else self.references.resolved(chair) or self.filtered[0]
        try:
            if target != self.current:
                self.load_clip(target)
            else:
                self._sync_picker()
                self._show_record()
        except Exception:
            self.current_chair = previous
            if previous:
                self.load_chair(previous)
            raise
        if chair in self.chair_labels.values():
            self.chair_picker.value = next(label for label, key in self.chair_labels.items() if key == chair)

    def candidate_label(self, name):
        metric = self._metrics(name)
        score = f"贴近 {metric['fraction']:.0%}" if metric else "打开后计算贴近"
        suffix = " ★ reference" if self.references.resolved(chair_id(name)) == name else ""
        return f"motion {Path(name).stem.rsplit('__',1)[1]} · {score}{suffix}"

    def _sync_picker(self):
        if self.current is None:
            return
        self.labels = {self.candidate_label(name): name for name in self.filtered}
        if self.current not in self.filtered:
            return
        self.picker.options = tuple(self.labels)
        self.picker.value = next(label for label, name in self.labels.items() if name == self.current)

    def _show_record(self, **_):
        if self.current is None:
            return
        chair = chair_id(self.current)
        candidates = self.references.groups.get(chair, [])
        reference = self.references.resolved(chair)
        self.chair_info.content = f"**{chair.replace('_', ' ')}**\n\n这把椅子有 **{len(candidates)} 段**已保留候选。"
        index = candidates.index(self.current) + 1 if self.current in candidates else 0
        current_id = Path(self.current).stem.rsplit('__', 1)[1]
        selected_id = Path(reference).stem.rsplit('__', 1)[1] if reference else "尚未选择"
        self.reference_info.content = f"正在回放 **{index}/{len(candidates)} · motion {current_id}**\n\nReference：**{selected_id}**"
        self.pick_reference.disabled = self.current not in candidates
        rows = ["| motion | 贴近占比 | 连续贴近 | reference |", "|---|---:|---:|---|"]
        for name in candidates:
            metric = self._metrics(name)
            rows.append(f"| {Path(name).stem.rsplit('__',1)[1]} | " +
                (f"{metric['fraction']:.0%} | {metric['longest_seconds']:.2f}s" if metric else "— | —") +
                f" | {'★' if name == reference else ''} |")
        self.candidate_table.content = "\n".join(rows)
        self.update_counts()

    def load_clip(self, name):
        same_chair = self.current is not None and chair_id(self.current) == chair_id(name)
        seconds = self.frame / float(self.clip["fps"]) if self.clip is not None else 0
        self._preserve_camera = same_chair and self.keep_view.value
        try:
            super().load_clip(name)
        finally:
            self._preserve_camera = False
        if same_chair and self.sync_time.value and not self._restart_clip:
            self.frame = min(round(seconds * float(self.clip["fps"])), len(self.clip["root_pos"]) - 1)
            self.playhead = float(self.frame)
            self.render_frame()
            self.update_proximity()
        # Every candidate switch changes the prominent motion ID and progress.
        self._show_record()

    def set_camera(self, client=None, view="斜后方"):
        if not self._preserve_camera or client is not None:
            super().set_camera(client, view)

    def navigate_chair(self, kind):
        chairs = self.visible_chairs
        if kind == "下一待选椅子":
            chairs = [chair for chair in self.references.groups if self.references.resolved(chair) is None]
        if not chairs:
            self.message.content = "这些椅子都已确定 reference。"
            return
        index = chairs.index(self.current_chair) if self.current_chair in chairs else -1
        chair = chairs[(index + (-1 if kind == "上一种椅子" else 1)) % len(chairs)]
        if chair not in self.visible_chairs:
            self.search.value = ""
            self.chair_filter.value = CHAIR_FILTERS[0]
            self.refresh_list(chair)
        else:
            self.load_chair(chair)

    def process_event(self, action, payload):
        if action == "camera":
            return super().process_event(action, payload)
        name, value = payload
        if action == "chair":
            if value in self.chair_labels:
                self.load_chair(self.chair_labels[value])
        elif action == "chair_nav":
            self.navigate_chair(value)
        elif action in ("reference", "clear_reference"):
            if name != self.current:
                self.message.content = "回放已切换，请在当前 motion 上重新点击选择。"
                return
            chair = chair_id(name)
            if action == "reference":
                self.references.choose(name)
            else:
                self.references.clear(chair)
            self._sync_picker()
            self._show_record()
            # Update the visible chair's status without interrupting A/B review.
            self.chair_labels = {
                f"{i+1:03d} · {len(self.references.groups[key])} 段 · "
                f"{'已定' if self.references.resolved(key) else '待选'} · {key}": key
                for i, key in enumerate(self.visible_chairs)}
            self.chair_picker.options = tuple(self.chair_labels) or ("当前范围没有椅子",)
            if chair in self.chair_labels.values():
                self.chair_picker.value = next(label for label, key in self.chair_labels.items() if key == chair)
            if action == "reference" and self.auto_next_chair.value:
                self.refresh_list()
                if self.current_chair == chair:
                    self.navigate_chair("下一待选椅子")
            else:
                # Keep this preview for direct A/B replacement, even in the pending filter.
                self.message.content = "reference 已保存。可继续比较，或切换到下一把椅子。"
        elif action == "reload_selection":
            selection = SelectionStore(self.args.motion, self.args.selection)
            references = ChairReferences(selection, self.args.references)
            if not references.groups:
                raise ValueError("最新列表中没有保留动作。")
            self.store, self.references = selection, references
            self.refresh_list()
        elif action == "export_references":
            # Recheck current first-stage decisions before exporting.
            latest = ChairReferences(SelectionStore(self.args.motion, self.args.selection), self.args.references)
            mode = "copy" if self.export_mode.value == "复制文件" else "symlink"
            output = latest.export(self.args.export_dir, mode)
            self.message.content = f"已导出 {latest.summary()['chairs']} 个 reference：`{output}`"
        elif action in ("decision", "notes", "tag", "scan", "export"):
            raise ValueError("此界面只选择 reference。")
        else:
            super().process_event(action, payload)
            if action == "threshold":
                self._sync_picker()
                self._show_record()

    def tick(self):
        super().tick()
        if (self._did_loop and self.playing.value and self.sequence.value
                and len(self.filtered) > 1):
            self._restart_clip = True
            try:
                self.navigate("下一段")
            finally:
                self._restart_clip = False
            self.last_time = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=ROOT / "data/motions/sitting")
    parser.add_argument("--model", type=Path, default=ROOT / "data/g1_description/g1_29dof.xml")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/selections/sitting_back.json")
    parser.add_argument("--references", type=Path, default=ROOT / "data/selections/sitting_back_references.json")
    parser.add_argument("--export-dir", type=Path, default=ROOT / "data/motions/sitting_back_references")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = ChairReviewer(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot start chair review: {exc}\n")
    app.run()


if __name__ == "__main__":
    main()
