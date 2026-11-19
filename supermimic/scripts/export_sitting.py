"""Export a sitting student with normalization inside ONNX."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from supermimic.distillation.sitting.models import load_sitting_student


def export(checkpoint_path, output):
    import onnx

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new ONNX output: {output}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = load_sitting_student(checkpoint)
    use_depth = policy.use_depth
    contract = dict(
        schema_version=1, task="g1_sitting", history_length=10, proprio_dim=93,
        goal_dim=730, action_dim=29, use_depth=use_depth,
        control_dt=.02, physics_dt=.005, history_order="oldest_to_newest",
        joint_names=G1_JOINT_NAMES, default_joint_pos=default_joint_pos_list(),
        action_scale=[action_scale(name) for name in G1_JOINT_NAMES],
    )
    if use_depth:
        from supermimic.camera import HDMI_CAMERA_METADATA
        contract.update(depth_shape=[1, 120, 160], depth_latent_dim=64, camera=HDMI_CAMERA_METADATA)
        sample = (torch.zeros(1, 10, 93), torch.zeros(1, 730), torch.ones(1, 1, 120, 160))
        input_names = ["proprio_history", "goal", "depth"]
    else:
        sample = (torch.zeros(1, 10, 93), torch.zeros(1, 730))
        input_names = ["proprio_history", "goal"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(policy, sample, str(output), dynamo=False, opset_version=17,
                          input_names=input_names, output_names=["actions"])
    model = onnx.load(output)
    onnx.helper.set_model_props(model, {"sitting_depth_contract": json.dumps(contract, sort_keys=True)})
    onnx.checker.check_model(model)
    onnx.save(model, output)
    inputs = "proprio_history[1,10,93], goal[1,730]"
    if use_depth:
        inputs += ", depth[1,1,120,160]"
    print(f"Saved {output}: {inputs} -> actions[1,29]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
