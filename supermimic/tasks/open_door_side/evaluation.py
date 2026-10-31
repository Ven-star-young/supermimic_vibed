"""Side-contact diagnostics reuse the native substep force collector."""

from copy import deepcopy

import torch

from ..open_door.evaluation import DoorContactMetrics
from .contacts import ALLOWED_BODIES, SIDE_CONTACT_SCHEMA, classify_contact_points


class SideDoorContactMetrics(DoorContactMetrics):
    allowed_rate_key = "allowed_side_substep_rate"

    def __init__(self, native):
        super().__init__(native)
        model = self.sim.mj_model
        bodies = torch.as_tensor(model.geom_bodyid.copy(), device=native.device)
        ids = bodies.new_tensor([model.body(f"robot/{name}").id for name in ALLOWED_BODIES])
        self.allowed_limb_geom = torch.isin(bodies, ids)

    def region_metadata(self):
        return {"side_contact_region": deepcopy(SIDE_CONTACT_SCHEMA)}

    def classify_contacts(self, local, robot_geom):
        return classify_contact_points(local, self.back_geom[robot_geom], self.allowed_limb_geom[robot_geom])
