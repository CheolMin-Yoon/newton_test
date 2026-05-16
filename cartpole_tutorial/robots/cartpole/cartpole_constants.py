"""CartPole robot configuration (faithful copy of built-in mjlab Balance task)."""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# XML is resolved relative to this file, not the installed mjlab package.
CARTPOLE_XML: Path = Path(__file__).parent / "xmls" / "cartpole.xml"
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)

# Balance task: pole starts upright (hinge_1 = 0).
_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)


def get_cartpole_robot_cfg() -> EntityCfg:
  """Get a fresh CartPole (Balance) robot configuration instance."""
  return EntityCfg(
    spec_fn=get_spec,
    articulation=_CARTPOLE_ARTICULATION,
    init_state=_BALANCE_INIT,
  )
