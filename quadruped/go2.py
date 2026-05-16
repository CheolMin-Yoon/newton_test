"""Go2 mjlab EntityCfg (cartpole_constants.py 대응).

mjlab scene 은 MJCF(MuJoCo MjSpec) 사용 → xml/go2.xml.
Pinocchio 동역학 codegen 은 URDF 사용 → urdf/go2_description.urdf (dynamics/export 가 참조).
둘 다 mj_opt/models/go2 에서 1회 복사해 self-contained 화.

경로는 repo 상대 (절대경로·옛 main.py 의 /home/frlab/mj_opt/... 금지).
"""

from pathlib import Path

GO2_XML = Path(__file__).resolve().parent / "xml" / "go2.xml"
GO2_URDF = Path(__file__).resolve().parent / "urdf" / "go2_description.urdf"


def get_go2_robot_cfg():
  """Go2 EntityCfg 인스턴스 (mjlab scene 등록용).

  cartpole 의 get_cartpole_robot_cfg() 패턴: spec_fn + articulation(+init_state).
  """
  # from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
  # from mjlab.actuator import XmlActuatorCfg
  # import mujoco
  assert GO2_XML.exists(), f"MJCF 없음: {GO2_XML} (mj_opt/models/go2/xml 에서 복사)"
  raise NotImplementedError(
    "EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(str(GO2_XML)), "
    "articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(...),)), init_state=...)"
  )
