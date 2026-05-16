"""A7: Pinocchio(cpin) → CasADi 심볼릭 export → dynamics/casadi_fns/go2_all_terms.casadi

codegen_env(numpy<2, robotpkg pinocchio, cuda_codegen casadi)에서 실행.
URDF 는 repo 상대경로 (절대경로 금지). 메시 불필요 → package_dirs 안 씀.

산출물 16항 / flat sizes (A9 검증 기준):
[324,18,18,108,6,9,3,3,54,3,9,9,12,36,432,24]
M,g,C_dq,Ag,hg,I_W,p_com,v_com,Jcom,base_p,R_WB,R_WH,r_feet,R_feet,J_feet_W,Jd_dq_feet_W

실행:
  conda run -n codegen_env python dynamics/export_pinocchio_casadi.py
"""

from pathlib import Path

# URDF: quadruped/ 패키지에 self-contained (mj_opt/models/go2 에서 1회 복사 필요)
URDF = Path(__file__).resolve().parent.parent / "quadruped" / "urdf" / "go2_description.urdf"
OUT = Path(__file__).resolve().parent / "casadi_fns" / "go2_all_terms.casadi"
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")  # base 컬럼순서 [FL,FR,RL,RR]


def main() -> None:
  import casadi as ca
  import pinocchio as pin
  from pinocchio import casadi as cpin

  assert URDF.exists(), f"URDF 없음: {URDF}  (mj_opt/models/go2/urdf 에서 복사)"

  model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())  # nq=19, nv=18
  cmodel = cpin.Model(model)
  cdata = cmodel.createData()
  base_fid = model.getFrameId("base")
  foot_fid = [model.getFrameId(n) for n in FEET]

  q = ca.SX.sym("q", model.nq)
  dq = ca.SX.sym("dq", model.nv)
  theta = ca.SX.sym("theta", 1)  # 예약(미사용) — cog.launch 시그니처 유지

  # TODO: cpin 심볼명/centroidal inertia 추출은 설치 pinocchio 버전에서 확인 (A6 역할).
  # TODO: 아래 16항 flat 크기를 위 sizes 와 정확히 일치시킬 것.
  raise NotImplementedError(
    "cpin 항 구현 필요. 스켈레톤은 Mythos.md / 세션 설계 노트 참조.\n"
    "  M=cpin.crba, g=cpin.computeGeneralizedGravity, Ag=cpin.computeCentroidalMap, ...\n"
    f"  → ca.Function('go2_all_terms',[q,dq,theta],[...16...]).expand().save('{OUT}')"
  )


if __name__ == "__main__":
  main()
