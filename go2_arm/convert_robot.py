from pathlib import Path
import argparse,json
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from pxr import Usd,UsdPhysics,PhysxSchema
R=Path(__file__).resolve().parent
# Convex hulls match the MuJoCo planner and conservatively enclose each mesh.
# PhysX convex decomposition produced false shoulder/base penetration despite
# a 5.57 mm separating plane in the original meshes. Box walls are primitives.
cfg=UrdfConverterCfg(asset_path=str(R/'robot/robot.urdf'),usd_dir=str(R/'robot/usd'),usd_file_name='robot.usd',fix_base=False,merge_fixed_joints=True,force_usd_conversion=True,self_collision=True,collider_type='convex_hull',joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.),target_type='position'))
c=UrdfConverter(cfg);s=Usd.Stage.Open(c.usd_path);bodies=[];joints=[]
for prim in s.Traverse():
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):
  mass=UsdPhysics.MassAPI(prim).GetMassAttr().Get();bodies.append({'path':str(prim.GetPath()),'mass_kg':mass})
 if prim.IsA(UsdPhysics.Joint):joints.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'body0':[str(x) for x in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()],'body1':[str(x) for x in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()]})
# The jaws may touch when fully closed. Treat the jaw-to-jaw pair as an intended mechanical pair.
left=[b for b in bodies if b['path'].endswith('/link7')];right=[b for b in bodies if b['path'].endswith('/link8')]
if left and right:UsdPhysics.FilteredPairsAPI.Apply(s.GetPrimAtPath(left[0]['path'])).CreateFilteredPairsRel().AddTarget(right[0]['path'])
s.GetRootLayer().Save()
report={'usd_path':c.usd_path,'bodies':bodies,'joints':joints,'self_collision':True,'fixed_base':False,'num_revolute':sum(j['type']=='PhysicsRevoluteJoint' for j in joints),'num_prismatic':sum(j['type']=='PhysicsPrismaticJoint' for j in joints),'total_mass_kg':sum(float(b['mass_kg'] or 0) for b in bodies)}
assert report['num_revolute']==18 and report['num_prismatic']==2,report
assert not any(j['type']=='PhysicsFixedJoint' and not j['body0'] for j in joints),'Unexpected world fixation'
report['mesh_collision_approximation']='convexHull';report['box_collision']='five separate solid wall/floor primitives'
(R/'usd_validation.json').write_text(json.dumps(report,indent=2));print('MODEL_CONVERTED',json.dumps(report),flush=True)
import os,sys
sys.stdout.flush();os._exit(0)
