"""Derive a PhysX asset with separate foot spheres and preserved calf inertia."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from pxr import UsdShade

LEGS = ("FL", "FR", "RL", "RR")
FOOT_MASS = .02


def parallel_axis(mass, offset):
    return mass * (offset.dot(offset)*np.eye(3) - np.outer(offset, offset))


def mass_properties(prim):
    mass = UsdPhysics.MassAPI(prim)
    q = mass.GetPrincipalAxesAttr().Get()
    axes = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
    inertia = axes @ np.diag(np.asarray(mass.GetDiagonalInertiaAttr().Get())) @ axes.T
    return float(mass.GetMassAttr().Get()), np.asarray(mass.GetCenterOfMassAttr().Get()), inertia


def set_mass(prim, mass, center, inertia):
    moments, axes = np.linalg.eigh(inertia)
    if min(moments) <= 0:
        raise ValueError(f"Nonpositive inertia for {prim.GetPath()}")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    qx, qy, qz, qw = Rotation.from_matrix(axes).as_quat()
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
    api.CreatePrincipalAxesAttr(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))


def split(stage, root, leg):
    calf_path = root.AppendChild(f"{leg}_calf")
    calf = stage.GetPrimAtPath(calf_path)
    assert calf.HasAPI(UsdPhysics.RigidBodyAPI)
    collisions = stage.GetPrimAtPath(calf_path.AppendChild("collisions"))
    if collisions.IsInstance():
        collisions.SetInstanceable(False)
    candidates = [p for p in Usd.PrimRange(collisions, Usd.TraverseInstanceProxies())
                  if p.IsA(UsdGeom.Sphere) and p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one foot sphere in {leg}, found {len(candidates)}")
    old = candidates[0]
    if old.IsInstanceProxy():
        raise ValueError(f"Foot sphere is still instanced: {old.GetPath()}")
    cache = UsdGeom.XformCache()
    calf_tf = cache.GetLocalToWorldTransform(calf)
    sphere_tf = cache.GetLocalToWorldTransform(old)
    offset = np.asarray(calf_tf.GetInverse().Transform(sphere_tf.ExtractTranslation()))
    if not np.allclose(offset, (-.002, 0., -.213), atol=1e-5):
        raise ValueError(f"Unexpected sphere location on {leg}: {offset}")
    radius = UsdGeom.Sphere(old).GetRadiusAttr().Get()
    if not np.isclose(radius, .022):
        raise ValueError(f"Unexpected sphere radius on {leg}: {radius}")
    old_mass, old_center, old_inertia = mass_properties(calf)
    foot_inertia = np.eye(3) * (.4*FOOT_MASS*radius**2)
    remaining_mass = old_mass - FOOT_MASS
    remaining_center = (old_mass*old_center - FOOT_MASS*offset) / remaining_mass
    inertia_at_origin = old_inertia + parallel_axis(old_mass, old_center)
    remaining_inertia = (inertia_at_origin - foot_inertia - parallel_axis(FOOT_MASS, offset)
                         - parallel_axis(remaining_mass, remaining_center))

    foot_path = root.AppendChild(f"{leg}_foot_contact")
    foot = UsdGeom.Xform.Define(stage, foot_path)
    root_tf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))
    tf = Gf.Matrix4d(calf_tf)
    tf.SetTranslateOnly(sphere_tf.ExtractTranslation())
    foot.AddTransformOp().Set(tf * root_tf.GetInverse())
    UsdPhysics.RigidBodyAPI.Apply(foot.GetPrim())
    set_mass(foot.GetPrim(), FOOT_MASS, np.zeros(3), foot_inertia)
    set_mass(calf, remaining_mass, remaining_center, remaining_inertia)

    sphere = UsdGeom.Sphere.Define(stage, foot_path.AppendChild("sphere"))
    sphere.CreateRadiusAttr(radius)
    sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    old_physx = PhysxSchema.PhysxCollisionAPI(old)
    if old.HasAPI(PhysxSchema.PhysxCollisionAPI):
        new_physx = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
        for getter in ("ContactOffset", "RestOffset", "TorsionalPatchRadius", "MinTorsionalPatchRadius"):
            source = getattr(old_physx, f"Get{getter}Attr")()
            if source.HasAuthoredValueOpinion():
                getattr(new_physx, f"Create{getter}Attr")(source.Get())
    material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial("physics")
    if material:
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material, materialPurpose="physics")
    UsdPhysics.CollisionAPI(old).CreateCollisionEnabledAttr(False)

    joint = UsdPhysics.FixedJoint.Define(stage, root.AppendChild("joints").AppendChild(f"{leg}_foot_contact_joint"))
    joint.CreateBody0Rel().SetTargets([calf_path])
    joint.CreateBody1Rel().SetTargets([foot_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*offset))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr(False)

    calf_mass, calf_center, calf_inertia = mass_properties(calf)
    foot_mass, _, foot_inertia = mass_properties(foot.GetPrim())
    mass_error = abs(old_mass - calf_mass - foot_mass)
    moment_error = np.linalg.norm(old_mass*old_center - calf_mass*calf_center - foot_mass*offset)
    inertia_error = np.max(np.abs(inertia_at_origin - calf_inertia - parallel_axis(calf_mass, calf_center)
                                  - foot_inertia - parallel_axis(foot_mass, offset)))
    assert max(mass_error, moment_error, inertia_error) < 1e-7
    return {"foot_body": str(foot_path), "radius_m": radius, "mass_error_kg": mass_error,
            "first_moment_error_kg_m": moment_error, "inertia_error_kg_m2": inertia_error}


def main():
    src = args.usd.resolve(strict=True)
    target = src.with_name("robot_contact_split.usda")
    if target.exists():
        raise FileExistsError(target)
    original = Usd.Stage.Open(str(src))
    root = original.GetDefaultPrim().GetPath()
    pending = target.with_name(f".foot_split_{os.getpid()}.usda")
    try:
        layer = Usd.Stage.CreateNew(str(pending))
        layer.GetRootLayer().subLayerPaths = [src.name]
        layer.SetDefaultPrim(layer.GetPrimAtPath(root))
        report = {leg: split(layer, root, leg) for leg in LEGS}
        layer.GetRootLayer().Save()
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps(report), flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
