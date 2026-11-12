"""Validated stage-one model/optimizer restoration; simulation episodes restart.

This is not a bitwise continuation of the old PhysX trajectories. A new reward
revision starts new curriculum statistics and a 500-update observation window.
"""
import hashlib,json
from pathlib import Path
import torch
from stage1_schedule import STAGE1_LEVELS


def assert_state_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.shape==right.shape
        assert torch.equal(left.detach().cpu(),right.detach().cpu())
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:assert_state_equal(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_state_equal(a,b)
    else:assert left==right


def assert_finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():assert_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:assert_finite(v)


def restore_stage1(path,env,loco,arm,home,stage1_budget,allow_mount_rails=False,amp=None):
    path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['format']=='roboduet_dualmode_v1' and saved['training_stage']==1
    assert saved['arm_level']==-1,'Only stage-one restoration is supported'
    assert_finite(saved['loco']);assert_finite(saved['arm'])
    old=saved['manifest'];first=int(saved['iteration'])+1
    from leg_action import LEG_ACTION_CONTROL
    assert old.get('leg_action_controller')==LEG_ACTION_CONTROL,'Leg controller changed; use a fresh run'
    assert old.get('curriculum_revision')=='full_speed_before_attitude_v8','Stage-one schedule changed; use a fresh run'
    amp_resume={}
    if amp is not None:
        assert old.get('stage1_amp')==amp.metadata,'AMP features, reference data or settings differ'
        assert not allow_mount_rails,'AMP resume requires identical robot assets'
        asset_root=Path(env.cfg.robot.spawn.usd_path).parents[2]
        for name,digest in old['asset_hashes'].items():
            assert hashlib.sha256((asset_root/name).read_bytes()).hexdigest()==digest,name
        assert_finite(saved['amp'])
        amp.discriminator.load_state_dict(saved['amp']['discriminator'],strict=True)
        amp.optimizer.load_state_dict(saved['amp']['optimizer'])
        assert_state_equal(saved['amp'],amp.save())
        inherited=int(saved.get('amp_updates',first-int(old.get('first_training_iteration',0))))
        assert 0<inherited<=first,'Invalid AMP update counter'
        amp_resume={'amp_discriminator_optimizer_restored_exactly':True,'inherited_amp_updates':inherited}
    elif 'amp' in saved:
        raise ValueError('An AMP checkpoint requires --amp to restore its discriminator and optimizer')
    assert 0<first<stage1_budget
    assert old['home_consistency_verified'] and not old['is_fixed_base']
    assert torch.allclose(torch.tensor(saved['home_arm_q']),home.detach().cpu(),atol=1e-6)
    assert old['loco_obs_dim']==env.get_loco_observation().shape[-1] and old['arm_obs_dim']==env.get_arm_observation().shape[-1]
    assert old['loco_actions']==12 and old['arm_actions']==8
    assert old['arm_controller_omega_rad_s']==env.omega.detach().cpu().tolist()
    assert old['arm_controller_damping_ratio']==env.zeta
    urdf=Path(env.cfg.robot.spawn.usd_path).parents[1]/'robot.urdf'
    new_hash=hashlib.sha256(urdf.read_bytes()).hexdigest()
    mass=float(env.robot.root_physx_view.get_masses()[0].sum());expected_delta=.08 if allow_mount_rails else 0.
    assert abs(mass-old['full_mass_kg']-expected_delta)<1e-4,'Unexpected mass change'
    if allow_mount_rails:
        proof=json.loads(Path(__file__).with_name('model_revision_validation.json').read_text())
        assert proof['passed'] and proof['existing_links_joints_visuals_colliders_inertias_preserved']
        assert proof['parent_urdf_sha256']==old['urdf_sha256'] and proof['new_urdf_sha256']==new_hash
        assert proof['added_mount_mass_kg']==.08
    else:assert old['urdf_sha256']==new_hash
    level=int(saved['stage1_curriculum_level']);assert 0<=level<len(STAGE1_LEVELS)
    assert abs(saved['range_scale']-STAGE1_LEVELS[level][0])<1e-6
    assert abs(saved['guidance_cap_deg']-STAGE1_LEVELS[level][1])<1e-6
    env.training_iteration=first
    with torch.inference_mode():env.set_stage1_curriculum(level)
    for name,algorithm in [('loco',loco),('arm',arm)]:
        algorithm.load(saved[name],load_cfg=None,strict=True)
        algorithm.learning_rate=algorithm.optimizer.param_groups[0]['lr']
        assert_state_equal(saved[name],algorithm.save())
    env.success_ema=0.;env.precision_success_ema=0.;env.last_curriculum_change_iteration=first
    return {'first_training_iteration':first,'resumed_from_iteration':int(saved['iteration']),
        'resume_checkpoint':str(path.resolve()),'resume_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'parent_run':old['run_dir'],'from_scratch':False,'resume_revision':'stage1_models_optimizers_new_episodes_v1',
        'models_normalizers_optimizers_restored_exactly':True,'curriculum_statistics_reset':True,
        'physics_episode_state_restored':False,'bitwise_trajectory_resume':False,
        'model_revision_allowed':allow_mount_rails,'parent_urdf_sha256':old['urdf_sha256'],'urdf_sha256':new_hash,
        'parent_mass_kg':old['full_mass_kg'],'added_mount_mass_kg':expected_delta,
        'inherited_loco_updates':int(saved.get('loco_updates',first)),'inherited_arm_updates':int(saved.get('arm_updates',0)),**amp_resume}
