"""Check bundled runtime data and unchanged formal-training sources."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(load_model=False):
    asset = json.loads((ROOT / 'asset_manifest.json').read_text())
    provenance = json.loads((ROOT / 'provenance/training_source_snapshot.json').read_text())
    expected = dict(asset['files'])
    expected.update({'amp_reference/' + k: v for k, v in asset['amp_reference']['clips'].items()})
    recorded = json.loads((ROOT / 'go2_arm/amp_reference/recording_manifest.json').read_text())
    expected.update({'amp_reference/' + row['file']: row['sha256'] for row in recorded['clips']})
    expected.update(provenance['source_hashes'])
    expected.update(provenance['asset_hashes'])
    errors = []
    for name, value in expected.items():
        path = ROOT / 'go2_arm' / name
        if not path.is_file() or path.is_symlink() or digest(path) != value:
            errors.append(name)
    if errors:
        raise ValueError('Missing or changed runtime files: ' + ', '.join(errors))
    report = {'verified_files': len(expected), 'amp_reference_clips': len(asset['amp_reference']['clips']),
              'supplemental_clips': len(recorded['clips']), 'source_and_physics_hashes_match': True}
    if load_model:
        import mujoco
        import numpy as np
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'go2_arm/robot/scene.xml'))
        assert model.nq == 27 and model.nv == 26
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        mass = float(model.body_mass.sum())
        assert abs(mass - provenance['runtime_mass_kg']) < 1e-4, mass
        for name in ['path_bank.npz', 'whole_body_bank.npz']:
            with np.load(ROOT / 'go2_arm' / name) as data:
                for key in data.files:
                    value = data[key]
                    if np.issubdtype(value.dtype, np.number):
                        assert np.isfinite(value).all(), (name, key)
        report.update(mujoco_model_loaded=True, free_base=True, mass_kg=mass, nq=model.nq, nv=model.nv,
                      trajectory_arrays_finite=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-model', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.load_model), indent=2))
