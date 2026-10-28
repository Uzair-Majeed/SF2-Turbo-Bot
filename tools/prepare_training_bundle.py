"""Build the small notebook upload ZIP using only Python's standard library.

Run from any directory: python tools/prepare_training_bundle.py
Re-run after adding recordings or editing training code. Nothing is uploaded.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_bundle(root=ROOT):
    root = Path(root).resolve()
    required = [
        'train_models/train_shared.py', 'PythonAPI/sf2_features.py',
        'tools/cloud_training.py', 'requirements-training.txt',
    ]
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    paths = [root / p for p in required]
    for relative in ['requirements.txt', 'DATA_SOURCES.md', 'TRAINING_START_HERE.md']:
        if (root / relative).is_file():
            paths.append(root / relative)
    for folder in ['recordings', 'normalized_character_datasets']:
        paths.extend(sorted((root / folder).glob('*.csv')))
    paths.extend(sorted((root / 'notebooks').glob('*.ipynb')))
    manifest = []
    destination = root / 'cloud_upload' / 'sf2_training_bundle.zip'
    destination.parent.mkdir(exist_ok=True)
    # Only this generated archive is replaced on a subsequent run.
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Expected a project-owned regular file: {path}')
            data = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            archive.writestr('sf2_training/' + relative, data)
            manifest.append({'path': relative, 'bytes': len(data),
                             'sha256': hashlib.sha256(data).hexdigest()})
        archive.writestr('sf2_training/upload_manifest.json', json.dumps(manifest, indent=2))
    print(f'Ready to upload: {destination}')
    print(f'{len(paths)} files; {destination.stat().st_size / 1024:.0f} KiB')
    print('Includes training sources and CSVs, not models, emulator, ROM, or match logs.')
    print('Only put permitted team recordings in recordings/. Hold out evaluation sessions elsewhere.')
    return destination


if __name__ == '__main__':
    build_bundle()
