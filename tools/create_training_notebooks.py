"""Regenerate the two checked-in notebooks; no third-party packages required."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, index):
    result = {'cell_type': kind, 'id': f'cell-{index:02d}', 'metadata': {},
              'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def create(smoke):
    cells = []

    def md(s):
        cells.append(cell('markdown', s, len(cells)))

    def code(s):
        cells.append(cell('code', s, len(cells)))

    title = '01 — First training check (supplied Ryu data)' if smoke else '02 — All characters + main-character specialist'
    md(f'''# {title}

Run cells from top to bottom by clicking each play button. Wait for a cell to finish before continuing.
If a cell turns red, stop and read its last error; do not skip ahead.

{'Optional pipeline check: no new recordings needed. This trains for only two epochs and does NOT produce a competition-ready fighter.' if smoke else 'Full training: use this notebook after recording your permitted demonstrations while controlling all 12 characters and rebuilding the upload ZIP. Notebook 01 is an optional two-epoch pipeline check, not a prerequisite.'}

Use only organizer-supplied data and your own team's recordings. Do not find or download internet gameplay datasets.
Cloud training uses the supplied scripts; the Windows emulator and keyboard recording stay on your PC.

**What you need:** `cloud_upload/sf2_training_bundle.zip`, built locally with `python tools/prepare_training_bundle.py`.
This is a small training-only ZIP, not the original participant archive.

**Colab:** upload this notebook using File → Upload notebook. Choose Runtime → Change runtime type → Python 3 and {'CPU (GPU is optional)' if smoke else 'an available GPU'}.
Upload the training ZIP when cell 2 asks. Save a copy of the notebook in your account.

**Kaggle:** import this notebook, attach the training ZIP as your own private input dataset, enable Internet for package installation, and choose {'CPU or GPU' if smoke else 'an available GPU'} in Settings.
Kaggle may unpack the ZIP automatically; both forms are supported. Do not publish your team data.

**Local Jupyter:** set `INPUT_PATH` below to the ZIP or the project folder. Python 3.12/3.13 is required for this pinned environment.

Cloud runtime files are temporary. The notebook can save Colab results to Drive; on Kaggle save a notebook version with outputs and download the result ZIP.
See [Colab FAQ](https://research.google.com/colaboratory/faq.html) and [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).
''')
    md('## 1. Settings\nFor your first run leave these values unchanged. Character `0` is Ryu; `4` is Ken. The final notebook lists all IDs in its data check.')
    code(f'''
    INPUT_PATH = ""  # Colab: leave blank for upload. Kaggle: auto-detect, or paste the input folder/ZIP path.
    MAIN_CHARACTER = {0 if smoke else 6}
    MAIN_WEIGHT = 1.0  # Keep neutral; the main already has substantially more recordings.
    USE_GPU = {not smoke}
    SAVE_TO_GOOGLE_DRIVE = {not smoke}  # Only applies on Colab; asks you to authorize Drive access.
    INCLUDE_KIT_DATA = False  # Notebook 01 always uses only the starter Ryu CSV.
    EPOCHS = {2 if smoke else 40}
    BATCH_SIZE = {128 if smoke else 256}
    SMOKE_TEST = {smoke}  # Use notebook 02 for all-character training; do not change this switch.
    ''')
    md('## 2. Load the training ZIP\nOn Colab, choose `sf2_training_bundle.zip` when the file picker appears. A fresh working folder is created on each run. Nothing is uploaded by the local packaging tool.')
    code('''
    from pathlib import Path, PurePosixPath
    import importlib.util
    import json
    import shutil
    import stat
    import sys
    import tempfile
    import zipfile

    ON_KAGGLE = Path('/kaggle/working').exists()
    ON_COLAB = not ON_KAGGLE and Path('/content').exists() and importlib.util.find_spec('google.colab') is not None
    BASE = Path('/kaggle/working') if ON_KAGGLE else (Path('/content') if ON_COLAB else Path.cwd())
    WORK = Path(tempfile.mkdtemp(prefix='sf2_training_', dir=BASE))
    def find_kaggle_sources():
        input_root = Path('/kaggle/input')
        archives = sorted(input_root.rglob('sf2_training_bundle.zip'))
        if archives:
            return archives
        return sorted({p.parent.parent for p in input_root.rglob('train_shared.py')
                       if p.parent.name == 'train_models'})

    if INPUT_PATH.strip():
        source = Path(INPUT_PATH).expanduser().resolve()
        if not source.exists() and ON_KAGGLE:
            choices = find_kaggle_sources()
            if len(choices) == 1:
                source = choices[0]
                print('Configured INPUT_PATH was not found; using the attached Kaggle input:', source)
            else:
                raise FileNotFoundError(
                    f'INPUT_PATH does not exist: {INPUT_PATH}. '
                    'Set INPUT_PATH to the mounted input path shown below, or attach exactly one training bundle. '
                    f'Found candidates: {choices}')
    elif ON_COLAB:
        from google.colab import files
        uploaded = files.upload()
        choices = [name for name in uploaded if name.lower().endswith('.zip')]
        if len(choices) != 1:
            raise ValueError('Upload exactly one sf2_training_bundle.zip, then rerun this cell.')
        source = WORK / 'input.zip'
        source.write_bytes(uploaded[choices[0]])
        del uploaded
    elif ON_KAGGLE:
        choices = find_kaggle_sources()
        if len(choices) != 1:
            raise ValueError('Attach one training bundle as input, or set INPUT_PATH to its exact ZIP/project path.')
        source = choices[0]
    else:
        raise ValueError('For local Jupyter, set INPUT_PATH to your training ZIP or project directory.')

    destination = WORK / 'project'
    destination.mkdir()
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                name = entry.filename.replace('\\\\', '/')
                parts = PurePosixPath(name).parts
                target = (destination / name).resolve()
                if (not target.is_relative_to(destination.resolve()) or ':' in name
                        or '..' in parts or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError(f'Unsafe archive member: {name}')
            archive.extractall(destination)
        candidates = [p.parent.parent for p in destination.rglob('train_shared.py')
                      if p.parent.name == 'train_models']
        if len(candidates) != 1:
            raise ValueError('Expected the training-only ZIP made by prepare_training_bundle.py.')
        ROOT = candidates[0]
    elif source.is_dir():
        # Only copy the training inputs, even if the source is the entire local project.
        if not (source / 'train_models' / 'train_shared.py').is_file():
            candidates = sorted({p.parent.parent for p in source.rglob('train_shared.py')
                                 if p.parent.name == 'train_models'})
            if len(candidates) != 1:
                raise ValueError(f'Could not find exactly one training project inside {source}; '
                                 f'found: {candidates}')
            source = candidates[0]
        ROOT = destination / 'sf2_training'
        ROOT.mkdir()
        for folder in ['train_models', 'PythonAPI', 'tools', 'recordings', 'normalized_character_datasets']:
            for path in (source / folder).glob('*'):
                if path.is_file() and path.suffix in {'.py', '.csv'}:
                    (ROOT / folder).mkdir(exist_ok=True)
                    shutil.copy2(path, ROOT / folder / path.name)
        for name in ['requirements-training.txt', 'DATA_SOURCES.md', 'upload_manifest.json']:
            if (source / name).exists():
                shutil.copy2(source / name, ROOT / name)
    else:
        raise ValueError(f'Input does not exist or is not a ZIP/project directory: {source}')
    for relative in ['train_models/train_shared.py', 'PythonAPI/sf2_features.py',
                     'tools/cloud_training.py', 'requirements-training.txt']:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f'Missing {relative}. Rebuild your training bundle.')
    print('Training project:', ROOT)
    print('No emulator or ROM is needed for training.')
    ''')
    md('## 3. Install an isolated training environment\nThis may take several minutes. Kaggle\'s Python can create virtual environments but may fail while bootstrapping pip, so this cell creates the environment without pip and uses the notebook host\'s pip to install packages into it. The notebook host\'s packages are not replaced. GPU mode also installs TensorFlow\'s CUDA dependencies. If no GPU is available, training can use CPU; it may take longer. A failed package install or TensorFlow import must be resolved before continuing.')
    code('''
    import subprocess
    import venv
    if sys.version_info[:2] not in [(3, 12), (3, 13)]:
        raise RuntimeError(f'This pinned setup targets Python 3.12/3.13; this runtime is {sys.version.split()[0]}. '
                           'Choose a compatible runtime instead of removing package pins blindly.')
    ENV = WORK / 'training_env'
    venv.EnvBuilder(with_pip=False).create(ENV)
    TRAIN_PY = ENV / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    install = [sys.executable, '-m', 'pip', '--python', str(TRAIN_PY),
               'install', '--disable-pip-version-check', '-r', str(ROOT / 'requirements-training.txt')]
    if USE_GPU and sys.platform.startswith('linux'):
        install.append('tensorflow[and-cuda]==2.21.0')
    subprocess.run(install, check=True)
    subprocess.run([str(TRAIN_PY), '-c',
                    "import tensorflow as tf; print('TensorFlow:', tf.__version__); "
                    "print('GPUs:', tf.config.list_physical_devices('GPU')); "
                    "print('Tensor check:', tf.reduce_sum(tf.ones([2, 2])))"], check=True)
    ''')
    md('## 4. Check the dataset before spending time training\nThe table counts the character YOU controlled, not opponents. Notebook 02 requires at least 2,000 eligible raw observations per character as a basic preflight check. This is not enough to guarantee skill, and is not an organizer rule. A second check after training requires at least 1,500 training samples per character. Aim for varied complete sessions, not just the smallest count.')
    code('''
    spec = importlib.util.spec_from_file_location('cloud_training', ROOT / 'tools/cloud_training.py')
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    if MAIN_CHARACTER not in range(12):
        raise ValueError('MAIN_CHARACTER must be 0 through 11.')
    AUDIT = workflow.audit_data(ROOT, smoke=SMOKE_TEST, include_kit=INCLUDE_KIT_DATA)
    workflow.require_coverage(AUDIT, smoke=SMOKE_TEST)
    if SMOKE_TEST:
        print('RYU-ONLY PIPELINE CHECK. Other characters have not been demonstrated.')
    else:
        print('Basic raw-data coverage passed. Actual training-window coverage is checked after training.')
    ''')
    md('## 5. Choose output storage and save the run information\nColab notebook 02 defaults to Drive so completed files survive runtime deletion. If you decline Drive access, set SAVE_TO_GOOGLE_DRIVE=False and download outputs before disconnecting. Kaggle writes to `/kaggle/working`; save a version with outputs. Each run uses a new folder, so previous models are preserved. This trainer does not resume interrupted training; `_best_tmp.keras` alone is not a complete model/scaler bundle.')
    code('''
    from datetime import datetime, timezone
    import uuid
    if ON_COLAB and SAVE_TO_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        RESULT_BASE = Path('/content/drive/MyDrive/SF2_training_runs')
    else:
        RESULT_BASE = BASE / 'sf2_results'
    label = 'smoke_ryu' if SMOKE_TEST else f'all_characters_main_{MAIN_CHARACTER}'
    RUN_DIR = RESULT_BASE / (label + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
                             + '_' + uuid.uuid4().hex[:6])
    MODEL_DIR = RUN_DIR / 'models'
    MODEL_DIR.mkdir(parents=True)
    (RUN_DIR / 'data_manifest.json').write_text(json.dumps(AUDIT, indent=2))
    workflow.snapshot_sources(ROOT, RUN_DIR)
    versions_code = ("import importlib.metadata as m, json, sys; "
                     "print(json.dumps({'python': sys.version, 'packages': "
                     "{p:m.version(p) for p in ['tensorflow','keras','numpy','pandas',"
                     "'scikit-learn','joblib','h5py']}}, indent=2))")
    environment = json.loads(subprocess.check_output([str(TRAIN_PY), '-c', versions_code], text=True))
    (RUN_DIR / 'environment.json').write_text(json.dumps(environment, indent=2))
    (RUN_DIR / 'requirements-local.txt').write_text(
        '\\n'.join(f'{p}=={v}' for p, v in environment['packages'].items()) + '\\n')
    print('Outputs:', RUN_DIR)
    print('Use requirements-local.txt for local inference; the runtime package freeze is intentionally omitted.')
    ''')
    md('## 6. Train\nKeep this cell running until it prints `Done`. The first notebook runs only 2 epochs. Notebook 02 trains a shared model and then a main-character specialist, using the same preprocessing. No separate `preprocess_windows.py` command is needed.\n\n**Evaluation limitation:** the current trainer splits within continuous sequences and neighboring windows may overlap. Its validation loss/attack metrics are exploratory, not independent fight results. Keep complete held-out sessions outside `recordings/` and test actual matches. Training changes to address the split are still needed before relying on validation for final model selection.')
    code('''
    if EPOCHS < 1 or BATCH_SIZE < 1 or MAIN_WEIGHT <= 0:
        raise ValueError('Epochs, batch size, and main weight must be positive.')
    command = [str(TRAIN_PY), '-u', str(ROOT / 'train_models/train_shared.py'),
               '--main-char', str(MAIN_CHARACTER), '--main-weight', str(MAIN_WEIGHT),
               '--epochs', str(EPOCHS), '--batch', str(BATCH_SIZE), '--out', str(MODEL_DIR),
               '--data']
    command.extend(str(ROOT / item['path']) for item in AUDIT['files'])
    if not SMOKE_TEST:
        command += ['--finetune', str(MAIN_CHARACTER)]
    (RUN_DIR / 'run_config.json').write_text(json.dumps({
        'smoke_test': SMOKE_TEST, 'main_character': MAIN_CHARACTER, 'main_weight': MAIN_WEIGHT,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'command': command,
        'competition_ready': False,
        'validation_caveat': 'Current trainer uses within-sequence splits; evaluate independent fights.'
    }, indent=2))
    workflow.run_logged(command, ROOT, RUN_DIR / 'training.log')
    ''')
    md('## 7. Verify saved files and inspect results\nThis reloads each model and scaler and checks finite outputs for all 12 character-token inputs. It does not test emulator compatibility, Player 2 gameplay, win rate, or rule compliance. The first notebook intentionally reports missing trained characters.')
    code('''
    verify = [str(TRAIN_PY), str(ROOT / 'tools/cloud_training.py'), '--verify', str(MODEL_DIR),
              '--main-char', str(MAIN_CHARACTER)]
    if not SMOKE_TEST:
        verify.append('--all-characters')
    subprocess.run(verify, check=True)
    report = json.loads((MODEL_DIR / 'training_report.json').read_text())
    print(json.dumps(report, indent=2))
    print('Attack precision/recall measure attack-vs-no-attack labels, not successful hits or winning.')
    ''')
    md('## 8. Export and download\nThe ZIP contains models, matching scalers/metadata, source snapshots, data hashes, training output, and dependency versions. It contains no raw recordings or emulator. Download it now. On Kaggle also save a notebook version with outputs to retain it; an unsaved session is not a backup.')
    code('''
    if not (RUN_DIR / 'artifact_checks.json').exists():
        raise RuntimeError('Run the verification cell successfully before exporting.')
    ZIP_PATH = Path(shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR))
    print('Download:', ZIP_PATH)
    if ON_COLAB:
        from google.colab import files
        files.download(str(ZIP_PATH))
    else:
        from IPython.display import FileLink, display
        import os
        # Relative links work in local Jupyter; Kaggle also exposes saved output files in its UI.
        display(FileLink(os.path.relpath(ZIP_PATH, Path.cwd())))
        print('If the link is unavailable, download this ZIP from the notebook Output/Files panel.')
    ''')
    md('''## 9. What to do on your Windows computer

1. Extract the result ZIP into a NEW folder, for example `trained_runs/first_check/` inside the project. Keep its `models/` folder intact. Do not merge it with older models.
2. In a Python 3.12 environment, install its `requirements-local.txt` and run `python -m pip check`. Keep the source feature encoder identical to the one used in training (a snapshot is included).
3. From the project root, start the emulator and load the ROM, then run:

   ```powershell
   python PythonAPI/run_bot.py 1 --tag practice --model-dir trained_runs/first_check/models
   ```

4. Open Gyroscope Bot and choose Ryu for the first check. The two-epoch model is only a pipeline test and may play poorly.
5. After practice, run `python tools/check_logs.py`. Its flags are estimates, not official judging.
6. Record more examples with `python PythonAPI/record.py 1`. Record every character, then rebuild the upload ZIP and use notebook 02.
7. For the full model, test all characters as both sides using the approved two-player setup. P2 command: replace `1` with `2`. Choose separate tags/log locations for repeat sessions so original logs are not overwritten.

**Before competition:** the current bot has rule-based spam/idle/forced-release overrides; they are not certified compliant with guideline rule 1. Address those and the validation split, and verify real match logging/timing. Producing model files is not proof of competition readiness.

**Data reminder:** record your own team playing all characters. Facing a character as an opponent is not a demonstration of controlling that character. Keep original match logs unchanged; never use hidden test fights for training.
''')
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'colab': {'name': '01_First_Training_Check.ipynb' if smoke else '02_All_Characters_Training.ipynb',
                  'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    folder = ROOT / 'notebooks'
    folder.mkdir(exist_ok=True)
    for smoke, name in [(True, '01_First_Training_Check.ipynb'), (False, '02_All_Characters_Training.ipynb')]:
        path = folder / name
        path.write_text(json.dumps(create(smoke), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(path)
