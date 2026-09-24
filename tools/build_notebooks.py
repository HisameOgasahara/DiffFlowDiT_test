import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result

def build(backend, number, title):
    cells = [cell("markdown", f"""
    # Anima 생성 비교 · {title}

    **목표:** 제공된 ComfyUI PNG의 프롬프트·seed·해상도로 생성하고, 중간 계산값과 T4 사용량을 저장합니다.

    1. Colab 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
    2. 아래 첫 셀에서 GitHub 소스를 clone합니다.
    3. 아래 셀을 순서대로 실행합니다. 모델 다운로드와 최초 변환에는 시간이 걸립니다.

    구현별 가상환경을 사용합니다. 런타임이 바뀌면 모델을 다시 내려받으며, 비교할 결과는 마지막 다운로드/업로드 셀로 옮깁니다.
    실제 로직은 `{backend}/run.py`와 `{backend}/upstream/`에 있습니다. 노트북에는 설치·설정·실행만 있습니다.
    """), cell("code", """
    # 1. 작업 폴더 준비
    from pathlib import Path
    import json, shutil, subprocess, sys

    REPOSITORY = 'https://github.com/HisameOgasahara/DiffFlowDiT_test.git'
    REVISION = 'main'  # 재현 실험에서는 사용할 태그 또는 브랜치로 고정
    PROJECT = Path('/content/DiffFlowDiT_test')
    DATA = Path('/content/sampling_synchro_data')
    if not PROJECT.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', REVISION, REPOSITORY, str(PROJECT)], check=True)
    subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], check=True)
    DATA.mkdir(parents=True, exist_ok=True)
    print('소스:', PROJECT)
    print('공통 설정·결과:', DATA)
    """), cell("markdown", """
    ## 설치

    Python 3.13.15 가상환경에 PyTorch 2.14.0+cu130과 호환 torchvision을 설치합니다.
    Colab 기본 PyTorch 버전이 달라도 가상환경에 기준 버전을 설치합니다.
    Transformers는 구현별 버전을 사용하며, 프레임워크는 동봉한 로컬 소스를 사용합니다.
    웹 UI·서버·터널은 시작하지 않습니다.
    """), cell("code", f"""
    # 2. 환경 설치
    BACKEND = {backend!r}
    INSTALL_LOG = DATA / f'install_{{BACKEND}}.log'
    with INSTALL_LOG.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', str(PROJECT / 'tools/setup_environment.py'), BACKEND],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        returncode = process.wait()
    if returncode:
        details = INSTALL_LOG.read_text(encoding='utf-8')[-12000:]
        raise RuntimeError(f'환경 설치 실패({{returncode}}): {{INSTALL_LOG}}\\n{{details}}')
    PYTHON = PROJECT / f'.venv-{{BACKEND}}/bin/python'
    environment_check = subprocess.run(
        [str(PYTHON), '-c', 'import sys, torch, torchvision; print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__); print("CUDA build:", torch.version.cuda); print("torchvision:", torchvision.__version__); assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name())'],
        capture_output=True, text=True,
    )
    print(environment_check.stdout, end='')
    if environment_check.returncode:
        raise RuntimeError(environment_check.stderr)
    """), cell("markdown", """
    ## 공통 생성 설정

    원본 PNG에서 추출했습니다. 기본값은 1216×832, 30스텝, CFG 4, seed 1113280783077040입니다.
    LoRA 노드는 연결되지 않았으므로 적용하지 않습니다. Positive 노드의 이름이 Negative여도 실제 연결 기준으로 추출했습니다.

    **원본 비교(`native`)**: ComfyUI는 ER-SDE+simple, Diffusers·DiffSynth는 원본 Flow Euler를 사용합니다. 이 차이는 결과에 명시됩니다.

    **계산 조건 통일(`matched_euler`)**: 세 구현 모두 ComfyUI simple 시간표·동일 FP32 노이즈·Euler를 사용합니다.
    프롬프트 가중치·Qwen·텍스트 어댑터·연산 정밀도·VAE는 ComfyUI 기준으로 처리합니다.
    """), cell("code", """
    # 3. 공통 HP 확인·수정
    CONFIG = DATA / 'generation.json'
    if not CONFIG.exists():
        shutil.copy2(PROJECT / 'config/generation.json', CONFIG)
    hp = json.loads(CONFIG.read_text(encoding='utf-8'))
    # 수정 예: hp['steps'] = 30
    # 수정 예: hp['seed'] = 1113280783077040
    # 수정 예: hp['prompt'] = '새 프롬프트'
    CONFIG.write_text(json.dumps(hp, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(hp, ensure_ascii=False, indent=2))

    MODE = 'matched_euler'  # 공통 Euler 비교. ComfyUI ER-SDE는 별도 셀에서 실행
    TRACE = 'selected'  # 첫·둘째·마지막 스텝 저장. 속도 측정만 할 때 'none'
    runtime = json.loads((PROJECT / 'config/runtime.json').read_text(encoding='utf-8'))
    # ComfyUI의 T4 자동 선택을 기준으로 구성요소별 연산 정밀도를 적용
    # hf_offload='auto': 모델 관리 / 'group': 더 세밀한 offload
    # comfy_memory='low': ComfyUI lowvram / 기본 'normal'
    RUNTIME = PROJECT / 'config/runtime_run.json'
    RUNTIME.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding='utf-8')
    print('실험:', MODE, '텐서 기록:', TRACE)
    """), cell("markdown", """
    ## 같은 모델 파일 준비

    세 구현에 같은 safetensors 파일을 제공합니다. 최초 다운로드 revision과 SHA256을 고정합니다.
    원본 PNG에는 가중치 해시가 없으므로, 파일명이 같은 것 이상으로 원본 생성 당시 파일과 동일하다고 보장하지 않습니다.
    텍스트 tokenizer 파일과 괄호 가중치·토큰 처리 규칙은 동봉한 ComfyUI를 기준으로 공유합니다.

    모델과 결과는 런타임의 `/content/sampling_synchro_data`에 저장합니다. 런타임 삭제 전에 필요한 결과를 다운로드하세요.
    """), cell("code", """
    # 4. 모델 준비 — 최초 1회 다운로드
    MODELS = DATA / 'models'
    subprocess.run([str(PYTHON), '-m', 'common.prepare_models', '--models', str(MODELS), '--config', str(CONFIG)], cwd=PROJECT, check=True)
    print('실행용 로컬 모델:', MODELS)
    """), cell("markdown", """
    ## 생성 실행

    실행 로그, 적용된 설정, 실제 dtype·장치, 호출 횟수, 시간·VRAM·RAM을 기록합니다.
    `TRACE='selected'`는 중간 텐서를 CPU로 복사하므로 속도 측정에 영향을 줍니다. 성능은 `TRACE='none'` 실행끼리 비교하세요.
    OOM이 나면 로그와 실패 정보를 저장합니다. 품질에 영향을 주는 양자화나 해상도 변경을 자동 적용하지 않습니다.
    """), cell("code", """
    # 5. 실제 로직은 .py에서 실행
    from datetime import datetime, timezone
    RUN_NAME = f"{BACKEND}_{MODE}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    OUTPUT = DATA / 'runs' / RUN_NAME
    LOG = OUTPUT.parent / f'{RUN_NAME}.log'
    OUTPUT.parent.mkdir(exist_ok=True)
    command = [str(PYTHON), '-u', str(PROJECT / BACKEND / 'run.py'), '--config', str(CONFIG),
               '--runtime', str(RUNTIME), '--models', str(MODELS), '--output', str(OUTPUT), '--mode', MODE, '--trace', TRACE]
    try:
        with LOG.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='')
                stream.write(line)
            code = process.wait()
        if code:
            raise RuntimeError(f'실행 실패({code}). 저장된 로그를 확인하세요.')
    finally:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOG, OUTPUT / 'run.log')
        subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], stdout=(OUTPUT / 'project_commit.txt').open('w'), check=True)
        print('결과 저장:', OUTPUT)
    """), cell("code", """
    # 6. 이미지와 측정값 확인
    from IPython.display import display, Image, JSON
    display(Image(filename=str(OUTPUT / 'image.png')))
    metrics = json.loads((OUTPUT / 'metrics.json').read_text(encoding='utf-8'))
    display(JSON({key: metrics.get(key) for key in ('status', 'effective_sampler', 'effective_schedule', 'stages', 'forward_calls', 'models', 'peak_cpu_rss_bytes')}))
    """), cell("markdown", """
    ## 결과를 다른 런타임으로 옮기기 (선택)

    현재 결과 ZIP을 다운로드하고, 다른 노트북에서는 이전 ZIP을 업로드합니다. 모델은 ZIP에 포함하지 않습니다.
    같은 런타임에 결과가 이미 있으면 건너뛰세요.
    """), cell("code", """
    # 결과 다운로드 — 필요할 때 True로 변경
    DOWNLOAD = False
    if DOWNLOAD:
        from google.colab import files
        archive = shutil.make_archive(str(DATA / RUN_NAME), 'zip', OUTPUT.parent, OUTPUT.name)
        files.download(archive)
    """), cell("code", """
    # 이전 실행 결과 업로드 — 필요할 때 True로 변경
    UPLOAD = False
    if UPLOAD:
        from google.colab import files
        import io, zipfile
        for name, payload in files.upload().items():
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                destination = (DATA / 'runs').resolve()
                for item in archive.infolist():
                    target = (destination / item.filename).resolve()
                    if not target.is_relative_to(destination) or target.exists():
                        raise ValueError(f'허용되지 않거나 이미 존재하는 결과 경로: {item.filename}')
                archive.extractall(destination)
    """), cell("markdown", """
    ## 다른 실행과 비교

    아래 목록에서 비교할 실행 폴더 2~3개를 선택합니다. 가능한 한 같은 MODE·HP·가중치를 선택하세요.
    결과 이미지, 차이 강조 이미지, 중간 텐서 오차와 측정표를 저장합니다.
    shape가 다르면 임의로 자르거나 맞추지 않고 차이로 표시합니다.
    """), cell("code", """
    # 7. 비교할 실행 폴더 선택
    RUNS = DATA / 'runs'
    available = sorted(p for p in RUNS.iterdir() if (p / 'metrics.json').exists())
    for path in available:
        print(path.name)
    SELECTED = []  # 예: ['comfyui_matched_euler_...', 'diffusers_matched_euler_...']
    if len(SELECTED) >= 2:
        COMPARE = DATA / 'comparisons' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        subprocess.run([str(PYTHON), '-m', 'common.compare', *[str(RUNS / name) for name in SELECTED], '--output', str(COMPARE)], cwd=PROJECT, check=True)
        if (COMPARE / 'comparison.png').exists():
            display(Image(filename=str(COMPARE / 'comparison.png')))
        print((COMPARE / 'comparison.md').read_text(encoding='utf-8'))
    else:
        print('다른 노트북도 실행한 뒤 SELECTED에 비교할 폴더 이름을 넣으세요.')
    """)]
    if backend == "comfyui":
        cells.extend([cell("markdown", """
        ## ComfyUI ER-SDE 생성

        같은 프롬프트·seed·해상도로 ER-SDE + simple 이미지를 별도 폴더에 저장합니다.
        """), cell("code", """
        ER_CONFIG = DATA / 'generation_er_sde.json'
        er_hp = json.loads(CONFIG.read_text(encoding='utf-8'))
        er_hp.update(sampler_name='er_sde', scheduler='simple')
        ER_CONFIG.write_text(json.dumps(er_hp, ensure_ascii=False, indent=2), encoding='utf-8')
        ER_OUTPUT = DATA / 'runs' / f"comfyui_er_sde_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        ER_LOG = ER_OUTPUT.parent / f'{ER_OUTPUT.name}.log'
        er_command = [str(PYTHON), '-u', str(PROJECT / 'comfyui/run.py'), '--config', str(ER_CONFIG),
                      '--runtime', str(RUNTIME), '--models', str(MODELS), '--output', str(ER_OUTPUT),
                      '--mode', 'native', '--trace', TRACE]
        try:
            with ER_LOG.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen(er_command, cwd=PROJECT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end='')
                    stream.write(line)
                code = process.wait()
            if code:
                raise RuntimeError(f'ER-SDE 실행 실패({code}): {ER_LOG}')
        finally:
            ER_OUTPUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ER_LOG, ER_OUTPUT / 'run.log')
            with (ER_OUTPUT / 'project_commit.txt').open('w') as stream:
                subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], stdout=stream, check=True)
        display(Image(filename=str(ER_OUTPUT / 'image.png')))
        print('ER-SDE 결과:', ER_OUTPUT)
        """)])
    cells.extend([cell("markdown", """
    ## 결과 폴더를 Google Drive에 복사 (선택)

    생성이 끝난 뒤 실행하세요. `runs` 전체(이미지·설정·중간 텐서·실행 로그)와 설치 로그를
    `MyDrive/Anima_results` 아래 새 폴더에 복사합니다. ComfyUI의 Euler·ER-SDE 결과도 함께 포함됩니다.
    """), cell("code", """
    from google.colab import drive
    from pathlib import Path
    from datetime import datetime, timezone
    import shutil

    drive.mount('/content/drive')
    source = Path('/content/sampling_synchro_data')
    destination = Path('/content/drive/MyDrive/Anima_results') / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    shutil.copytree(source / 'runs', destination)
    for install_log in source.glob('install_*.log'):
        shutil.copy2(install_log, destination / install_log.name)
    print('복사 완료:', destination)
    """)])
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": f"{number}_{backend}.ipynb", "gpuType": "T4"}},
                "nbformat": 4, "nbformat_minor": 5}
    for i, item in enumerate(cells):
        item["id"] = f"{backend}-{i:02d}"
    (ROOT / f"{number}_{backend}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    for values in (("comfyui", "01", "ComfyUI UI 없는 실행"), ("diffusers", "02", "Hugging Face Diffusers"), ("diffsynth", "03", "DiffSynth-Studio")):
        build(*values)
