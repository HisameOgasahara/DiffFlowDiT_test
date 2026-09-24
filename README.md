# Anima 생성 구현 비교

Anima의 ComfyUI 생성 이미지·생성 설정·T4 실행 환경을 baseline으로 삼아, ComfyUI·Diffusers·DiffSynth의 추론 구현을 비교합니다. 먼저 UI 없는 ComfyUI 코드로 baseline을 재현하고, 다른 구현의 계산 조건을 맞춘 뒤 구성요소나 최적화를 하나씩 바꾸어 이미지·중간 텐서·시간·메모리 차이를 측정하는 ablation을 목표로 합니다.

## 시작

1. 아래 노트북 중 하나를 Colab에서 엽니다.
2. T4 GPU를 선택하고 첫 셀에서 이 저장소를 clone합니다.
3. 설치 → 공통 HP → 모델 준비 → 생성 셀을 실행합니다.
4. 다른 두 노트북은 각각 새 Colab 런타임에서 같은 순서로 실행합니다.
5. 결과 ZIP을 다운로드하고 다른 런타임에 업로드한 뒤, 마지막 비교 셀에서 실행 폴더를 선택합니다.

- [ComfyUI 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/01_comfyui.ipynb)
- [Diffusers 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/02_diffusers.ipynb)
- [DiffSynth 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/03_diffsynth.ipynb)

소스는 `/content/DiffFlowDiT_test`, 설정·모델·결과는 `/content/sampling_synchro_data`에 저장됩니다. 런타임 삭제 전에 결과 ZIP을 다운로드하세요. 기존 clone은 자동 갱신되지 않습니다.

## Baseline 이미지와 생성 설정

[Baseline 이미지](config/reference_colab_20260924.png)는 아래 설정과 실행 환경의 ComfyUI에서 생성한 기준 출력입니다. 각 구현의 재현 결과를 이 이미지와 비교합니다.

| 항목 | 값 |
|---|---|
| DiT | anima-base-v1.0.safetensors |
| Text encoder | qwen_3_06b_base.safetensors |
| VAE | qwen_image_vae.safetensors |
| 모델 저장소 revision | [circlestone-labs/Anima · f973fc4](https://huggingface.co/circlestone-labs/Anima/tree/f973fc41ec7545364ac9776c2440285f43ff2a30/split_files) |
| 크기 | 1216 × 832, batch 1 |
| Seed | 1113280783077040 |
| Steps / CFG | 30 / 4 |
| Sampler / scheduler | er_sde / simple |
| Denoise | 1 |
| LoRA | 연결되지 않아 적용되지 않음 |

프롬프트와 생성 설정은 [config/generation.json](config/generation.json), 실행 그래프는 [config/prompt.json](config/prompt.json), UI 배치는 [config/workflow.json](config/workflow.json)에 있습니다. 모델은 [config/model_source.json](config/model_source.json)의 고정 revision에서 내려받습니다.

## Baseline 실행 환경

| 항목 | 값 |
|---|---|
| Baseline 생성 노트북 | [anima_comfyui_colab.ipynb](https://github.com/HisameOgasahara/irodori_test/blob/5a2458c5efb6f32ac6ecac3701cead6b57f8374c/anima_comfyui_colab.ipynb) |
| ComfyUI / frontend | 0.37.0 / 1.53.6 |
| ComfyUI 커밋 범위 (파일 저장 시점 기준) | [b5cc883…1568e6c](https://github.com/Comfy-Org/ComfyUI/compare/b5cc8830279eae909a59de030af1e50761c36751...1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7) |
| 비교 코드의 ComfyUI 커밋 | [f427c3a](https://github.com/Comfy-Org/ComfyUI/commit/f427c3a285502cc452bbf134b810668844b79ffa) |
| Templates | 0.11.69 |
| OS / Python | Linux / 3.13.15 (GCC 13.3.0) |
| PyTorch / CUDA 빌드 | 2.14.0+cu130 |
| GPU / VRAM | Tesla T4 / 14.56 GB |
| allocator | cudaMallocAsync |
| DiT / VAE 기본 dtype (T4 코드 경로) | FP16 / FP16 |
| 텍스트 인코더 dtype (T4 코드 경로) | 가중치 FP16 / Qwen 계산 FP32 |
| 기본 attention 구현 (코드 경로) | PyTorch SDPA (`scaled_dot_product_attention`) |
| 실행 인자 | `main.py --listen 127.0.0.1 --port 8188 --enable-manager` |
| 설치 방식 | Colab Python으로 `uv venv --seed --system-site-packages`, ComfyUI requirements 및 manager_requirements 설치 |

세 비교 노트북은 Python 3.13.15 가상환경에 PyTorch 2.14.0+cu130과 호환 torchvision 0.29.0+cu130을 설치합니다.

## 비교 모드

| 모드 | ComfyUI | Diffusers | DiffSynth |
|---|---|---|---|
| native | PNG의 ER-SDE + simple | 원본 FlowMatch Euler | 원본 FlowMatch Euler |
| matched_euler | Euler + simple | 공통 simple 배열 + 원본 Euler step | 공통 simple 배열 + 원본 Euler step |

Baseline 재현에는 ComfyUI 노트북 마지막의 ER-SDE + simple 셀을 사용합니다. 기본 실행인 `matched_euler`는 세 구현의 sampler를 Euler로 통일한 비교 조건입니다. shift=3, 1000점 표의 simple sigma 배열과 CPU FP32 초기 노이즈를 공유하며, 이 조건에서는 ComfyUI Euler 출력을 기준으로 다른 두 구현을 비교합니다.

Ablation에서는 기준 실행과 변경 실행 사이에 비교 대상 요소만 바꾸고 나머지 조건을 유지합니다. 구현마다 남아 있는 패키지·정밀도·연산·offload 차이도 함께 기록해야 결과 차이를 해석할 수 있습니다.

## 코드 위치

- 각 구현의 `run.py`: 로딩·생성 실행, `upstream/`: 복사한 원본 코드
- `common/`: 모델 준비·기록·결과 비교·공통 정렬 코드
- Diffusers·DiffSynth의 `align_comfy.py`: ComfyUI 기준 조건 처리·연산·VAE 연결
- `comfyui/headless_init.py`: allocator·DynamicVRAM 초기화
- `source_manifest.json`: 원본 커밋과 복사한 파일별 SHA256

## T4 메모리와 최적화

실행 정책은 [config/runtime.json](config/runtime.json)에 있습니다. ComfyUI는 dtype 자동 선택과 자체 메모리 관리, Diffusers는 ComponentsManager 자동 CPU offload, DiffSynth는 CPU offload와 VRAM 여유 1.5 GiB를 기본으로 사용합니다.

Transformers / Hub는 Diffusers 환경에서 5.17.0 / 1.32.0, 나머지에서 4.57.6 / 0.36.0을 사용합니다.

## 결과 읽기

실행 폴더에 이미지, 생성·환경·모델 설정, 시간표, 로그, `metrics.json`의 시간·메모리·dtype·실제 sampler가 저장됩니다. `TRACE='selected'`는 조건 텐서와 첫·둘째·마지막 스텝을 기록합니다. 성능 비교에는 저장·CPU 복사 비용을 제외하는 `TRACE='none'`을 사용하세요.

UI 없는 ComfyUI 실행의 baseline 픽셀 일치와 세 구현 사이의 수치 일치는 아직 검증되지 않았습니다.

## 원본 라이선스

ComfyUI 코어는 GPL-3.0 계열, Diffusers·DiffSynth는 Apache-2.0입니다. 각 `upstream/`의 LICENSE를 확인하세요.
