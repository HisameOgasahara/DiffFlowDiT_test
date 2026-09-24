# Anima 생성 구현 비교

ComfyUI에서 생성한 PNG를 기준으로 ComfyUI·Diffusers·DiffSynth의 생성 결과와 실행 비용을 비교합니다. ComfyUI는 웹 UI 없이 코어 Python API로 실행합니다.

## 시작

1. 아래 노트북 중 하나를 Colab에서 엽니다.
2. T4 GPU를 선택하고 첫 셀에서 이 저장소를 clone합니다.
3. 설치 → 공통 HP → 모델 준비 → 생성 셀을 실행합니다.
4. 다른 두 노트북은 각각 새 Colab 런타임에서 같은 순서로 실행합니다.
5. 결과 ZIP을 다운로드하고 다른 런타임에 업로드한 뒤, 마지막 비교 셀에서 실행 폴더를 선택합니다.

- [ComfyUI 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/01_comfyui.ipynb)
- [Diffusers 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/02_diffusers.ipynb)
- [DiffSynth 노트북 실행](https://colab.research.google.com/github/HisameOgasahara/DiffFlowDiT_test/blob/main/03_diffsynth.ipynb)

소스는 `/content/DiffFlowDiT_test`, 설정·모델·결과는 `/content/sampling_synchro_data`에 저장됩니다. Google Drive 연결은 필요 없습니다. 런타임을 삭제하면 로컬 데이터도 사라지므로 결과 ZIP을 먼저 다운로드하세요. 모델 가중치는 Git에 포함하지 않고 실행 시 고정한 Hugging Face revision에서 내려받습니다. 기존 clone은 자동으로 덮어쓰지 않으며, 실행 결과에 프로젝트 커밋을 기록합니다.

## 원본 이미지 기준

| 항목 | 값 |
|---|---|
| DiT | anima-base-v1.0.safetensors |
| Text encoder | qwen_3_06b_base.safetensors |
| VAE | qwen_image_vae.safetensors |
| 크기 | 1216 × 832, batch 1 |
| Seed | 1113280783077040 |
| Steps / CFG | 30 / 4 |
| Sampler / scheduler | er_sde / simple |
| Denoise | 1 |
| LoRA | 연결되지 않아 적용되지 않음 |

전체 프롬프트는 `config/generation.json`, 원본 실행 그래프는 `config/prompt.json`, UI 배치는 `config/workflow.json`입니다. 프롬프트의 철자·줄바꿈·쉼표를 고치지 않았습니다. PNG는 EXIF가 아닌 텍스트 메타데이터로 워크플로를 저장합니다.

PNG에는 원래 실행의 정확한 코드 커밋·가중치 해시·연산 dtype이 없습니다. 이번 복사본은 현재 로컬 reference의 스냅샷이며, 원래 T4 세션과 같다고 가정하지 않습니다. 모델 다운로드 revision은 config/model_source.json에 고정했습니다.

## 비교 모드

| 모드 | ComfyUI | Diffusers | DiffSynth |
|---|---|---|---|
| native | PNG의 ER-SDE + simple | 원본 FlowMatch Euler | 원본 FlowMatch Euler |
| matched_euler | Euler + simple | 공통 simple 배열 + 원본 Euler step | 공통 simple 배열 + 원본 Euler step |

native는 프롬프트·seed·해상도·steps·CFG를 공유하면서 원본 구현 차이를 드러냅니다. HF/DiffSynth에서 ER-SDE를 몰래 Euler로 바꾼 뒤 같은 설정이라고 취급하지 않습니다. 실제 sampler는 metrics.json에 기록합니다.

matched_euler는 ComfyUI 모델의 shift=3, 1000점 표에서 선택한 simple sigma 배열, CPU FP32 초기 노이즈, Euler를 공유합니다. HF/DiffSynth에서는 solver 상태를 FP32로 유지하고 모델 계산은 runtime dtype을 사용합니다. 토큰 처리, 텍스트 조건, 모델 내부, VAE는 원본대로 남깁니다. 이 모드는 원본 PNG의 ER-SDE 재현이 아니며 결과 일치를 보장하지 않습니다. ER-SDE 왕복 이식은 다음 비교 단계입니다.

## 코드 위치

```text
01_comfyui.ipynb / 02_diffusers.ipynb / 03_diffsynth.ipynb
config/                 PNG에서 추출한 HP와 실행 정책
common/                 모델 준비, 기록, 결과 비교
comfyui/run.py          UI 없이 원본 코어 호출
comfyui/upstream/       comfy와 필요한 보조 모듈
diffusers/run.py        원본 Anima modular pipeline 호출
diffusers/convert_weights.py  공식 변환 함수를 구성요소별 프로세스에서 실행
diffusers/upstream/     원본 src/diffusers 및 변환 스크립트
diffsynth/run.py        원본 AnimaImagePipeline 호출
diffsynth/upstream/     원본 diffsynth 패키지
source_manifest.json   원본 커밋 및 복사한 파일별 SHA256
```

동적 모델 등록과 공통 로더 의존성을 보존하기 위해 각 패키지의 다른 모델 파일도 일부 포함합니다. ComfyUI의 main.py, server.py, nodes.py, 프런트엔드, 확장 노드 로딩은 실행 경로에 포함하지 않습니다. 원본 코어 파일은 바이트 단위로 보존하며, 비교용 변경은 run.py의 외부 래퍼에 있습니다. main.py의 allocator·DynamicVRAM 초기화만 headless_init.py에 옮겼습니다. CUDA allocator 원본 파일도 보존했습니다.

## T4 메모리와 최적화

- ComfyUI 기본은 원래처럼 dtype 자동 선택과 메모리 관리입니다. 명시적 FP16 비교는 runtime의 comfy_dtype를 float16으로 바꿉니다.
- Diffusers 기본은 ComponentsManager 자동 CPU offload와 FP16입니다. 부족하면 hf_offload=group으로 leaf 단위 offload를 사용합니다.
- HF 소스의 Hub API 요구 때문에 HF는 Transformers 5.17.0 / Hub 1.32.0, 다른 두 환경은 Transformers 4.57.6 / Hub 0.36.0으로 분리합니다. PyTorch 2.8.0은 공통입니다.
- DiffSynth는 CPU offload와 GPU 연산 dtype을 분리하고 VRAM 여유를 남깁니다. 디스크 offload는 기본으로 사용하지 않습니다.
- 새 가상환경에 FlashAttention·SageAttention·양자화 패키지를 추가하지 않습니다. 선택 가능한 attention API·모델 dtype·장치를 기록하지만 실제 CUDA kernel 전체를 profiler로 검증한 것은 아닙니다.
- Anima 텍스트 어댑터 호출 횟수를 기록합니다. ComfyUI/HF의 사전 계산과 DiffSynth의 반복 계산 차이를 확인할 수 있습니다.
- 자동으로 해상도·steps·가중치 정밀도를 낮추지 않습니다. OOM·NaN은 실패 또는 비교 결과로 남깁니다.
- FP16의 수치 안정성과 T4 실측은 Colab 실행으로 확인해야 합니다. CPU 테스트는 이를 대신하지 않습니다.

## 결과 읽기

각 실행 폴더에는 image.png, generation.json, runtime.json, models.json, metrics.json, schedule.json, run.log가 저장됩니다. trace=selected이면 첫·둘째·마지막 스텝의 latent·예측과 조건 텐서도 저장됩니다. ComfyUI ER-SDE의 추가 노이즈도 선택된 스텝에서 기록합니다.

공통 `x_before`, `denoised`, `final_model_latent`는 모델의 정규화 latent 공간을 비교합니다. ComfyUI의 VAE 입력 latent는 별도 `final_vae_latent`입니다. shape가 다르면 비교기가 억지로 정렬하지 않습니다. 텍스트 어댑터 전후 값과 토큰 ID를 보고 차이를 좁혀갑니다.

trace=selected에는 파일 저장·CPU 복사 비용이 포함됩니다. 성능은 같은 HP로 trace=none을 실행하고 cold/warm 여부를 통제해 비교하세요. 첫 실행에는 kernel 초기화도 포함됩니다. PyTorch allocated/reserved는 전체 GPU 프로세스 사용량과 같지 않습니다. CPU RSS는 0.1초 간격으로 샘플링합니다.

GPU 실행 성공이나 원본 이미지와의 수치 일치는 아직 검증 결과가 아닙니다. 로컬 검증 범위는 `VALIDATION.md`에 기록합니다.

## 원본 라이선스

ComfyUI 코어는 GPL-3.0 계열, Diffusers·DiffSynth는 Apache-2.0입니다. 각 upstream의 LICENSE와 파일 헤더를 유지했습니다. 소스 출처와 정확한 파일 해시는 source_manifest.json에 있습니다. 이 폴더를 재배포할 때 포함한 각 라이선스 조건을 확인해야 합니다.
