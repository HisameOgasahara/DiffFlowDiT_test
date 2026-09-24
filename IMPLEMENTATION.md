# 구현 경계와 변경 내역

## 그대로 보존한 부분

각 upstream 폴더는 로컬 reference에서 복사했으며 source_manifest.json의 파일 해시로 동일성을 검사합니다. 모델 클래스, attention, 가중치 변환, 원본 sampler, VAE 코드를 수정하지 않았습니다. 토크나이저 리소스와 라이선스도 포함합니다.

## 외부에서 추가한 부분

| 위치 | 역할 |
|---|---|
| comfyui/run.py | 노드 대신 load_clip → encode → load_diffusion_model → sample → VAE.decode 직접 호출 |
| comfyui/headless_init.py | main.py에서 서버와 무관한 allocator·DynamicVRAM 초기화만 분리 |
| diffusers/run.py | 원본 AnimaAutoBlocks와 ComponentsManager를 조립하고 측정 래퍼 추가 |
| diffusers/convert_weights.py | 원본 변환 함수를 CPU 메모리 절감을 위해 구성요소별 프로세스로 실행 |
| diffsynth/run.py | 원본 pipeline의 unit_runner·model_fn·step·decode에 측정 래퍼 추가 |
| common/runtime.py | 설정, SHA256, 텐서 추적, 시간·메모리·호출 횟수 기록 |

## matched_euler에서만 바뀌는 부분

- ComfyUI: sampler 선택을 ER-SDE에서 원본 Euler로 변경. 원본 simple 배열이 공통 계산과 일치하는지 검증.
- Diffusers: 원본 set_timesteps 호출 후 sigmas와 timesteps를 명시적으로 교체. 원본 step에 FP32 예측값을 전달해 FP32 solver 상태 유지.
- DiffSynth: 원본 set_timesteps 호출 후 sigmas와 timesteps 교체. 초기 노이즈를 공통 FP32 텐서로 공급. model_fn 입력은 모델 dtype으로 캐스팅하고 출력은 FP32로 반환.

Euler 수식 자체는 새로 작성하지 않습니다. CPU 테스트는 세 원본 Euler 구현을 같은 시간표와 상수 벡터장에 연결해 알려진 해와 비교합니다. 이 테스트는 실모델의 이미지 일치나 T4 성능을 입증하지 않습니다.

## ComfyUI 기준으로 적용하는 처리

- common/comfy_alignment.py: 원본 AnimaTokenizer, Qwen의 FP32 embedding·최종 norm·causal/padding mask·attention 연산, T5 토큰 가중치와 512 길이 zero padding을 연결합니다.
- diffusers/align_comfy.py: 원래 HF 모델의 모듈·가중치를 사용하면서 DiT residual을 FP32로 유지하고 ComfyUI의 RMSNorm·RoPE·attention·addcmul 순서를 적용합니다.
- diffsynth/align_comfy.py: Qwen 인코딩 후 텍스트 어댑터를 positive/negative별 한 번 계산합니다. DiT는 이 조건을 각 스텝에서 재사용합니다.
- 구성요소별 정밀도는 실행 장치에서 ComfyUI 자동 선택 함수로 정합니다. Qwen 연산은 SDClipModel과 같은 FP32를 사용합니다.
- VAE는 FP32 latent 역정규화 후 선택된 VAE dtype으로 변환하며, RMS 정규화·attention·단일 프레임 causal convolution·최종 픽셀 변환을 맞춥니다.
- 원본 upstream 파일은 유지하며 외부 연결 파일에서 실행 함수를 적용합니다. ER-SDE는 ComfyUI 노트북의 별도 셀에서 실행합니다.
