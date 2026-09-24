# 검증 범위

## 로컬에서 확인한 항목

- PNG의 실행 그래프에서 추출한 프롬프트·seed·생성 HP가 저장된 generation.json과 일치합니다.
- 노트북 3개의 JSON 스키마와 코드 셀 문법을 검사했습니다.
- ComfyUI 코어, DiffSynth Anima pipeline, Diffusers Anima modular blocks를 CPU 환경에서 import·구성했습니다.
- ComfyUI·DiffSynth는 Transformers 4.57.6 / Hub 0.36.0, Diffusers는 Transformers 5.17.0 / Hub 1.32.0으로 분리했습니다. 초기 공통 버전 구성의 Hub API 오류를 이 분리로 해결했습니다.
- 원본 safetensors 3개의 헤더만 내려받아 tensor 이름·shape를 확인했습니다. 전체 가중치는 로컬에 내려받지 않았습니다.
- DiffSynth의 registry가 세 파일의 key/shape 해시를 모두 인식합니다.
- 원본 HF 변환 함수에 meta tensor를 넣어 DiT·텍스트 어댑터·Qwen3·VAE의 key/shape를 strict 검증했습니다. 결과는 validation/conversion_shapes.log에 있습니다.
- 같은 sigma 배열에서 원본 ComfyUI·Diffusers·DiffSynth의 Euler 갱신을 상수 벡터장의 알려진 해와 비교했습니다.
- 공통 초기 노이즈와 원본 ComfyUI prepare_noise_inner의 결과를 비교했습니다.
- 비교기가 다른 seed, 다른 tensor shape, NaN을 구분하며 비교 이미지와 JSON을 생성하는지 검사했습니다.
- 복사한 upstream 파일의 SHA256을 원본 manifest와 대조했습니다.

## T4에서 확인해야 할 항목

- 전체 가중치 다운로드·CPU 로딩의 피크 RAM
- CUDA attention 선택, DynamicVRAM 초기화, CPU offload의 실제 작동
- 1216×832 / 30스텝의 OOM 여부, FP16 수치 안정성, 실행 시간
- 세 구현의 실제 조건·예측·latent·최종 이미지 오차
- 원본 PNG의 생성 환경과 이번 소스 스냅샷 차이

로컬 CPU 검사는 실제 T4 생성 성공이나 원본 이미지와의 일치를 의미하지 않습니다. 노트북에서 기록한 결과로 다음 수정 대상을 결정합니다.
