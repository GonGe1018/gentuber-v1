# GenTuber v1

[English](README.md)

웹캠 또는 영상 입력으로 애니메이션 캐릭터를 실시간 구동합니다. MediaPipe로 신체 포즈를 추출하고, Stable Diffusion 기반 파이프라인이 해당 포즈를 따라하는 애니메이션 캐릭터를 생성합니다.

## 데모

```bash
# Side-by-side 데모 생성 (Source | Skeleton | Generated)
uv run gentuber --no-gui --source input.mp4 --output demo.mp4 --half-body
```

RTX 5070 Ti 기준 256px에서 ~13 FPS

## 아키텍처

```
[웹캠 / 영상] → [MediaPipe 포즈 추출] → [디퓨전 엔진 (GPU)] → [화면 / MP4]
                  스켈레톤 추출            IP-Adapter (캐릭터 외형)
                                          ControlNet (포즈 가이드)
                                          LCM (고속 추론)
```

각 단계는 별도 스레드에서 실행되며, 큐 크기가 제한되어 있어 지연이 누적되지 않습니다.

## 주요 기능

- **IP-Adapter + ControlNet**: 캐릭터 외형과 포즈가 독립적인 경로로 주입되어 서로 충돌하지 않습니다.
- **Temporal Feedback**: 이전 프레임의 latent가 다음 프레임 생성에 직접 재사용되어 프레임 간 스타일 일관성을 유지합니다.
- **고정 노이즈**: 매 프레임 동일한 latent 텐서를 사용하여 포즈 변화만 출력에 반영됩니다.
- **반샷 (VTuber) 모드**: 상체만 표시하는 스켈레톤으로 VTuber 스타일 버스트샷을 생성합니다.
- **헤드리스 녹화**: `--output result.mp4` 옵션으로 GUI 없이 영상을 처리하고 저장합니다.

## 요구 사항

- Python 3.12+
- CUDA 12.8+
- NVIDIA GPU (RTX 5070 Ti, 16GB VRAM에서 테스트됨)

## 설치

```bash
git clone https://github.com/GonGe1018/gentuber-v1
cd gentuber-v1
uv sync
```

`uv sync`으로 PyTorch CUDA 휠을 포함한 모든 의존성이 자동 설치됩니다.

## 빠른 시작

```bash
# 웹캠 실시간 (기본 IP-Adapter 백엔드)
uv run gentuber --source 0

# 영상 파일 → MP4 저장 (GUI 없음)
uv run gentuber --source input.mp4 --output result.mp4

# 커스텀 캐릭터 레퍼런스 이미지 사용
uv run gentuber --source 0 --reference my_character.png
```

## 백엔드

기본(유일한 활성) 백엔드는 `ip_adapter`입니다 — IP-Adapter Plus (캐릭터 외형) + ControlNet (포즈) + LCM-LoRA (속도) + latent 레벨 temporal feedback.

| 백엔드 | FPS (384px) | 설명 |
|---|---|---|
| `ip_adapter` | ~13 | Latent feedback 기반 캐릭터 애니메이션 |

> 레거시 백엔드 (lcm_graph, sdturbo_graph, controlnet 등)는 `src/legacy/`에 보관되어 있습니다. [src/legacy/README.md](src/legacy/README.md) 참조.

## CLI 옵션

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--source` | `assets/test_input.mp4` | 영상 파일 경로 또는 웹캠 인덱스 (0, 1, ...) |
| `--output`, `-o` | — | MP4로 저장 후 종료 (헤드리스, GUI 없음) |
| `--reference` | `assets/reference.png` | IP-Adapter용 캐릭터 레퍼런스 이미지 |
| `--backend` | `ip_adapter` | 디퓨전 백엔드 |
| `--steps` | `4` | 추론 스텝 수 |
| `--size` | `384` | 출력 해상도: `256` / `384` / `512` |
| `--ip-scale` | `0.5` | IP-Adapter 강도 (0.3=약함, 0.7=강한 캐릭터 보존) |
| `--cn-scale` | `2.0` | ControlNet 포즈 강도 |
| `--feedback` | `0.3` | 시간적 피드백 (0.3=강한 일관성, 1.0=피드백 없음) |
| `--seed` | `42` | 노이즈 시드 (-1 = 랜덤) |
| `--prompt` | config.py 참조 | 생성 프롬프트 |
| `--half-body` | off | VTuber 모드: 상체만 표시 |
| `--quality` | — | 프리셋: `fast` / `balanced` / `quality` |
| `--max-fps` | `60` | 디스플레이 갱신 상한 |
| `--no-skeleton` | off | 스켈레톤 오버레이 숨김 |
| `--no-interp` | off | 시간적 스무딩 비활성화 |
| `--no-hands` | off | 손 감지 건너뛰기 |
| `--motion-lo` | `0.008` | 적응형 모션: 떨림 임계값 |
| `--motion-hi` | `0.04` | 적응형 모션: 큰 움직임 리셋 임계값 |
| `--motion-max` | `0.85` | 적응형 모션: 최대 피드백 강도 상한 |

## 사용 예시

```bash
# 강한 캐릭터 보존, 적당한 포즈
uv run gentuber --source 0 --ip-scale 0.6 --cn-scale 1.0

# 강한 포즈 반영, 약한 캐릭터
uv run gentuber --source 0 --ip-scale 0.4 --cn-scale 2.0

# 시간적 피드백 없이 (각 프레임 독립)
uv run gentuber --source 0 --feedback 1.0

# 댄스 영상 일괄 처리
uv run gentuber --source dance.mp4 -o dance_anime.mp4 --steps 4

# VTuber 반샷 모드
uv run gentuber --source 0 --half-body
```

## IP-Adapter 백엔드 동작 원리

```
프레임 1:  고정 노이즈 → UNet + ControlNet(포즈₁) + IP-Adapter(캐릭터) → Latent₁ → 디코드 → 출력₁
                                                                            ↓
프레임 2:  Latent₁ + 노이즈 → UNet + ControlNet(포즈₂) + IP-Adapter(캐릭터) → Latent₂ → 디코드 → 출력₂
                                                                                ↓
프레임 3:  Latent₂ + 노이즈 → UNet + ControlNet(포즈₃) + IP-Adapter(캐릭터) → Latent₃ → 디코드 → 출력₃
```

- **IP-Adapter Plus**: CLIP 패치 임베딩을 cross-attention에 주입하여 캐릭터 외형 보존 (시작 시 캐싱, 프레임당 추가 비용 없음)
- **ControlNet**: 매 디노이징 스텝마다 OpenPose 스켈레톤으로 포즈 가이드
- **Latent Feedback**: 이전 프레임의 latent를 직접 재사용 (VAE 인코드 사이클 없음 → 색상 드리프트 없음)
- **적응형 모션**: 작은 움직임 = 적은 디노이즈 스텝 (빠름), 큰 움직임 = txt2img 리셋 (깨끗함)

## 설정

`config.py`를 편집하여 기본값을 변경할 수 있습니다:

```python
cfg.video_source = 0                        # 웹캠
cfg.engine_backend = "ip_adapter"           # 최고 캐릭터 일관성
cfg.reference_image = "my_character.png"    # 캐릭터 이미지
cfg.ip_adapter_scale = 0.6                  # 캐릭터 강도
cfg.controlnet_conditioning_scale = 1.8     # 포즈 강도
cfg.temporal_feedback_strength = 0.2        # 스타일 일관성
cfg.output_width = cfg.output_height = 256  # 해상도
```

## 모델

첫 실행 시 `~/.cache/huggingface/`에 자동 다운로드됩니다:

| 모델 | 크기 | 용도 |
|---|---|---|
| `KBlueLeaf/kohaku-v2.1` | ~2 GB | 애니메이션 SD1.5 기본 모델 |
| `latent-consistency/lcm-lora-sdv1-5` | ~200 MB | LCM-LoRA 고속 추론 |
| `h94/IP-Adapter` (ip-adapter-plus_sd15) | ~98 MB | 캐릭터 외형 보존 |
| `h94/IP-Adapter` (image_encoder) | ~1.2 GB | CLIP ViT-H-14 (캐싱 후 언로드) |
| `lllyasviel/control_v11p_sd15_openpose` | ~361 MB | ControlNet OpenPose |
| `madebyollin/taesd` | ~5 MB | 경량 VAE 디코더 |
| MediaPipe 포즈/손 모델 | ~14 MB | 포즈 추출 |

## 프로젝트 구조

```
main.py                                  # 파이프라인 오케스트레이션 + CLI
config.py                                # 모든 설정 파라미터
src/
  diffusion_engine_ip_adapter.py         # IP-Adapter + ControlNet + LCM (기본)
  capture.py                             # 스레드 기반 영상 캡처 (FPS 동기화)
  pose_extractor.py                      # MediaPipe → OpenPose 스켈레톤
  interpolator.py                        # 시간적 프레임 블렌딩
  renderer.py                            # OpenCV 디스플레이 + FPS 오버레이
  settings_gui.py                        # Tkinter 설정 패널
  legacy/                                # 보관된 백엔드 (legacy/README.md 참조)
assets/
  reference.png                          # 기본 캐릭터 레퍼런스
  models/                                # MediaPipe 모델 파일
```

## 버전 히스토리

| 브랜치 | 설명 |
|---|---|
| `master` | 현재 — IP-Adapter + ControlNet + latent feedback |
| `dev/v0.0.7-latent-feedback` | Latent 레벨 temporal feedback |
| `dev/v0.0.6-vtuber-halfbody` | VTuber 반샷 모드 |
| `dev/v0.0.5-ip-adapter-gui` | IP-Adapter + GUI 설정 패널 |
| `dev/v0.0.3-reference-img2img` | 레퍼런스 이미지 img2img 모드 |
| `dev/v0.0.2-source-img2img` | 카메라 프레임 img2img (StreamDiffusion 방식) |
| `dev/v0.0.1-noise-based` | 순수 노이즈 + T2I-Adapter |
| `dev/v0.0.1-alpha` | 초기 프로토타입 |

## 라이선스

MIT
