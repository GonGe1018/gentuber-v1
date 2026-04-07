# Realtime Live2D

[English](README.md)

웹캠 또는 영상 입력으로 애니메이션 캐릭터를 실시간 구동합니다. MediaPipe로 신체 포즈를 추출하고, Stable Diffusion 기반 파이프라인이 해당 포즈를 따라하는 애니메이션 캐릭터를 생성합니다.

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
- **Temporal Feedback**: 이전 프레임이 다음 프레임 생성의 입력으로 사용되어 프레임 간 스타일 일관성을 유지합니다.
- **고정 노이즈**: 매 프레임 동일한 latent 텐서를 사용하여 포즈 변화만 출력에 반영됩니다.
- **다중 백엔드**: ~5 FPS (최고 품질)부터 ~73 FPS (최고 속도)까지 선택 가능합니다.
- **헤드리스 녹화**: `--output result.mp4` 옵션으로 GUI 없이 영상을 처리하고 저장합니다.

## 요구 사항

- Python 3.12+
- CUDA 12.8+
- NVIDIA GPU (RTX 5070 Ti, 16GB VRAM에서 테스트됨)

## 설치

```bash
git clone https://github.com/GonGe1018/realtime-live2d
cd realtime-live2d
uv sync
```

`uv sync`으로 PyTorch CUDA 휠을 포함한 모든 의존성이 자동 설치됩니다.

## 빠른 시작

```bash
# 웹캠 실시간 (기본 IP-Adapter 백엔드)
uv run live2d --source 0

# 영상 파일 → MP4 저장 (GUI 없음)
uv run live2d --source input.mp4 --output result.mp4

# 커스텀 캐릭터 레퍼런스 이미지 사용
uv run live2d --source 0 --reference my_character.png
```

## 백엔드 비교

| 백엔드 | FPS (384px) | 캐릭터 일관성 | 포즈 정확도 | 용도 |
|---|---|---|---|---|
| `ip_adapter` | ~5-17 | 최고 | 양호 | 캐릭터 기반 애니메이션 |
| `lcm_graph` | ~60 | 낮음 | 양호 | 빠른 프로토타이핑 |
| `sdturbo_graph` | ~63 | 낮음 | 양호 | 빠른 프로토타이핑 |
| `controlnet` | ~19 | 중간 | 양호 | ControlNet 실험 |
| `sdturbo` | ~25 | 낮음 | 양호 | Eager 모드 디버깅 |
| `t2i` | ~25 | 낮음 | 양호 | T2I-Adapter 실험 |

`ip_adapter` 백엔드는 IP-Adapter Plus (캐릭터 외형) + ControlNet (포즈) + LCM-LoRA (속도) + 이전 프레임 피드백을 조합합니다.

## CLI 옵션

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--source` | `assets/test_input.mp4` | 영상 파일 경로 또는 웹캠 인덱스 (0, 1, ...) |
| `--output`, `-o` | — | MP4로 저장 후 종료 (헤드리스, GUI 없음) |
| `--reference` | `assets/reference.png` | IP-Adapter용 캐릭터 레퍼런스 이미지 |
| `--backend` | `ip_adapter` | `ip_adapter` / `lcm_graph` / `sdturbo_graph` / `sdturbo` / `t2i` / `controlnet` |
| `--steps` | `1` (ip_adapter는 4) | 추론 스텝 수 |
| `--size` | `384` | 출력 해상도: `256` / `384` / `512` |
| `--ip-scale` | `0.5` | IP-Adapter 강도 (0.3=약함, 0.7=강한 캐릭터 보존) |
| `--cn-scale` | `1.5` | ControlNet 포즈 강도 |
| `--feedback` | `0.3` | 시간적 피드백 (0.3=강한 일관성, 1.0=피드백 없음) |
| `--strength` | `0.5` | 레거시 백엔드용 img2img 강도 |
| `--seed` | `42` | 노이즈 시드 (-1 = 랜덤) |
| `--prompt` | config.py 참조 | 생성 프롬프트 |
| `--quality` | — | 프리셋: `fast` / `balanced` / `quality` |
| `--max-fps` | `60` | 디스플레이 갱신 상한 |
| `--no-skeleton` | off | 스켈레톤 오버레이 숨김 |
| `--no-interp` | off | 시간적 스무딩 비활성화 |
| `--no-hands` | off | 손 감지 건너뛰기 |

## 사용 예시

```bash
# 강한 캐릭터 보존, 적당한 포즈
uv run live2d --source 0 --ip-scale 0.6 --cn-scale 1.0

# 강한 포즈 반영, 약한 캐릭터
uv run live2d --source 0 --ip-scale 0.4 --cn-scale 1.5

# 시간적 피드백 없이 (각 프레임 독립)
uv run live2d --source 0 --feedback 1.0

# 댄스 영상 일괄 처리
uv run live2d --source dance.mp4 -o dance_anime.mp4 --steps 4

# 빠른 백엔드로 프로토타이핑 (~60 FPS)
uv run live2d --source 0 --backend lcm_graph
```

## IP-Adapter 백엔드 동작 원리

```
프레임 1:  고정 노이즈 → UNet + ControlNet(포즈₁) + IP-Adapter(캐릭터) → 출력₁
                                                                          ↓
프레임 2:  출력₁ + 노이즈 → UNet + ControlNet(포즈₂) + IP-Adapter(캐릭터) → 출력₂
                                                                              ↓
프레임 3:  출력₂ + 노이즈 → UNet + ControlNet(포즈₃) + IP-Adapter(캐릭터) → 출력₃
```

- **IP-Adapter Plus**: CLIP 패치 임베딩을 cross-attention에 주입하여 캐릭터 외형 보존 (시작 시 캐싱, 프레임당 추가 비용 없음)
- **ControlNet**: 매 디노이징 스텝마다 OpenPose 스켈레톤으로 포즈 가이드
- **Temporal Feedback**: 이전 출력이 img2img 입력으로 피드백되어 스타일 연속성 유지
- **고정 노이즈**: 첫 프레임에 결정적 latent 사용으로 재현 가능한 기준점 생성

## 설정

`config.py`를 편집하여 기본값을 변경할 수 있습니다:

```python
cfg.video_source = 0                        # 웹캠
cfg.engine_backend = "ip_adapter"           # 최고 캐릭터 일관성
cfg.reference_image = "my_character.png"    # 캐릭터 이미지
cfg.ip_adapter_scale = 0.5                  # 캐릭터 강도
cfg.controlnet_conditioning_scale = 1.5     # 포즈 강도
cfg.temporal_feedback_strength = 0.3        # 스타일 일관성
cfg.output_width = cfg.output_height = 384  # 해상도
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
| `TencentARC/t2iadapter_openpose_sd14v1` | ~77 MB | T2I-Adapter (lcm_graph 백엔드) |
| `madebyollin/taesd` | ~5 MB | 경량 VAE 디코더 |
| MediaPipe 포즈/손 모델 | ~14 MB | 포즈 추출 |

## 프로젝트 구조

```
main.py                                  # 파이프라인 오케스트레이션 + CLI
config.py                                # 모든 설정 파라미터
src/
  diffusion_engine_ip_adapter.py         # IP-Adapter + ControlNet + LCM (기본)
  diffusion_engine_lcm_graph.py          # KohakuV2 + LCM-LoRA + CUDA graph
  diffusion_engine_sdturbo_graph.py      # SD-Turbo + T2I-Adapter + CUDA graph
  diffusion_engine.py                    # LCM + ControlNet (eager)
  diffusion_engine_t2i.py               # LCM + T2I-Adapter (eager)
  diffusion_engine_sdturbo.py            # SD-Turbo + T2I-Adapter (eager)
  capture.py                             # 스레드 기반 영상 캡처 (FPS 동기화)
  pose_extractor.py                      # MediaPipe → OpenPose 스켈레톤
  interpolator.py                        # 시간적 프레임 블렌딩
  renderer.py                            # OpenCV 디스플레이 + FPS 오버레이
assets/
  reference.png                          # 기본 캐릭터 레퍼런스
  models/                                # MediaPipe 모델 파일
```

## 버전 히스토리

| 브랜치 | 설명 |
|---|---|
| `master` | 현재 — IP-Adapter + ControlNet + temporal feedback |
| `dev/v0.0.3-reference-img2img` | 레퍼런스 이미지 img2img 모드 |
| `dev/v0.0.2-source-img2img` | 카메라 프레임 img2img (StreamDiffusion 방식) |
| `dev/v0.0.1-noise-based` | 순수 노이즈 + T2I-Adapter |
| `dev/v0.0.1-alpha` | 초기 프로토타입 |

## 라이선스

MIT
