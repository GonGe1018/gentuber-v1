# 파이프라인 단계별 출력

프로젝트 개발 과정에서 각 단계를 검증하며 생성된 파일들.

## 파일 목록

| 파일 | 단계 | 백엔드 | 설정 | 설명 |
|---|---|---|---|---|
| `stage1_skeleton.mp4` | Stage 1 | — | MediaPipe Pose | 포즈 추출 결과 (스켈레톤 오버레이) |
| `stage2_skeleton_input.png` | Stage 2 | — | — | UNet 입력용 스켈레톤 이미지 (단일 프레임) |
| `stage2_output.png` | Stage 2 | controlnet | steps=1, 384×384 | ControlNet 단일 프레임 생성 결과 |
| `t2i_output.mp4` | Stage 2 | t2i | steps=1, 384×384 | T2I-Adapter 백엔드 영상 출력 |
| `stage3_output.mp4` | Stage 3 | lcm_graph | steps=1, 384×384 | LCM+CUDA graph 파이프라인 영상 |
| `graph_engine_output.mp4` | Stage 3 | sdturbo_graph | steps=1, 384×384 | SD-Turbo+CUDA graph 파이프라인 영상 |
| `quality_check.mp4` | QA | lcm_graph | steps=1, 384×384 | 품질 검증용 영상 |
| `quality_check_thumb.png` | QA | lcm_graph | — | 품질 검증 썸네일 |

## 공통 설정

- 모델: KBlueLeaf/kohaku-v2.1 + LCM-LoRA (lcm_graph), stabilityai/sd-turbo (sdturbo_graph)
- T2I-Adapter: TencentARC/t2iadapter_openpose_sd14v1
- VAE: madebyollin/taesd
- 프롬프트: `"anime girl, full body, white background, high quality"`
- GPU: NVIDIA GeForce RTX 5070 Ti
