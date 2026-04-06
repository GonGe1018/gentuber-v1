# 댄스 영상 Temporal Coherence 실험

소스: `../00_source/dance_real.mp4` (Mixkit 댄스 영상, 720×1280, 24 FPS, 10초)

모든 출력: 384×384, 24 FPS, lcm_graph 백엔드 (KohakuV2 + LCM-LoRA)

## 결과 비교

| 파일 | 방식 | 설정 | FPS | avg_diff | 설명 |
|---|---|---|---|---|---|
| `dance_anime.mp4` | 랜덤 노이즈 | `--strength 1.0 --temporal 1.0` | ~55 | 31.7 | 매 프레임 독립 생성, 깜빡임 심함 |
| `dance_anime_temporal.mp4` | 고정 노이즈 | `--temporal 0.5` | ~54 | 8.4 | 같은 노이즈 재사용, 포즈만 변화 |
| `dance_anime_fixednoise.mp4` | 고정 노이즈 | `--temporal 0.5` | ~51 | 8.3 | 위와 동일 (코드 정리 후 재생성) |
| `dance_anime_img2img.mp4` | img2img 피드백 | `--strength 0.7` | ~22 | 8.0 | 이전 프레임 x0를 t=699로 re-noise |
| `dance_anime_img2img_s5.mp4` | img2img 피드백 | `--strength 0.5` | ~15 | 9.3 | 더 강한 구조 보존, mean drift 있음 |
| `dance_img2img_s3.mp4` | img2img 피드백 | `--strength 0.3` | ~20 | 9.7 | t=299, alpha=0.59 |
| `dance_img2img_s5.mp4` | img2img 피드백 | `--strength 0.5` | ~26 | 7.8 | t=499, alpha=0.28 |
| `dance_img2img_s7.mp4` | img2img 피드백 | `--strength 0.7` | ~24 | 8.3 | t=699, alpha=0.08 |

## avg_diff 설명

연속 프레임 간 평균 픽셀 차이 (0-255 스케일). 낮을수록 부드러운 전환.

- 30+ = 매 프레임 완전히 다른 이미지 (깜빡임)
- 8~10 = 부드러운 전환
- 0~2 = 거의 정지 화면

## 결론

- **고정 노이즈** (`--temporal 0.5`): 60 FPS 유지하면서 avg_diff 8.4로 가장 실용적
- **img2img 피드백** (`--strength 0.5~0.7`): avg_diff는 비슷하지만 이전 프레임 구조를 직접 보존. 단, FPS 절반 + mean drift 발생
