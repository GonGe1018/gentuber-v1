# 고정 포즈 Temporal Coherence 비교

소스: `../00_source/static_pose.mp4` (dance_real.mp4 첫 프레임 × 120, 고정 포즈)

고정 포즈에서 각 방식의 안정성을 측정. 포즈가 변하지 않으므로 이상적으로는 출력도 변하지 않아야 함.

모든 출력: 384×384, 24 FPS, 60프레임, lcm_graph 백엔드

## 결과 비교

| 파일 | 방식 | 설정 | FPS | avg_diff | mean_drift | 평가 |
|---|---|---|---|---|---|---|
| `static_no_coherence.mp4` | 랜덤 노이즈 | `--strength 1.0 --temporal 1.0` | 51.4 | 25.5 | -33.0 | 깜빡임 심함 |
| `static_fixed_noise.mp4` | 고정 노이즈 | `--temporal 0.5` | 52.7 | 8.4 | **-3.6** | 안정적 ✅ |
| `static_img2img_s7.mp4` | img2img 피드백 | `--strength 0.7` | 23.6 | 5.5 | -42.9 | 부드럽지만 어두워짐 |
| `static_img2img_s5.mp4` | img2img 피드백 | `--strength 0.5` | 24.6 | 4.7 | -75.4 | 더 어두워짐 |
| `static_img2img_s3.mp4` | img2img 피드백 | `--strength 0.3` | 24.5 | 5.4 | -87.4 | 심하게 어두워짐 |

## 지표 설명

- **avg_diff**: 연속 프레임 간 평균 픽셀 차이 (0-255). 낮을수록 안정적.
- **mean_drift**: 마지막 프레임 mean - 첫 프레임 mean. 0에 가까울수록 좋음. 음수 = 점점 어두워짐.

## 결론

- **고정 노이즈**가 drift 거의 없이(-3.6) 가장 안정적
- **img2img 피드백**은 avg_diff는 낮지만 epsilon 예측 편향이 누적되어 mean drift 심각
- strength가 낮을수록(이전 프레임 더 많이 보존) drift가 더 심해짐
- 고정 포즈 기준으로는 고정 노이즈 방식이 최적
