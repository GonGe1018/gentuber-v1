"""Test IP-Adapter with various ip_scale / cn_scale combinations."""

import sys

sys.path.insert(0, ".")
import cv2, queue, threading, time, numpy as np
from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_ip_adapter import DiffusionEngineIPAdapter
from src.pose_extractor import PoseExtractor


def run_test(label, ip_scale, cn_scale, steps=4, max_frames=15):
    cfg.video_source = "assets/dance_real.mp4"
    cfg.output_width = cfg.output_height = 384
    cfg.capture_width = cfg.capture_height = 384
    cfg.engine_backend = "ip_adapter"
    cfg.reference_image = "assets/reference.png"
    cfg.ip_adapter_scale = ip_scale
    cfg.controlnet_conditioning_scale = cn_scale
    cfg.num_inference_steps = steps

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=8)
    capture = VideoCapture(
        cfg.video_source, width=384, height=384, queue_size=2, loop=False
    )
    extractor = PoseExtractor(width=384, height=384, detect_hands=False)
    engine = DiffusionEngineIPAdapter(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)
    engine.load()

    stop = threading.Event()

    def pw():
        while not stop.is_set():
            f = capture.read(timeout=0.5)
            if f is None:
                stop.set()
                break
            c, _ = extractor.process(f)
            ctrl = extractor.preprocess(c)
            if pose_queue.full():
                try:
                    pose_queue.get_nowait()
                except:
                    pass
            pose_queue.put(ctrl)

    capture.start()
    threading.Thread(target=pw, daemon=True).start()
    engine.start()

    tag = f"ip{str(ip_scale).replace('.', '')}_cn{str(cn_scale).replace('.', '')}_st{steps}"
    fname = f"assets/dance_ip_{tag}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(fname, fourcc, 24.0, (384, 384))
    count = 0
    prev = None
    diffs = []
    means = []
    t0 = time.perf_counter()
    while count < max_frames:
        try:
            frame = out_queue.get(timeout=30)
        except:
            break
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        f32 = frame.astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(f32 - prev).mean())
        means.append(f32.mean())
        prev = f32
        count += 1

    elapsed = time.perf_counter() - t0
    writer.release()
    engine.stop()
    capture.stop()
    extractor.close()
    avg = np.mean(diffs) if diffs else 0
    print(
        f"{label:35s}: {count} frames, {count / elapsed:5.1f} FPS, avg_diff={avg:5.1f}, drift={means[-1] - means[0]:+5.1f} -> {fname}"
    )


if __name__ == "__main__":
    run_test("ip=0.4 cn=1.0 steps=4", 0.4, 1.0, 4)
    run_test("ip=0.5 cn=1.0 steps=4", 0.5, 1.0, 4)
    run_test("ip=0.6 cn=1.0 steps=4", 0.6, 1.0, 4)
    run_test("ip=0.5 cn=1.5 steps=4", 0.5, 1.5, 4)
