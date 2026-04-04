"""
diffusion_engine_sdturbo.py — SD-Turbo + T2I-Adapter engine.

SD-Turbo is a single-step adversarially-trained model (no LCM scheduler
needed). It runs faster than LCM at equivalent quality for 1-step inference.

Model: stabilityai/sd-turbo
Adapter: TencentARC/t2iadapter_openpose_sd14v1
"""

import queue
import threading
from typing import Optional

import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
)
from PIL import Image


class DiffusionEngineSDTurbo:
    """
    Drop-in replacement for DiffusionEngine using SD-Turbo + T2I-Adapter.

    Parameters
    ----------
    cfg : config.Config
    in_queue  : queue.Queue
    out_queue : queue.Queue
    """

    def __init__(self, cfg, in_queue: queue.Queue, out_queue: queue.Queue):
        self.cfg = cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._pipe: Optional[StableDiffusionAdapterPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def load(self) -> None:
        cfg = self.cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        device = cfg.device

        print("[SDTurbo] Loading T2I-Adapter ...")
        adapter = T2IAdapter.from_pretrained(
            cfg.t2i_adapter_model_id, torch_dtype=dtype
        )

        print("[SDTurbo] Loading SD-Turbo pipeline ...")
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            adapter=adapter,
            torch_dtype=dtype,
            safety_checker=None,
        )
        # SD-Turbo uses its own scheduler — no LCM needed
        pipe.vae = AutoencoderTiny.from_pretrained(
            cfg.taesd_model_id, torch_dtype=dtype
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        try:
            pipe.unet.set_attn_processor(
                __import__("diffusers").models.attention_processor.AttnProcessor2_0()
            )
            print("[SDTurbo] SDPA attention enabled")
        except Exception:
            pipe.enable_attention_slicing()

        print("[SDTurbo] Pre-computing text embeddings ...")
        with torch.inference_mode():
            self._prompt_embeds, self._neg_embeds = pipe.encode_prompt(
                prompt=cfg.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,  # SD-Turbo: guidance_scale=0
                negative_prompt=None,
            )

        self._pipe = pipe

        H, W = cfg.output_height, cfg.output_width
        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        self._transfer_stream = torch.cuda.Stream()

        print("[SDTurbo] Warming up (cudnn tuning) ...")
        dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        with torch.inference_mode():
            for _ in range(8):
                pipe(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=dummy,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    width=W,
                    height=H,
                    output_type="np",
                )
        torch.cuda.synchronize()
        print("[SDTurbo] Ready.")

    def start(self) -> "DiffusionEngineSDTurbo":
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="sdturbo"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        cfg = self.cfg
        torch.set_num_threads(2)
        generator = torch.Generator(device=cfg.device).manual_seed(42)

        while self._running:
            control_map = None
            try:
                while True:
                    item = self.in_queue.get_nowait()
                    if item is None:
                        self._running = False
                        break
                    control_map = item
            except queue.Empty:
                pass

            if not self._running:
                break

            if control_map is None:
                try:
                    control_map = self.in_queue.get(timeout=0.05)
                    if control_map is None:
                        self._running = False
                        break
                except queue.Empty:
                    continue

            if (
                control_map.dtype == np.float16
                and control_map.ndim == 3
                and control_map.shape[0] == 3
            ):
                ctrl_np = control_map
            else:
                ctrl_np = control_map.transpose(2, 0, 1).astype(np.float16) / 255.0
            self._pinned_buf[0].copy_(torch.from_numpy(ctrl_np), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                ctrl_tensor = self._pinned_buf.to(
                    device=cfg.device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

            with torch.inference_mode():
                result = self._pipe(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=ctrl_tensor,
                    num_inference_steps=1,
                    guidance_scale=0.0,  # SD-Turbo: CFG-free
                    width=cfg.output_width,
                    height=cfg.output_height,
                    generator=generator,
                    output_type="np",
                )

            frame = (result.images[0].clip(0, 1) * 255).astype(np.uint8)

            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(frame)
