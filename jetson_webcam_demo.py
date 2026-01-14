#!/usr/bin/env python3
"""
Live webcam captioning demo for Jetson Orin Nano.

Press SPACEBAR to capture and generate caption.
Press 'q' to quit.

Usage:
    # PyTorch backend (default)
    python jetson_webcam_demo.py --checkpoint checkpoints_caption_poc/final.pt

    # ONNX backend (faster on Jetson)
    python jetson_webcam_demo.py --onnx model_decoder.onnx --config model_config.json

    # Adjust generation parameters
    python jetson_webcam_demo.py --checkpoint final.pt --steps 25 --temperature 0.9 --max_len 12
"""

import argparse
import json
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor

from model import DiffusionTransformer, ModelConfig
from diffusion import DiscreteDiffusion


class CaptionGenerator:
    """Handles caption generation from images using PyTorch or ONNX."""

    def __init__(
        self,
        decoder,
        vision_model,
        image_processor,
        tokenizer,
        device: str = "cpu",
        max_len: int = 12,
        steps: int = 25,
        temperature: float = 0.8,
        backend: str = "pytorch",
        onnx_session=None,
    ):
        self.decoder = decoder
        self.vision_model = vision_model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.steps = steps
        self.temperature = temperature
        self.backend = backend
        self.onnx_session = onnx_session

        # Token IDs
        self.pad_id = tokenizer.token_to_id("<PAD>")
        self.mask_id = tokenizer.token_to_id("<MASK>")
        self.bos_id = tokenizer.token_to_id("<BOS>")
        self.eos_id = tokenizer.token_to_id("<EOS>")

        # For PyTorch backend
        if backend == "pytorch":
            self.vocab_size = decoder.vocab_size
        else:
            # For ONNX, need to get vocab size from config
            self.vocab_size = tokenizer.get_vocab_size()

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP features from PIL image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        outputs = self.vision_model(pixel_values)
        features = outputs.last_hidden_state.cpu().numpy()  # [1, seq_len, d_model]

        return features

    @torch.no_grad()
    def generate_pytorch(self, image_features: torch.Tensor) -> str:
        """Generate caption using PyTorch backend."""
        batch_size = 1
        image_features = image_features.to(self.device)

        # Create attention mask
        image_mask = torch.ones(image_features.shape[:2], device=self.device)

        # Start with all masks
        x = torch.full((batch_size, self.max_len), self.mask_id, device=self.device)

        # Iterative denoising
        timesteps = torch.linspace(1.0, 0.0, self.steps + 1, device=self.device)

        for step in range(self.steps):
            t = timesteps[step].expand(batch_size)
            t_next = timesteps[step + 1].expand(batch_size)

            # Forward pass
            logits = self.decoder(
                x, t,
                encoder_output=image_features,
                encoder_attention_mask=image_mask,
            )

            # Sample
            probs = torch.softmax(logits / self.temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1)
            sampled = sampled.view(batch_size, self.max_len)

            # Update mask
            current_mask_rate = 1 - torch.cos(t * torch.pi / 2)
            next_mask_rate = 1 - torch.cos(t_next * torch.pi / 2)
            keep_mask_prob = (next_mask_rate / current_mask_rate.clamp(min=1e-8)).clamp(max=1.0)

            is_masked = (x == self.mask_id)
            rand = torch.rand(batch_size, self.max_len, device=self.device)
            keep_mask = rand < keep_mask_prob.unsqueeze(1)

            unmask = is_masked & ~keep_mask
            x[unmask] = sampled[unmask]

        # Decode tokens
        tokens = x[0].tolist()
        tokens = [t for t in tokens if t not in [self.pad_id, self.bos_id, self.eos_id, self.mask_id]]
        caption = self.tokenizer.decode(tokens)

        return caption

    def generate_onnx(self, image_features: np.ndarray) -> str:
        """Generate caption using ONNX backend."""
        batch_size = 1

        # Start with all masks
        x = np.full((batch_size, self.max_len), self.mask_id, dtype=np.int64)

        # Iterative denoising
        timesteps = np.linspace(1.0, 0.0, self.steps + 1, dtype=np.float32)

        for step in range(self.steps):
            t = np.array([timesteps[step]], dtype=np.float32)
            t_next = timesteps[step + 1]

            # ONNX inference
            outputs = self.onnx_session.run(
                None,
                {
                    'tokens': x,
                    'timestep': t,
                    'image_features': image_features.astype(np.float32),
                }
            )
            logits = outputs[0]  # [batch, seq_len, vocab_size]

            # Sample
            probs = self._softmax(logits / self.temperature)
            sampled = self._multinomial_sample(probs)

            # Update mask (cosine schedule)
            current_mask_rate = 1 - np.cos(t[0] * np.pi / 2)
            next_mask_rate = 1 - np.cos(t_next * np.pi / 2)
            keep_mask_prob = np.clip(next_mask_rate / max(current_mask_rate, 1e-8), 0, 1)

            is_masked = (x == self.mask_id)
            rand = np.random.rand(batch_size, self.max_len)
            keep_mask = rand < keep_mask_prob

            unmask = is_masked & ~keep_mask
            x[unmask] = sampled[unmask]

        # Decode
        tokens = x[0].tolist()
        tokens = [t for t in tokens if t not in [self.pad_id, self.bos_id, self.eos_id, self.mask_id]]
        caption = self.tokenizer.decode(tokens)

        return caption

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _multinomial_sample(probs: np.ndarray) -> np.ndarray:
        """Sample from multinomial distribution."""
        batch, seq_len, vocab_size = probs.shape
        sampled = np.zeros((batch, seq_len), dtype=np.int64)

        for b in range(batch):
            for s in range(seq_len):
                sampled[b, s] = np.random.choice(vocab_size, p=probs[b, s])

        return sampled

    def generate(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Generate caption from webcam frame.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            (caption, inference_time_seconds)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Extract features
        start = time.perf_counter()
        image_features = self.extract_features(pil_image)
        feature_time = time.perf_counter() - start

        # Generate caption
        start = time.perf_counter()
        if self.backend == "pytorch":
            caption = self.generate_pytorch(torch.from_numpy(image_features))
        else:
            caption = self.generate_onnx(image_features)
        generation_time = time.perf_counter() - start

        total_time = feature_time + generation_time

        print(f"\nTiming breakdown:")
        print(f"  CLIP features: {feature_time:.3f}s")
        print(f"  Caption generation: {generation_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")

        return caption, total_time


def load_pytorch_model(checkpoint_path: str, device: str):
    """Load PyTorch model from checkpoint."""
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    decoder_config = checkpoint['decoder_config']
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)

    decoder = DiffusionTransformer(decoder_config)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.to(device)
    decoder.eval()

    return decoder, checkpoint.get('train_config', {})


def load_onnx_model(onnx_path: str, config_path: str):
    """Load ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

    print(f"Loading ONNX model: {onnx_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    print(f"  Using provider: {session.get_providers()[0]}")

    return session, config


def draw_caption_overlay(frame: np.ndarray, caption: str, inference_time: float) -> np.ndarray:
    """Draw caption and info overlay on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw semi-transparent black bar at bottom
    bar_height = 100
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Draw caption text (word wrap if needed)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)

    # Word wrap caption
    max_width = w - 40
    words = caption.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_w, text_h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    # Draw lines
    y_offset = h - bar_height + 30
    for line in lines:
        cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness)
        y_offset += 30

    # Draw timing info
    timing_text = f"{inference_time:.2f}s"
    cv2.putText(frame, timing_text, (w - 100, h - 20), font, 0.6, (100, 255, 100), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Live webcam captioning demo")
    parser.add_argument("--checkpoint", type=str, help="PyTorch checkpoint path")
    parser.add_argument("--onnx", type=str, help="ONNX model path")
    parser.add_argument("--config", type=str, help="Config JSON (for ONNX)")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--max_len", type=int, default=12, help="Max caption length")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")

    args = parser.parse_args()

    # Determine backend
    if args.onnx:
        if not args.config:
            raise ValueError("Must provide --config with --onnx")
        backend = "onnx"
    elif args.checkpoint:
        backend = "pytorch"
    else:
        raise ValueError("Must provide either --checkpoint or --onnx")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Backend: {backend}\n")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Load CLIP vision encoder
    print(f"Loading CLIP model: {args.clip_model}")
    vision_model = CLIPVisionModel.from_pretrained(args.clip_model)
    vision_model.to(device)
    vision_model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # Load decoder
    decoder = None
    onnx_session = None
    train_config = {}

    if backend == "pytorch":
        decoder, train_config = load_pytorch_model(args.checkpoint, device)
        max_len = args.max_len or train_config.get('max_caption_len', 12)
    else:
        onnx_session, config = load_onnx_model(args.onnx, args.config)
        max_len = args.max_len or config.get('max_caption_len', 12)

    # Create caption generator
    print("\nInitializing caption generator...")
    generator = CaptionGenerator(
        decoder=decoder,
        vision_model=vision_model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device,
        max_len=max_len,
        steps=args.steps,
        temperature=args.temperature,
        backend=backend,
        onnx_session=onnx_session,
    )

    # Open webcam
    print(f"\nOpening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return

    print("\n" + "="*70)
    print("WEBCAM CAPTIONING DEMO")
    print("="*70)
    print("Press SPACEBAR to capture and generate caption")
    print("Press 'q' to quit")
    print("="*70 + "\n")

    current_caption = "Press SPACEBAR to generate caption..."
    inference_time = 0.0
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            # Debounce (prevent multiple captures in quick succession)
            current_time = time.time()
            if current_time - last_capture_time > 1.0:
                print("\n📸 Capturing frame...")
                last_capture_time = current_time

                # Generate caption
                current_caption, inference_time = generator.generate(frame)
                print(f"\n✨ Caption: {current_caption}\n")

        # Draw overlay
        display_frame = draw_caption_overlay(frame, current_caption, inference_time)

        # Show frame
        cv2.imshow('Webcam Captioning Demo', display_frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo finished!")


if __name__ == "__main__":
    main()
