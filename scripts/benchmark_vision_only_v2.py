#!/usr/bin/env python3
"""Robust vision-only deployment benchmark: memory, latency, and throughput."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_records(path, n):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('source_subset') in ('supervised', 'unsupervised'):
                records.append(r)
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_tensor(records, data_root, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    images = []
    missing = 0
    for r in records:
        p = Path(data_root) / r['path']
        try:
            with Image.open(p) as img:
                images.append(tfm(img.convert('RGB')))
        except Exception:
            missing += 1
            images.append(torch.zeros(3, img_size, img_size))
    return torch.stack(images), missing


def get_model(name):
    if name == 'resnet50':
        m = models.resnet50()
        m.fc = torch.nn.Linear(m.fc.in_features, 7)
    elif name == 'swin_tiny':
        m = models.swin_t()
        m.head = torch.nn.Linear(m.head.in_features, 7)
    else:
        raise ValueError(name)
    return m


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def summarize(values):
    mean = sum(values) / len(values)
    sd = (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return {
        'mean': mean,
        'sd': sd,
        'cv': sd / mean if mean else None,
        'min': min(values),
        'max': max(values),
        'values': values,
    }


def benchmark_model(name, checkpoint, images, device, batch_sizes, n_single, repeats):
    model = get_model(name)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state)
    model = model.to(device).eval()
    torch.backends.cudnn.benchmark = True

    warmup_batch = min(32, len(images))
    with torch.inference_mode():
        for _ in range(10):
            _ = model(images[:warmup_batch].to(device))
    sync()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    with torch.inference_mode():
        for i in range(min(n_single, len(images))):
            sync()
            t0 = time.perf_counter()
            x = images[i:i + 1].to(device, non_blocking=True)
            _ = model(x)
            sync()
            latencies.append((time.perf_counter() - t0) * 1000)

    throughput = {}
    with torch.inference_mode():
        for bs in batch_sizes:
            n_batches = max(1, len(images) // bs)
            vals = []
            for _ in range(repeats):
                sync()
                t0 = time.perf_counter()
                for b in range(n_batches):
                    x = images[b * bs:(b + 1) * bs].to(device, non_blocking=True)
                    _ = model(x)
                sync()
                vals.append((n_batches * bs) / (time.perf_counter() - t0))
            throughput[str(bs)] = summarize(vals)

    sustained_bs = max(batch_sizes)
    n_batches = max(1, len(images) // sustained_bs)
    sustained = []
    with torch.inference_mode():
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            for b in range(n_batches):
                x = images[b * sustained_bs:(b + 1) * sustained_bs].to(device, non_blocking=True)
                _ = model(x)
            sync()
            sustained.append((n_batches * sustained_bs) / (time.perf_counter() - t0))

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    result = {
        'model': name,
        'checkpoint': checkpoint,
        'peak_memory_gb': peak_memory,
        'allocated_memory_gb_after_benchmark': allocated_memory,
        'single_image_latency_ms_including_h2d': summarize(latencies),
        'single_image_latency_ms_percentiles': {
            'p50': percentile(latencies, 0.50),
            'p90': percentile(latencies, 0.90),
            'p95': percentile(latencies, 0.95),
            'p99': percentile(latencies, 0.99),
        },
        'batch_throughput_images_per_second': throughput,
        'sustained_batch_size': sustained_bs,
        'sustained_images_per_second': summarize(sustained),
    }
    del model
    torch.cuda.empty_cache()
    sync()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='work/splits/test.jsonl')
    ap.add_argument('--data-root', default='data/raw')
    ap.add_argument('--img-size', type=int, default=448)
    ap.add_argument('--n-images', type=int, default=512)
    ap.add_argument('--models', nargs='+', default=['resnet50', 'swin_tiny'])
    ap.add_argument('--checkpoint', action='append', default=[], help='model=/path/to/model.pt')
    ap.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32, 64])
    ap.add_argument('--n-single', type=int, default=200)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        if '=' not in item:
            raise ValueError(f'--checkpoint must be model=/path, got {item}')
        name, path = item.split('=', 1)
        checkpoints[name] = path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    records = load_records(args.test, args.n_images)
    images, missing = build_tensor(records, args.data_root, args.img_size)
    results = {
        'metadata': {
            'test_file': args.test,
            'n_images_requested': args.n_images,
            'n_images_used': len(records),
            'n_images_failed_to_load': missing,
            'img_size': args.img_size,
            'device': str(device),
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'repeats': args.repeats,
            'single_image_samples': min(args.n_single, len(images)),
            'latency_scope': 'CPU tensor to GPU copy plus forward pass',
        },
        'models': {},
    }
    for name in args.models:
        print(f'Benchmarking {name} at {args.img_size}px...', flush=True)
        results['models'][name] = benchmark_model(
            name, checkpoints.get(name), images, device,
            args.batch_sizes, args.n_single, args.repeats,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
