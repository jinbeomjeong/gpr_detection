"""Benchmark image decoding paths for GPR training and inference inputs.

Measures, per image and single-threaded (pinned to the E-cores by default):
  * AIHub B-scan JPG (256x256 gray, inside zip files): zip read, in-memory decode,
    disk read + decode (samples copied under the output directory), uint8 caches.
  * Morocco JPG (224x224 gray): in-memory decode, disk read + decode, uint8 caches.
  * 4TU radargrams: raw PNG decode vs ``np.load`` vs ``np.load(mmap_mode="r")``.
  * GPU JPEG decoding with nvjpeg (``torchvision.io.decode_jpeg(device="cuda")``),
    one image at a time and in batches, with and without host/device copies.
  * DataLoader throughput with 0/2/4 worker processes as a reference.

Every decoder output is compared with the OpenCV grayscale decode of the same file.
The source data under GPR_Data is only read; nothing is extracted there.

Outputs (default ``runs/bench/decode_<YYYYmmdd_HHMMSS>/``):
  results.csv   one row per (dataset, stage, method) timing
  results.json  conditions, system load, correctness, nvjpeg init, worker runs
  summary.md    tables and epoch estimates for docs/experiments/decode_benchmark.md
  samples/      copied sample files and uint8 caches used for the disk stages

Run (PowerShell, from the project root):
  conda run -n gpr-dev python -I scripts/bench/decode_bench.py
  conda run -n gpr-detr python -I scripts/bench/decode_bench.py --no-gpu --threads 0
"""

import argparse
import csv
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
import PIL
import psutil
import torch
import torchvision
from PIL import Image, features
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image, decode_jpeg

try:  # Only the gpr-detr env ships simplejpeg (pulled in by rfdetr[train]).
    import simplejpeg
except ImportError:
    simplejpeg = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(r"C:\workspace\GPR_Data")
AIHUB_SOURCE_DIR = Path(
    "261.3D 지표투과레이더 탐사 데이터",
    "01-1.정식개방데이터",
    "Training",
    "01.원천데이터",
)
AIHUB_BSCAN_ZIP_PATTERN = "TS_B scan_*.zip"
MOROCCO_IMAGE_DIR = Path(
    "모로코_관로공동GPR영상_가공본_261016/03_yolo_dataset/images/train"
)
FOURTU_DIR = Path("4TU_네덜란드_매설관GPR_가공본_261015")
FOURTU_PNG_SUBDIR = "03_radargrams_png_raw"
FOURTU_NPY_SUBDIR = "05_radargrams_npy"

DEFAULT_SAMPLE_COUNTS = {"aihub": 500, "morocco": 500, "4tu": 200}
DATASET_NAMES = tuple(DEFAULT_SAMPLE_COUNTS)
# Bits 16-27 are the twelve E-cores of the i7-14700 (P-core threads are 0-15).
DEFAULT_AFFINITY_MASK = "0x0FFF0000"
GPU_BATCH_SIZES = (32, 128)
WORKER_BATCH_SIZE = 32
# One DataLoader epoch walks the sample list this many times so that an epoch
# lasts long enough (about a second) to time against worker scheduling noise.
WORKER_EPOCH_PASSES = 4
ZIP_OPEN_REPEATS = 3
LOAD_SAMPLE_INTERVAL_S = 2.0
LOAD_SNAPSHOT_INTERVAL_S = 1.0
TOP_PROCESS_COUNT = 5

# Image counts for the one-epoch estimates (project data audit, 2026-10-29).
EPOCH_IMAGE_COUNTS = {
    "aihub": {"AIHub 관로 (~5.6k)": 5_600, "AIHub B-scan 전체": 74_838},
    "morocco": {"모로코 전체": 2_239},
    "4tu": {"4TU 전체": 759},
}


class Stage(StrEnum):
    """Which part of the input path a timing row covers."""

    ZIP_READ = "zip_read"
    DISK_READ = "disk_read"
    MEMORY_DECODE = "memory_decode"
    DISK_READ_DECODE = "disk_read_decode"
    CACHE_READ = "cache_read"
    GPU_DECODE = "gpu_decode"


@dataclass(frozen=True)
class BenchConfig:
    """Command-line settings shared by every benchmark section."""

    datasets: tuple[str, ...]
    sample_counts: dict[str, int]
    repeats: int
    warmup: int
    worker_counts: tuple[int, ...]
    out_dir: Path
    data_root: Path
    seed: int
    affinity_mask: str
    use_gpu: bool


@dataclass(frozen=True)
class ZipMember:
    """One file inside an AIHub zip archive."""

    zip_path: Path
    name: str


@dataclass
class UnitTimings:
    """Raw timings of one measurement: ``per_unit_ns[repeat, unit]``."""

    per_unit_ns: np.ndarray
    warmup_s: float
    images_per_unit: int = 1


@dataclass
class TimingRow:
    """One summarized timing, written as a CSV row."""

    dataset: str
    stage: str
    method: str
    n_images: int
    repeats: int
    median_ms: float
    p5_ms: float
    p95_ms: float
    mean_ms: float
    images_per_s: float
    warmup_ms_per_image: float
    note: str = ""


@dataclass
class SectionResult:
    """Everything one dataset section produced."""

    rows: list[TimingRow] = field(default_factory=list)
    correctness: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decoders: bytes -> 2-D uint8 gray array (or BGR array for the Ultralytics path)
# ---------------------------------------------------------------------------


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    # bytearray keeps torch.frombuffer from warning about a read-only buffer.
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def decode_cv2_gray(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def decode_cv2_color_bgr(data: bytes) -> np.ndarray:
    """Ultralytics default: 3-channel BGR, no gray conversion."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_cv2_color_to_gray(data: bytes) -> np.ndarray:
    return cv2.cvtColor(decode_cv2_color_bgr(data), cv2.COLOR_BGR2GRAY)


def decode_cv2_unchanged_first_channel(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return image if image.ndim == 2 else image[..., 0]


def decode_pil_gray(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("L"))


def decode_pil_draft_gray(data: bytes) -> np.ndarray:
    """PIL with ``draft('L', size)``: full size, so only the colour mode is hinted."""
    with Image.open(io.BytesIO(data)) as image:
        image.draft("L", image.size)
        return np.asarray(image.convert("L"))


def decode_torchvision_jpeg_gray(data: bytes) -> np.ndarray:
    return decode_jpeg(_bytes_to_tensor(data), mode=ImageReadMode.GRAY).numpy()[0]


def decode_torchvision_image_gray(data: bytes) -> np.ndarray:
    return decode_image(_bytes_to_tensor(data), mode=ImageReadMode.GRAY).numpy()[0]


def decode_simplejpeg_gray(data: bytes) -> np.ndarray:
    return simplejpeg.decode_jpeg(data, colorspace="GRAY")[..., 0]


def decode_npy_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _jpeg_decoders() -> dict[str, Callable[[bytes], np.ndarray]]:
    decoders = {
        "cv2_gray": decode_cv2_gray,
        "cv2_color_to_gray": decode_cv2_color_to_gray,
        "cv2_color_bgr": decode_cv2_color_bgr,
        "pil_gray": decode_pil_gray,
        "pil_draft_gray": decode_pil_draft_gray,
        "tv_decode_jpeg_gray": decode_torchvision_jpeg_gray,
        "tv_decode_image_gray": decode_torchvision_image_gray,
    }
    if simplejpeg is not None:
        decoders["simplejpeg_gray"] = decode_simplejpeg_gray
    return decoders


PNG_DECODERS: dict[str, Callable[[bytes], np.ndarray]] = {
    "cv2_gray": decode_cv2_gray,
    "cv2_unchanged_ch0": decode_cv2_unchanged_first_channel,
    "pil_gray": decode_pil_gray,
    "tv_decode_image_gray": decode_torchvision_image_gray,
}


# ---------------------------------------------------------------------------
# File readers used by the DataLoader workers (looked up by name after spawn)
# ---------------------------------------------------------------------------


def read_file_cv2_gray(path: Path) -> np.ndarray:
    return decode_cv2_gray(path.read_bytes())


def read_file_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


FILE_READERS: dict[str, Callable[[Path], np.ndarray]] = {
    "cv2_gray": read_file_cv2_gray,
    "npy_load": read_file_npy,
}


class FileImageDataset(Dataset):
    """Reads one file per item; repeats the list ``passes`` times per epoch."""

    def __init__(self, paths: Sequence[Path], reader_name: str, passes: int) -> None:
        self.paths = list(paths)
        self.reader_name = reader_name
        self.passes = passes

    def __len__(self) -> int:
        return len(self.paths) * self.passes

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index % len(self.paths)]
        return torch.from_numpy(FILE_READERS[self.reader_name](path))


class ZipImageDataset(Dataset):
    """Reads JPG bytes from zip archives, keeping one open handle per archive."""

    def __init__(self, members: Sequence[ZipMember], passes: int) -> None:
        self.members = list(members)
        self.passes = passes
        self._open_archives: dict[Path, zipfile.ZipFile] = {}

    def __getstate__(self) -> dict:
        # Open zip handles cannot cross the spawn boundary; each worker reopens.
        state = self.__dict__.copy()
        state["_open_archives"] = {}
        return state

    def __len__(self) -> int:
        return len(self.members) * self.passes

    def __getitem__(self, index: int) -> torch.Tensor:
        member = self.members[index % len(self.members)]
        archive = self._open_archives.get(member.zip_path)
        if archive is None:
            archive = zipfile.ZipFile(member.zip_path)
            self._open_archives[member.zip_path] = archive
        return torch.from_numpy(decode_cv2_gray(archive.read(member.name)))


def collate_as_list(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    # 4TU radargrams differ in width, so the batch is not stacked.
    return batch


def init_worker(_worker_id: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# System state: affinity, CPU/GPU load, versions
# ---------------------------------------------------------------------------


def apply_affinity(mask_text: str) -> list[int] | None:
    """Pins this process (and children it spawns later) to the cores in the mask."""
    if mask_text.lower() == "none":
        return None
    mask = int(mask_text, 16)
    available = range(psutil.cpu_count(logical=True))
    cores = [core for core in available if mask >> core & 1]
    if not cores:
        raise ValueError(f"affinity mask {mask_text} selects no existing core")
    process = psutil.Process()
    process.cpu_affinity(cores)
    return process.cpu_affinity()


def query_gpu_state() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    utilization, used_mib, total_mib = (float(v) for v in output.split(","))
    return {
        "util_pct": utilization,
        "mem_used_mib": used_mib,
        "mem_total_mib": total_mib,
    }


def snapshot_system_load(pinned_cores: list[int] | None) -> dict:
    """System CPU (total and on the pinned cores) over 1 s, top processes, GPU."""
    processes = list(psutil.process_iter(["name"]))
    for process in processes:
        try:
            process.cpu_percent(None)
        except psutil.Error:
            continue
    before = psutil.cpu_times(percpu=True)
    time.sleep(LOAD_SNAPSHOT_INTERVAL_S)
    per_core = busy_percent_per_core(before, psutil.cpu_times(percpu=True))
    top = []
    for process in processes:
        try:
            top.append((process.cpu_percent(None), process.info["name"], process.pid))
        except psutil.Error:
            continue
    top.sort(reverse=True)
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cpu_total_pct": round(statistics.fmean(per_core), 1),
        "cpu_pinned_pct": _mean_on_cores(per_core, pinned_cores),
        "cpu_per_core_pct": per_core,
        "top_processes": [
            {"name": name, "pid": pid, "cpu_pct_of_one_core": round(pct, 1)}
            for pct, name, pid in top[:TOP_PROCESS_COUNT]
        ],
        "gpu": query_gpu_state(),
    }


def busy_percent_per_core(before: Sequence, after: Sequence) -> list[float]:
    """Busy share of each core between two ``psutil.cpu_times(percpu=True)`` calls.

    Computed from explicit deltas instead of ``psutil.cpu_percent`` so that the
    snapshot and the background sampler do not share psutil's global baseline.
    """
    busy = []
    for start, end in zip(before, after, strict=True):
        total_s = sum(end) - sum(start)
        idle_s = end.idle - start.idle
        busy.append(round(100.0 * (1.0 - idle_s / total_s), 1) if total_s > 0 else 0.0)
    return busy


def _mean_on_cores(per_core: list[float], cores: list[int] | None) -> float | None:
    if not cores:
        return None
    return round(statistics.fmean(per_core[core] for core in cores), 1)


class LoadSampler:
    """Samples system CPU and GPU utilization in the background during the run."""

    def __init__(self, pinned_cores: list[int] | None) -> None:
        self.pinned_cores = pinned_cores
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return self.summary()

    def _run(self) -> None:
        previous = psutil.cpu_times(percpu=True)
        while not self._stop.wait(LOAD_SAMPLE_INTERVAL_S):
            current = psutil.cpu_times(percpu=True)
            per_core = busy_percent_per_core(previous, current)
            previous = current
            self.samples.append(
                {
                    "cpu_total_pct": statistics.fmean(per_core),
                    "cpu_pinned_pct": _mean_on_cores(per_core, self.pinned_cores),
                    "gpu_util_pct": query_gpu_state().get("util_pct"),
                }
            )

    def summary(self) -> dict:
        result: dict = {"n_samples": len(self.samples)}
        for key in ("cpu_total_pct", "cpu_pinned_pct", "gpu_util_pct"):
            values = [s[key] for s in self.samples if s[key] is not None]
            if values:
                result[key] = {
                    "mean": round(statistics.fmean(values), 1),
                    "min": round(min(values), 1),
                    "max": round(max(values), 1),
                }
        return result


def collect_versions() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "pillow_libjpeg_turbo": features.check_feature("libjpeg_turbo"),
        "pillow_jpeg_version": features.version("jpg"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "simplejpeg": getattr(simplejpeg, "__version__", None),
        "gpu": gpu_name,
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_units(
    func: Callable, units: Sequence, repeats: int, warmup: int, images_per_unit: int = 1
) -> UnitTimings:
    """Times ``func(unit)`` for every unit, after ``warmup`` untimed full passes."""
    warmup_start = time.perf_counter()
    for _ in range(warmup):
        for unit in units:
            func(unit)
    warmup_s = (time.perf_counter() - warmup_start) / max(warmup, 1)
    per_unit_ns = np.empty((repeats, len(units)), dtype=np.int64)
    for repeat in range(repeats):
        for index, unit in enumerate(units):
            start = time.perf_counter_ns()
            func(unit)
            per_unit_ns[repeat, index] = time.perf_counter_ns() - start
    return UnitTimings(per_unit_ns, warmup_s, images_per_unit)


def summarize(
    dataset: str, stage: Stage, method: str, timings: UnitTimings, note: str = ""
) -> TimingRow:
    per_image_ms = timings.per_unit_ns / 1e6 / timings.images_per_unit
    pass_seconds = timings.per_unit_ns.sum(axis=1) / 1e9
    repeats, n_units = timings.per_unit_ns.shape
    n_images = n_units * timings.images_per_unit
    row = TimingRow(
        dataset=dataset,
        stage=str(stage),
        method=method,
        n_images=n_images,
        repeats=repeats,
        median_ms=float(np.median(per_image_ms)),
        p5_ms=float(np.percentile(per_image_ms, 5)),
        p95_ms=float(np.percentile(per_image_ms, 95)),
        mean_ms=float(per_image_ms.mean()),
        images_per_s=float(n_images / np.median(pass_seconds)),
        warmup_ms_per_image=timings.warmup_s * 1e3 / n_images,
        note=note,
    )
    print(
        f"  {dataset:8s} {row.stage:17s} {method:34s} "
        f"median {row.median_ms:7.3f} ms  {row.images_per_s:9.1f} img/s",
        flush=True,
    )
    return row


def compare_outputs(
    reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray]
) -> dict:
    """Max abs difference and share of differing pixels against the reference."""
    max_abs_diff = 0
    differing_pixels = 0
    total_pixels = 0
    shape_mismatches = 0
    for expected, actual in zip(reference, candidate, strict=True):
        if actual.ndim == 3:
            actual = actual[..., 0]
        if expected.shape != actual.shape:
            shape_mismatches += 1
            continue
        difference = np.abs(expected.astype(np.int32) - actual.astype(np.int32))
        max_abs_diff = max(max_abs_diff, int(difference.max()))
        differing_pixels += int(np.count_nonzero(difference))
        total_pixels += difference.size
    return {
        "max_abs_diff": max_abs_diff,
        "differing_pixel_ratio": differing_pixels / total_pixels
        if total_pixels
        else None,
        "shape_mismatches": shape_mismatches,
        "dtype": str(candidate[0].dtype),
    }


def _correctness_entry(
    dataset: str, method: str, reference_name: str, stats: dict
) -> dict:
    return {"dataset": dataset, "method": method, "reference": reference_name, **stats}


# ---------------------------------------------------------------------------
# Sampling and copying
# ---------------------------------------------------------------------------


def sample_aihub_members(
    source_dir: Path, count: int, rng: random.Random
) -> list[ZipMember]:
    """Draws ``count`` JPGs across the Training B-scan zips, proportional to size."""
    population: list[ZipMember] = []
    for zip_path in sorted(source_dir.glob(AIHUB_BSCAN_ZIP_PATTERN)):
        with zipfile.ZipFile(zip_path) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
        population.extend(ZipMember(zip_path, name) for name in names)
    return rng.sample(population, count)


def read_zip_members(members: Sequence[ZipMember]) -> list[bytes]:
    archives: dict[Path, zipfile.ZipFile] = {}
    try:
        payloads = []
        for member in members:
            if member.zip_path not in archives:
                archives[member.zip_path] = zipfile.ZipFile(member.zip_path)
            payloads.append(archives[member.zip_path].read(member.name))
        return payloads
    finally:
        for archive in archives.values():
            archive.close()


def write_samples(
    payloads: Sequence[bytes], sample_dir: Path, suffix: str
) -> list[Path]:
    """Writes ASCII-named copies so every reader (incl. cv2.imread) can open them."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, payload in enumerate(payloads):
        path = sample_dir / f"{index:04d}{suffix}"
        path.write_bytes(payload)
        paths.append(path)
    return paths


def write_sample_manifest(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Benchmark building blocks shared by the datasets
# ---------------------------------------------------------------------------


def bench_memory_decoders(
    dataset: str,
    payloads: Sequence[bytes],
    decoders: dict[str, Callable[[bytes], np.ndarray]],
    config: BenchConfig,
    section: SectionResult,
) -> None:
    """Decodes preloaded bytes with every decoder; checks outputs against cv2_gray."""
    reference = [decode_cv2_gray(payload) for payload in payloads]
    for name, decoder in decoders.items():
        timings = time_units(decoder, payloads, config.repeats, config.warmup)
        section.rows.append(summarize(dataset, Stage.MEMORY_DECODE, name, timings))
        outputs = [decoder(payload) for payload in payloads]
        stats = compare_outputs(reference, outputs)
        section.correctness.append(_correctness_entry(dataset, name, "cv2_gray", stats))


def bench_disk_readers(
    dataset: str,
    paths: Sequence[Path],
    decoders: dict[str, Callable[[bytes], np.ndarray]],
    config: BenchConfig,
    section: SectionResult,
) -> None:
    """Times a plain file read, then file read + decode for each decoder."""
    timings = time_units(Path.read_bytes, paths, config.repeats, config.warmup)
    section.rows.append(summarize(dataset, Stage.DISK_READ, "read_bytes", timings))
    for name, decoder in decoders.items():

        def read_and_decode(path: Path, decoder=decoder) -> np.ndarray:
            return decoder(path.read_bytes())

        timings = time_units(read_and_decode, paths, config.repeats, config.warmup)
        section.rows.append(summarize(dataset, Stage.DISK_READ_DECODE, name, timings))


def bench_uint8_caches(
    dataset: str,
    images: Sequence[np.ndarray],
    cache_dir: Path,
    config: BenchConfig,
    section: SectionResult,
) -> None:
    """Pre-decoded caches: one flat uint8 memmap, and one .npy file per image.

    The flat memmap stores images back to back with (offset, height, width) per
    image, because a dataset may mix image sizes (Morocco has 224 and 336 px).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    flat_path = cache_dir / "flat_uint8.npy"
    total_pixels = sum(image.size for image in images)
    flat = np.lib.format.open_memmap(
        flat_path, mode="w+", dtype=np.uint8, shape=(total_pixels,)
    )
    layout: list[tuple[int, int, int]] = []
    offset = 0
    for image in images:
        flat[offset : offset + image.size] = image.ravel()
        layout.append((offset, *image.shape))
        offset += image.size
    flat.flush()
    del flat
    per_file_paths = []
    for index, image in enumerate(images):
        path = cache_dir / f"{index:04d}.npy"
        np.save(path, image)
        per_file_paths.append(path)

    memmap = np.load(flat_path, mmap_mode="r")

    def read_cached_image(index: int) -> np.ndarray:
        start, height, width = layout[index]
        return np.array(memmap[start : start + height * width]).reshape(height, width)

    shuffled = list(range(len(images)))
    random.Random(config.seed).shuffle(shuffled)
    timings = time_units(read_cached_image, shuffled, config.repeats, config.warmup)
    section.rows.append(
        summarize(
            dataset,
            Stage.CACHE_READ,
            "memmap_uint8_random",
            timings,
            note="flat uint8 memmap + offsets, random order, copied out",
        )
    )
    timings = time_units(read_file_npy, per_file_paths, config.repeats, config.warmup)
    section.rows.append(
        summarize(
            dataset,
            Stage.CACHE_READ,
            "npy_per_image",
            timings,
            note="one uint8 .npy per image (Ultralytics cache='disk' style)",
        )
    )


def bench_zip_reads(
    members: Sequence[ZipMember], config: BenchConfig, section: SectionResult
) -> None:
    """Zip open (central directory) cost and per-member read+inflate cost."""
    open_ms = {}
    for zip_path in sorted({member.zip_path for member in members}):
        timings_ms = []
        for _ in range(ZIP_OPEN_REPEATS):
            start = time.perf_counter()
            with zipfile.ZipFile(zip_path):
                pass
            timings_ms.append((time.perf_counter() - start) * 1e3)
        open_ms[zip_path.name] = round(statistics.median(timings_ms), 2)
    section.details["zip_open_ms_median"] = open_ms

    archives = {path: zipfile.ZipFile(path) for path in {m.zip_path for m in members}}
    try:
        timings = time_units(
            lambda member: archives[member.zip_path].read(member.name),
            members,
            config.repeats,
            config.warmup,
        )
    finally:
        for archive in archives.values():
            archive.close()
    section.rows.append(
        summarize(
            "aihub",
            Stage.ZIP_READ,
            "zipfile_read",
            timings,
            note="handles kept open; deflate inflate included; OS cache warm",
        )
    )


# ---------------------------------------------------------------------------
# GPU (nvjpeg) decoding
# ---------------------------------------------------------------------------


def _synchronized(func: Callable) -> Callable:
    def wrapper(unit):
        result = func(unit)
        torch.cuda.synchronize()
        return result

    return wrapper


def measure_nvjpeg_initialization(tensors: Sequence[torch.Tensor]) -> dict:
    """First-call costs, kept out of the steady-state numbers."""
    start = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    context_s = time.perf_counter() - start

    start = time.perf_counter()
    decode_jpeg(tensors[0], mode=ImageReadMode.GRAY, device="cuda")
    torch.cuda.synchronize()
    first_single_s = time.perf_counter() - start

    start = time.perf_counter()
    decode_jpeg(
        list(tensors[: GPU_BATCH_SIZES[0]]), mode=ImageReadMode.GRAY, device="cuda"
    )
    torch.cuda.synchronize()
    first_batch_s = time.perf_counter() - start
    return {
        "cuda_context_s": round(context_s, 3),
        "nvjpeg_first_single_call_s": round(first_single_s, 3),
        f"nvjpeg_first_batch{GPU_BATCH_SIZES[0]}_call_s": round(first_batch_s, 3),
    }


def bench_gpu_decoding(
    dataset: str, payloads: Sequence[bytes], config: BenchConfig, section: SectionResult
) -> None:
    """nvjpeg single/batched decode vs CPU decode + host-to-device copy."""
    tensors = [_bytes_to_tensor(payload) for payload in payloads]
    if "gpu_init" not in section.details:
        section.details["gpu_init"] = measure_nvjpeg_initialization(tensors)

    def decode_single(tensor: torch.Tensor) -> torch.Tensor:
        return decode_jpeg(tensor, mode=ImageReadMode.GRAY, device="cuda")

    def decode_batch(batch: list[torch.Tensor]) -> list[torch.Tensor]:
        return decode_jpeg(batch, mode=ImageReadMode.GRAY, device="cuda")

    def cpu_decode_then_upload(payload: bytes) -> torch.Tensor:
        return torch.from_numpy(decode_cv2_gray(payload)).to("cuda")

    variants: list[tuple[str, Callable, Sequence, int, str]] = [
        ("nvjpeg_single", decode_single, tensors, 1, "output stays on GPU"),
        (
            "nvjpeg_single+D2H",
            lambda t: decode_single(t).cpu(),
            tensors,
            1,
            "output copied back to host",
        ),
        (
            "cv2_gray+H2D",
            cpu_decode_then_upload,
            payloads,
            1,
            "CPU decode then pageable copy to GPU",
        ),
    ]
    for batch_size in GPU_BATCH_SIZES:
        batches = [
            list(tensors[start : start + batch_size])
            for start in range(0, len(tensors) - batch_size + 1, batch_size)
        ]
        variants.append(
            (
                f"nvjpeg_batch{batch_size}",
                decode_batch,
                batches,
                batch_size,
                "full batches only; output stays on GPU",
            )
        )
        variants.append(
            (
                f"nvjpeg_batch{batch_size}+D2H",
                lambda batch: [image.cpu() for image in decode_batch(batch)],
                batches,
                batch_size,
                "full batches only; output copied to host",
            )
        )

    for name, func, units, images_per_unit, note in variants:
        if not units:
            continue
        timings = time_units(
            _synchronized(func), units, config.repeats, config.warmup, images_per_unit
        )
        section.rows.append(summarize(dataset, Stage.GPU_DECODE, name, timings, note))

    reference = [decode_cv2_gray(payload) for payload in payloads]
    outputs = [decode_single(tensor).cpu().numpy()[0] for tensor in tensors]
    stats = compare_outputs(reference, outputs)
    section.correctness.append(
        _correctness_entry(dataset, "nvjpeg_single", "cv2_gray", stats)
    )


# ---------------------------------------------------------------------------
# DataLoader worker throughput
# ---------------------------------------------------------------------------


def _worker_affinities() -> list[list[int]]:
    affinities = []
    for child in psutil.Process().children(recursive=True):
        try:
            affinities.append(child.cpu_affinity())
        except psutil.Error:
            continue
    return affinities


def bench_dataloader(
    dataset: Dataset, n_images_per_epoch: int, worker_count: int, repeats: int
) -> dict:
    """Epoch throughput for one worker count.

    The first epoch includes spawning the workers and is reported separately.
    """
    loader = DataLoader(
        dataset,
        batch_size=WORKER_BATCH_SIZE,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
        collate_fn=collate_as_list,
        worker_init_fn=init_worker if worker_count > 0 else None,
    )
    epoch_seconds = []
    worker_affinities: list[list[int]] = []
    for epoch in range(repeats + 1):
        start = time.perf_counter()
        for _batch in loader:
            if epoch == 0 and not worker_affinities and worker_count > 0:
                worker_affinities = _worker_affinities()
        epoch_seconds.append(time.perf_counter() - start)
    del loader
    steady = epoch_seconds[1:]
    return {
        "workers": worker_count,
        "first_epoch_s": round(epoch_seconds[0], 3),
        "epoch_s_median": round(statistics.median(steady), 4),
        "images_per_s": round(n_images_per_epoch / statistics.median(steady), 1),
        "worker_affinities": worker_affinities,
    }


def bench_worker_pipelines(
    dataset_name: str,
    pipelines: dict[str, Dataset],
    images_per_epoch: int,
    config: BenchConfig,
    section: SectionResult,
) -> None:
    results = []
    for pipeline_name, dataset in pipelines.items():
        for worker_count in (0, *config.worker_counts):
            result = bench_dataloader(
                dataset, images_per_epoch, worker_count, config.repeats
            )
            result["pipeline"] = pipeline_name
            results.append(result)
            print(
                f"  {dataset_name:8s} workers={worker_count} {pipeline_name:28s} "
                f"{result['images_per_s']:9.1f} img/s "
                f"(first epoch {result['first_epoch_s']} s)",
                flush=True,
            )
    section.details["dataloader"] = results


# ---------------------------------------------------------------------------
# Dataset sections
# ---------------------------------------------------------------------------


def count_shapes(images: Sequence[np.ndarray]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image in images:
        key = "x".join(map(str, image.shape))
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_aihub(config: BenchConfig, rng: random.Random) -> SectionResult:
    section = SectionResult()
    count = config.sample_counts["aihub"]
    print(f"[aihub] sampling {count} JPGs from the Training B-scan zips", flush=True)
    members = sample_aihub_members(config.data_root / AIHUB_SOURCE_DIR, count, rng)
    payloads = read_zip_members(members)
    sample_dir = config.out_dir / "samples" / "aihub"
    paths = write_samples(payloads, sample_dir, ".jpg")
    write_sample_manifest(
        sample_dir / "manifest.csv",
        [
            {"copy": p.name, "zip": m.zip_path.name, "member": m.name, "bytes": len(b)}
            for p, m, b in zip(paths, members, payloads, strict=True)
        ],
    )
    section.details["mean_jpeg_bytes"] = round(statistics.fmean(map(len, payloads)))

    bench_zip_reads(members, config, section)
    bench_memory_decoders("aihub", payloads, _jpeg_decoders(), config, section)
    bench_disk_readers("aihub", paths, _jpeg_decoders(), config, section)
    images = [decode_cv2_gray(payload) for payload in payloads]
    section.details["image_shapes"] = count_shapes(images)
    bench_uint8_caches("aihub", images, sample_dir / "cache", config, section)
    if config.use_gpu:
        bench_gpu_decoding("aihub", payloads, config, section)
    if config.worker_counts:
        bench_worker_pipelines(
            "aihub",
            {
                "zip+cv2_gray": ZipImageDataset(members, WORKER_EPOCH_PASSES),
                "disk_jpg+cv2_gray": FileImageDataset(
                    paths, "cv2_gray", WORKER_EPOCH_PASSES
                ),
            },
            len(paths) * WORKER_EPOCH_PASSES,
            config,
            section,
        )
    return section


def run_morocco(config: BenchConfig, rng: random.Random) -> SectionResult:
    section = SectionResult()
    count = config.sample_counts["morocco"]
    image_dir = config.data_root / MOROCCO_IMAGE_DIR
    population = sorted(image_dir.glob("*.jpg"))
    paths = rng.sample(population, min(count, len(population)))
    print(
        f"[morocco] {len(paths)} of {len(population)} train JPGs (read in place)",
        flush=True,
    )
    payloads = [path.read_bytes() for path in paths]
    section.details["mean_jpeg_bytes"] = round(statistics.fmean(map(len, payloads)))

    bench_memory_decoders("morocco", payloads, _jpeg_decoders(), config, section)
    bench_disk_readers("morocco", paths, _jpeg_decoders(), config, section)
    images = [decode_cv2_gray(payload) for payload in payloads]
    section.details["image_shapes"] = count_shapes(images)
    cache_dir = config.out_dir / "samples" / "morocco" / "cache"
    bench_uint8_caches("morocco", images, cache_dir, config, section)
    if config.use_gpu:
        bench_gpu_decoding("morocco", payloads, config, section)
    if config.worker_counts:
        bench_worker_pipelines(
            "morocco",
            {
                "disk_jpg+cv2_gray": FileImageDataset(
                    paths, "cv2_gray", WORKER_EPOCH_PASSES
                )
            },
            len(paths) * WORKER_EPOCH_PASSES,
            config,
            section,
        )
    return section


def sample_4tu_pairs(
    root: Path, count: int, rng: random.Random
) -> list[tuple[Path, Path]]:
    """Matching (png, npy) pairs by their path relative to the radargram folders."""
    png_root = root / FOURTU_PNG_SUBDIR
    npy_root = root / FOURTU_NPY_SUBDIR
    pairs = []
    for png_path in sorted(png_root.rglob("*.png")):
        npy_path = (npy_root / png_path.relative_to(png_root)).with_suffix(".npy")
        if npy_path.exists():
            pairs.append((png_path, npy_path))
    return rng.sample(pairs, min(count, len(pairs)))


def bench_npy_loads(
    npy_paths: Sequence[Path], config: BenchConfig, section: SectionResult
) -> None:
    def load_mmap_open_only(path: Path) -> np.ndarray:
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def load_mmap_materialized(path: Path) -> np.ndarray:
        return np.array(np.load(path, mmap_mode="r", allow_pickle=False))

    readers = {
        "np_load": read_file_npy,
        "np_load_mmap_open_only": load_mmap_open_only,
        "np_load_mmap+copy": load_mmap_materialized,
    }
    notes = {
        "np_load_mmap_open_only": "lazy: no pixel is read yet",
        "np_load_mmap+copy": "mmap then np.array (all pixels read)",
    }
    for name, reader in readers.items():
        timings = time_units(reader, npy_paths, config.repeats, config.warmup)
        section.rows.append(
            summarize("4tu", Stage.DISK_READ_DECODE, name, timings, notes.get(name, ""))
        )
    full = [read_file_npy(path) for path in npy_paths]
    mapped = [load_mmap_materialized(path) for path in npy_paths]
    section.correctness.append(
        _correctness_entry(
            "4tu", "np_load_mmap+copy", "np_load", compare_outputs(full, mapped)
        )
    )


def run_4tu(config: BenchConfig, rng: random.Random) -> SectionResult:
    section = SectionResult()
    pairs = sample_4tu_pairs(
        config.data_root / FOURTU_DIR, config.sample_counts["4tu"], rng
    )
    print(f"[4tu] {len(pairs)} PNG/npy pairs (read in place)", flush=True)
    png_paths = [png for png, _ in pairs]
    npy_paths = [npy for _, npy in pairs]
    png_payloads = [path.read_bytes() for path in png_paths]
    npy_payloads = [path.read_bytes() for path in npy_paths]
    section.details["mean_png_bytes"] = round(statistics.fmean(map(len, png_payloads)))
    section.details["mean_npy_bytes"] = round(statistics.fmean(map(len, npy_payloads)))

    bench_memory_decoders("4tu", png_payloads, PNG_DECODERS, config, section)
    timings = time_units(decode_npy_bytes, npy_payloads, config.repeats, config.warmup)
    section.rows.append(
        summarize("4tu", Stage.MEMORY_DECODE, "npy_from_bytes", timings)
    )
    bench_disk_readers(
        "4tu", png_paths, {"png_cv2_gray": decode_cv2_gray}, config, section
    )
    bench_npy_loads(npy_paths, config, section)
    if config.worker_counts:
        bench_worker_pipelines(
            "4tu",
            {
                "disk_png+cv2_gray": FileImageDataset(
                    png_paths, "cv2_gray", WORKER_EPOCH_PASSES
                ),
                "disk_npy+np_load": FileImageDataset(
                    npy_paths, "npy_load", WORKER_EPOCH_PASSES
                ),
            },
            len(pairs) * WORKER_EPOCH_PASSES,
            config,
            section,
        )
    return section


SECTION_RUNNERS: dict[str, Callable[[BenchConfig, random.Random], SectionResult]] = {
    "aihub": run_aihub,
    "morocco": run_morocco,
    "4tu": run_4tu,
}


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def write_results_csv(path: Path, rows: Sequence[TimingRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _markdown_row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _timing_table(rows: Sequence[TimingRow]) -> list[str]:
    header = ["stage", "method", "median ms", "p5 ms", "p95 ms", "img/s"]
    header += ["warmup ms/img", "note"]
    lines = [_markdown_row(header), "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        cells = [row.stage, row.method, f"{row.median_ms:.3f}", f"{row.p5_ms:.3f}"]
        cells += [f"{row.p95_ms:.3f}", f"{row.images_per_s:.0f}"]
        cells += [f"{row.warmup_ms_per_image:.3f}", row.note]
        lines.append(_markdown_row(cells))
    return lines


def _epoch_paths(rows: Sequence[TimingRow]) -> list[tuple[str, float]]:
    """(path label, ms per image) for every complete input path of a dataset.

    For zip data a complete path is the zip read plus an in-memory decode, so
    those two medians are added; disk and cache rows are complete on their own.
    """
    zip_read = next((row for row in rows if row.stage == Stage.ZIP_READ), None)
    paths = []
    for row in rows:
        if row.stage in (Stage.DISK_READ, Stage.ZIP_READ):
            continue
        label = f"{row.stage}/{row.method}"
        if zip_read is not None and row.stage in (
            Stage.MEMORY_DECODE,
            Stage.GPU_DECODE,
        ):
            paths.append((f"zip_read + {label}", zip_read.median_ms + row.median_ms))
        else:
            paths.append((label, row.median_ms))
    return paths


def _epoch_estimate_lines(
    dataset: str, rows: Sequence[TimingRow], loader_runs: Sequence[dict]
) -> list[str]:
    """Epoch time = per-image median x image count (single thread), or count / img/s."""
    counts = EPOCH_IMAGE_COUNTS.get(dataset, {})
    if not counts:
        return []
    header = "| path | ms/img | " + " | ".join(f"{name} (s)" for name in counts) + " |"
    lines = [header, "|---|---:|" + "---:|" * len(counts)]
    for label, ms_per_image in _epoch_paths(rows):
        seconds = " | ".join(f"{ms_per_image * n / 1e3:.1f}" for n in counts.values())
        lines.append(f"| {label} | {ms_per_image:.3f} | {seconds} |")
    for run in loader_runs:
        ms_per_image = 1e3 / run["images_per_s"]
        seconds = " | ".join(f"{n / run['images_per_s']:.1f}" for n in counts.values())
        label = f"DataLoader {run['pipeline']} workers={run['workers']}"
        lines.append(f"| {label} | {ms_per_image:.3f} | {seconds} |")
    return lines


def write_summary_markdown(path: Path, report: dict, rows: Sequence[TimingRow]) -> None:
    lines = [f"# decode benchmark {report['started']}", ""]
    lines.append(f"- command: `{report['command']}`")
    lines.append(f"- affinity: {report['affinity_cores']}")
    lines.append(f"- load at start: {_load_line(report['load_start'])}")
    lines.append(f"- load at end: {_load_line(report['load_end'])}")
    lines.append(f"- load during run: {report['load_during']}")
    lines.append("")
    for dataset in report["config"]["datasets"]:
        dataset_rows = [row for row in rows if row.dataset == dataset]
        details = report["sections"][dataset]["details"]
        loader_runs = details.get("dataloader", [])
        lines += [f"## {dataset}", "", *_timing_table(dataset_rows), ""]
        lines += [*_epoch_estimate_lines(dataset, dataset_rows, loader_runs), ""]
        for key, value in details.items():
            if key != "dataloader":
                lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
        for run in loader_runs:
            lines.append(
                f"- DataLoader {run['pipeline']} workers={run['workers']}: "
                f"{run['images_per_s']} img/s, first epoch {run['first_epoch_s']} s"
            )
        lines.append("")
    lines += [*_correctness_table(report), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _correctness_table(report: dict) -> list[str]:
    header = ["dataset", "method", "reference", "max abs diff", "differing ratio"]
    header += ["shape mismatch"]
    lines = ["## correctness (vs reference)", "", _markdown_row(header)]
    lines.append("|---|---|---|---:|---:|---:|")
    for dataset in report["config"]["datasets"]:
        for entry in report["sections"][dataset]["correctness"]:
            ratio = entry["differing_pixel_ratio"]
            ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
            cells = [entry["dataset"], entry["method"], entry["reference"]]
            cells += [entry["max_abs_diff"], ratio_text, entry["shape_mismatches"]]
            lines.append(_markdown_row(cells))
    return lines


def _load_line(snapshot: dict) -> str:
    gpu = snapshot["gpu"]
    top = ", ".join(
        f"{p['name']}({p['cpu_pct_of_one_core']})" for p in snapshot["top_processes"]
    )
    return (
        f"CPU total {snapshot['cpu_total_pct']} %, "
        f"pinned cores {snapshot['cpu_pinned_pct']} %, "
        f"GPU {gpu.get('util_pct')} % / {gpu.get('mem_used_mib')} MiB; top: {top}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_worker_counts(text: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in text.split(",") if part.strip())
    return tuple(count for count in counts if count > 0)


def _parse_datasets(text: str) -> tuple[str, ...]:
    if text == "all":
        return DATASET_NAMES
    names = tuple(part.strip() for part in text.split(","))
    unknown = [name for name in names if name not in DATASET_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown dataset(s): {unknown}")
    return names


def parse_args(argv: Sequence[str] | None = None) -> BenchConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset",
        type=_parse_datasets,
        default=DATASET_NAMES,
        help="all, or comma list of aihub,morocco,4tu (default: all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="samples per dataset (default: aihub 500, morocco 500, 4tu 200)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed passes (default 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="untimed passes (default 1)"
    )
    parser.add_argument(
        "--threads",
        default="2,4",
        help="DataLoader worker counts for the throughput reference, "
        "comma list; 0 skips the DataLoader runs (default 2,4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default runs/bench/decode_<timestamp>)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument(
        "--affinity",
        default=DEFAULT_AFFINITY_MASK,
        help="hex core mask or 'none' (default 0x0FFF0000 = E-cores)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="skip nvjpeg runs")
    args = parser.parse_args(argv)

    if args.n is None:
        sample_counts = dict(DEFAULT_SAMPLE_COUNTS)
    else:
        sample_counts = dict.fromkeys(DATASET_NAMES, args.n)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or PROJECT_ROOT / "runs" / "bench" / f"decode_{timestamp}"
    return BenchConfig(
        datasets=args.dataset,
        sample_counts=sample_counts,
        repeats=args.repeats,
        warmup=args.warmup,
        worker_counts=_parse_worker_counts(args.threads),
        out_dir=out_dir.resolve(),
        data_root=args.data_root,
        seed=args.seed,
        affinity_mask=args.affinity,
        use_gpu=not args.no_gpu and torch.cuda.is_available(),
    )


def main() -> None:
    config = parse_args()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    affinity_cores = apply_affinity(config.affinity_mask)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    print(f"output: {config.out_dir}\naffinity: {affinity_cores}", flush=True)

    started = datetime.now()
    load_start = snapshot_system_load(affinity_cores)
    sampler = LoadSampler(affinity_cores)
    sampler.start()
    rng = random.Random(config.seed)
    sections: dict[str, SectionResult] = {}
    for dataset in config.datasets:
        sections[dataset] = SECTION_RUNNERS[dataset](config, rng)
    load_during = sampler.stop()
    load_end = snapshot_system_load(affinity_cores)

    rows = [row for section in sections.values() for row in section.rows]
    report = {
        "started": started.isoformat(timespec="seconds"),
        "elapsed_s": round((datetime.now() - started).total_seconds(), 1),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "config": {
            **asdict(config),
            "out_dir": str(config.out_dir),
            "data_root": str(config.data_root),
        },
        "affinity_cores": affinity_cores,
        "versions": collect_versions(),
        "load_start": load_start,
        "load_end": load_end,
        "load_during": load_during,
        "sections": {
            name: {"correctness": s.correctness, "details": s.details}
            for name, s in sections.items()
        },
        "rows": [asdict(row) for row in rows],
    }
    write_results_csv(config.out_dir / "results.csv", rows)
    (config.out_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_markdown(config.out_dir / "summary.md", report, rows)
    print(f"done in {report['elapsed_s']} s -> {config.out_dir}", flush=True)


if __name__ == "__main__":
    main()
