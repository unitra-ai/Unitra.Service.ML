#!/usr/bin/env python3
"""
Cloud MT Performance Benchmark Tool

Tests Modal-hosted MADLAD-400 translation performance with various inputs.
Designed to match the local mt_benchmark.rs for direct comparison.

Usage:
    python scripts/cloud_mt_benchmark.py
    python scripts/cloud_mt_benchmark.py --stress --concurrency 10
"""

import argparse
import asyncio
import io
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# Configuration
# =============================================================================

MODAL_TRANSLATE_URL = "https://nikmomo--unitra-mt-translate.modal.run"
MODAL_HEALTH_URL = "https://nikmomo--unitra-mt-health.modal.run"
API_KEY = os.environ.get(
    "MODAL_API_KEY",
    "unitra_ml_7KxN9mPqR4vZ8wL2YdF6jHsT3bGcU5aQnW1eXoIpM0uVrJtBhCkDfSgAyElN"
)

# Timeout settings
REQUEST_TIMEOUT = 120.0  # seconds
WARMUP_TIMEOUT = 180.0   # longer timeout for cold start


# =============================================================================
# Test Cases (matching mt_benchmark.rs exactly)
# =============================================================================

def get_test_cases() -> list[tuple[str, str, str]]:
    """Get test cases matching the local benchmark."""
    return [
        # Short sentences (< 10 tokens)
        ("Hello", "en", "zh"),
        ("Hello world", "en", "zh"),
        ("How are you?", "en", "zh"),
        ("Good morning!", "en", "ja"),
        ("Thank you very much", "en", "ko"),

        # Medium sentences (10-30 tokens)
        ("The quick brown fox jumps over the lazy dog.", "en", "zh"),
        ("Machine translation has improved significantly in recent years.", "en", "zh"),
        ("I would like to book a table for two people tonight.", "en", "ja"),

        # Long sentences (30-50 tokens)
        ("Artificial intelligence is transforming the way we live, work, and communicate with each other in unprecedented ways.", "en", "zh"),
        ("The development of large language models has opened new possibilities for natural language processing applications.", "en", "zh"),

        # Very long sentences (50+ tokens)
        ("In the field of machine learning, transformer models have revolutionized the way we approach natural language understanding and generation tasks, enabling applications that were previously considered impossible.", "en", "zh"),

        # Chinese to English
        ("你好", "zh", "en"),
        ("你好世界", "zh", "en"),
        ("今天天气真好，我想去公园散步。", "zh", "en"),
        ("人工智能正在改变我们的生活方式。", "zh", "en"),

        # Japanese to English
        ("こんにちは", "ja", "en"),
        ("今日はいい天気ですね。", "ja", "en"),

        # Korean to English
        ("안녕하세요", "ko", "en"),
        ("오늘 날씨가 좋습니다.", "ko", "en"),

        # Multi-language test
        ("Bonjour le monde", "fr", "en"),
        ("Hola mundo", "es", "en"),
        ("Guten Tag", "de", "en"),
    ]


def get_stress_test_cases() -> list[tuple[str, str, str]]:
    """Get additional test cases for stress testing."""
    return [
        # Technical content
        ("The CUDA execution provider supports all transformer operations natively, providing 3-5x faster inference compared to DirectML.", "en", "zh"),
        ("Neural machine translation uses encoder-decoder architecture with attention mechanisms.", "en", "ja"),

        # Mixed content
        ("Please send the report to john@example.com before 5:00 PM EST.", "en", "zh"),
        ("The meeting is scheduled for 2025-01-15 at 14:30.", "en", "ja"),

        # Edge cases
        ("A", "en", "zh"),  # Single character
        ("123", "en", "zh"),  # Numbers only
        ("Hello! How are you? I'm fine, thank you.", "en", "zh"),  # Multiple sentences

        # Long technical paragraph
        ("In the context of deep learning, the transformer architecture has become the de facto standard for sequence-to-sequence tasks. The self-attention mechanism allows the model to weigh the importance of different parts of the input sequence when generating each output token, leading to better performance on long-range dependencies.", "en", "zh"),
    ]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single translation benchmark."""
    input: str
    input_lang: str
    output_lang: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens_per_sec: float
    success: bool = True
    error: str | None = None


@dataclass
class BatchBenchmarkResult:
    """Result of a batch translation benchmark."""
    inputs: list[str]
    input_lang: str
    output_lang: str
    outputs: list[str]
    total_tokens: int
    latency_ms: float
    tokens_per_sec: float
    success: bool = True
    error: str | None = None


@dataclass
class StressTestResult:
    """Result of stress testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    errors: list[str] = field(default_factory=list)


@dataclass
class HealthStatus:
    """Health status of the MT service."""
    status: str
    model_id: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    warm: bool


# =============================================================================
# API Client
# =============================================================================

class CloudMTClient:
    """HTTP client for Modal MT service."""

    def __init__(
        self,
        translate_url: str = MODAL_TRANSLATE_URL,
        health_url: str = MODAL_HEALTH_URL,
        api_key: str = API_KEY,
        timeout: float = REQUEST_TIMEOUT,
    ):
        self.translate_url = translate_url
        self.health_url = health_url
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CloudMTClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"X-API-Key": self.api_key},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    async def health_check(self) -> HealthStatus:
        """Check service health."""
        response = await self.client.get(self.health_url)
        response.raise_for_status()
        data = response.json()
        return HealthStatus(
            status=data["status"],
            model_id=data["model_id"],
            model_loaded=data["model_loaded"],
            gpu_available=data["gpu_available"],
            gpu_memory_used_gb=data.get("gpu_memory_used_gb", 0),
            gpu_memory_total_gb=data.get("gpu_memory_total_gb", 0),
            warm=data.get("warm", False),
        )

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        """Translate a single text."""
        response = await self.client.post(
            self.translate_url,
            json={
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        response.raise_for_status()
        return response.json()

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        """Translate a batch of texts."""
        response = await self.client.post(
            self.translate_url,
            json={
                "texts": texts,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Benchmark Functions
# =============================================================================

async def run_warmup(client: CloudMTClient, iterations: int = 3) -> list[float]:
    """Run warmup iterations."""
    print(f"\n{'='*60}")
    print(f"=== Warmup ({iterations} iterations) ===")
    print(f"{'='*60}")

    latencies = []
    for i in range(1, iterations + 1):
        start = time.perf_counter()
        try:
            await client.translate("Hello", "en", "zh")
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            print(f"Warmup {i}: {latency_ms:.1f}ms")
        except Exception as e:
            print(f"Warmup {i}: FAILED - {e}")

    return latencies


async def run_single_benchmark(
    client: CloudMTClient,
    test_cases: list[tuple[str, str, str]],
) -> list[BenchmarkResult]:
    """Run single translation benchmarks."""
    results = []
    total = len(test_cases)

    print(f"\n{'='*60}")
    print(f"=== Running {total} Single Translation Tests ===")
    print(f"{'='*60}\n")

    for i, (text, src_lang, tgt_lang) in enumerate(test_cases, 1):
        display_text = text[:50] + "..." if len(text) > 50 else text
        print(f"[{i}/{total}] '{display_text}' ({src_lang} -> {tgt_lang})... ", end="", flush=True)

        start = time.perf_counter()
        try:
            response = await client.translate(text, src_lang, tgt_lang)
            latency_ms = (time.perf_counter() - start) * 1000

            tokens_used = response.get("tokens_used", len(text.split()))
            output = response["translation"]
            output_tokens = len(output.split())

            tokens_per_sec = (tokens_used / latency_ms) * 1000 if latency_ms > 0 else 0

            print("OK")
            print(f"   Output: '{output[:60]}{'...' if len(output) > 60 else ''}'")
            print(f"   Tokens: ~{tokens_used}, Latency: {latency_ms:.1f}ms, Speed: {tokens_per_sec:.1f} tok/s")

            results.append(BenchmarkResult(
                input=text,
                input_lang=src_lang,
                output_lang=tgt_lang,
                output=output,
                input_tokens=tokens_used,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                tokens_per_sec=tokens_per_sec,
            ))
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            print(f"FAILED: {e}")
            results.append(BenchmarkResult(
                input=text,
                input_lang=src_lang,
                output_lang=tgt_lang,
                output="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                tokens_per_sec=0,
                success=False,
                error=str(e),
            ))

        print()

    return results


async def run_batch_benchmark(
    client: CloudMTClient,
    batch_sizes: list[int] = [2, 4, 8, 16],
) -> list[BatchBenchmarkResult]:
    """Run batch translation benchmarks."""
    results = []
    base_texts = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I love machine learning.",
        "This is a test sentence.",
        "Artificial intelligence is amazing.",
        "Neural networks are powerful.",
        "Deep learning transforms industries.",
        "Natural language processing is fascinating.",
        "Computer vision enables new applications.",
        "Reinforcement learning solves complex problems.",
        "Generative models create new content.",
        "Transfer learning reduces training time.",
        "Attention mechanisms improve accuracy.",
        "Transformer models revolutionized NLP.",
        "Large language models are versatile.",
        "Machine translation bridges languages.",
    ]

    print(f"\n{'='*60}")
    print(f"=== Running Batch Translation Tests ===")
    print(f"{'='*60}\n")

    for batch_size in batch_sizes:
        texts = base_texts[:batch_size]
        print(f"Batch size {batch_size}... ", end="", flush=True)

        start = time.perf_counter()
        try:
            response = await client.translate_batch(texts, "en", "zh")
            latency_ms = (time.perf_counter() - start) * 1000

            total_tokens = response.get("total_tokens", sum(len(t.split()) for t in texts))
            translations = response["translations"]

            tokens_per_sec = (total_tokens / latency_ms) * 1000 if latency_ms > 0 else 0

            print(f"OK - {latency_ms:.1f}ms, {total_tokens} tokens, {tokens_per_sec:.1f} tok/s")
            print(f"   Avg per item: {latency_ms/batch_size:.1f}ms")

            results.append(BatchBenchmarkResult(
                inputs=texts,
                input_lang="en",
                output_lang="zh",
                outputs=translations,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                tokens_per_sec=tokens_per_sec,
            ))
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            print(f"FAILED: {e}")
            results.append(BatchBenchmarkResult(
                inputs=texts,
                input_lang="en",
                output_lang="zh",
                outputs=[],
                total_tokens=0,
                latency_ms=latency_ms,
                tokens_per_sec=0,
                success=False,
                error=str(e),
            ))

    return results


async def run_stress_test(
    client: CloudMTClient,
    num_requests: int = 50,
    concurrency: int = 5,
) -> StressTestResult:
    """Run stress test with concurrent requests."""
    print(f"\n{'='*60}")
    print(f"=== Stress Test ({num_requests} requests, {concurrency} concurrent) ===")
    print(f"{'='*60}\n")

    test_cases = get_test_cases() + get_stress_test_cases()
    latencies: list[float] = []
    errors: list[str] = []
    successful = 0
    failed = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def make_request(idx: int) -> float | None:
        nonlocal successful, failed
        text, src, tgt = test_cases[idx % len(test_cases)]

        async with semaphore:
            start = time.perf_counter()
            try:
                await client.translate(text, src, tgt)
                latency = (time.perf_counter() - start) * 1000
                successful += 1
                return latency
            except Exception as e:
                failed += 1
                errors.append(str(e))
                return None

    # Progress indicator
    completed = 0

    async def make_request_with_progress(idx: int) -> float | None:
        nonlocal completed
        result = await make_request(idx)
        completed += 1
        if completed % 10 == 0 or completed == num_requests:
            print(f"Progress: {completed}/{num_requests} ({successful} ok, {failed} failed)")
        return result

    start_time = time.perf_counter()

    tasks = [make_request_with_progress(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Collect successful latencies
    latencies = [r for r in results if r is not None]

    if latencies:
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return StressTestResult(
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time_ms=total_time_ms,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=sorted_latencies[p50_idx],
            p95_latency_ms=sorted_latencies[min(p95_idx, len(sorted_latencies)-1)],
            p99_latency_ms=sorted_latencies[min(p99_idx, len(sorted_latencies)-1)],
            requests_per_second=(successful / total_time_ms) * 1000,
            errors=errors[:10],  # Keep first 10 errors
        )
    else:
        return StressTestResult(
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=failed,
            total_time_ms=total_time_ms,
            avg_latency_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            requests_per_second=0,
            errors=errors[:10],
        )


# =============================================================================
# Summary and Reporting
# =============================================================================

def print_summary(
    results: list[BenchmarkResult],
    batch_results: list[BatchBenchmarkResult] | None = None,
    stress_result: StressTestResult | None = None,
    health: HealthStatus | None = None,
) -> None:
    """Print comprehensive benchmark summary."""
    print(f"\n{'='*60}")
    print("=== Cloud MT Performance Summary ===")
    print(f"{'='*60}\n")

    # Service info
    if health:
        print("Service Status:")
        print(f"  Model: {health.model_id}")
        print(f"  GPU: {'Available' if health.gpu_available else 'Not Available'}")
        if health.gpu_memory_total_gb > 0:
            print(f"  GPU Memory: {health.gpu_memory_used_gb:.2f} / {health.gpu_memory_total_gb:.2f} GB")
        print(f"  Warm: {health.warm}")
        print()

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        print("No successful translations.")
        return

    # Calculate statistics
    total_latency = sum(r.latency_ms for r in successful)
    total_input_tokens = sum(r.input_tokens for r in successful)
    total_output_tokens = sum(r.output_tokens for r in successful)

    avg_latency = total_latency / len(successful)
    min_latency = min(r.latency_ms for r in successful)
    max_latency = max(r.latency_ms for r in successful)
    avg_tokens_per_sec = sum(r.tokens_per_sec for r in successful) / len(successful)

    print("Single Translation Results:")
    print(f"  Total Translations: {len(successful)}/{len(results)}")
    print(f"  Total Input Tokens: {total_input_tokens}")
    print(f"  Total Output Tokens: {total_output_tokens}")
    print()
    print("  Latency Statistics:")
    print(f"    Average: {avg_latency:.1f}ms")
    print(f"    Min: {min_latency:.1f}ms")
    print(f"    Max: {max_latency:.1f}ms")
    print()
    print("  Throughput:")
    print(f"    Average: {avg_tokens_per_sec:.1f} tokens/sec")
    print(f"    Total: {(total_input_tokens / total_latency) * 1000:.1f} tokens/sec")
    print()

    # By input length (estimate tokens as words * 1.3)
    def estimate_tokens(text: str) -> int:
        return max(int(len(text.split()) * 1.3), len(text) // 4)

    short = [r for r in successful if estimate_tokens(r.input) < 10]
    medium = [r for r in successful if 10 <= estimate_tokens(r.input) < 30]
    long = [r for r in successful if estimate_tokens(r.input) >= 30]

    print("  By Input Length:")
    if short:
        avg = sum(r.latency_ms for r in short) / len(short)
        print(f"    Short (<10 tokens): {avg:.1f}ms avg ({len(short)} samples)")
    if medium:
        avg = sum(r.latency_ms for r in medium) / len(medium)
        print(f"    Medium (10-30 tokens): {avg:.1f}ms avg ({len(medium)} samples)")
    if long:
        avg = sum(r.latency_ms for r in long) / len(long)
        print(f"    Long (30+ tokens): {avg:.1f}ms avg ({len(long)} samples)")
    print()

    # By language pair
    print("  By Language Pair:")
    lang_pairs: dict[str, list[BenchmarkResult]] = {}
    for r in successful:
        key = f"{r.input_lang} -> {r.output_lang}"
        lang_pairs.setdefault(key, []).append(r)

    for pair, items in sorted(lang_pairs.items()):
        avg = sum(r.latency_ms for r in items) / len(items)
        print(f"    {pair}: {avg:.1f}ms avg ({len(items)} samples)")

    # Batch results
    if batch_results:
        print(f"\n{'='*60}")
        print("Batch Translation Results:")
        print(f"{'='*60}")
        successful_batch = [r for r in batch_results if r.success]
        for r in successful_batch:
            batch_size = len(r.inputs)
            print(f"  Batch {batch_size}: {r.latency_ms:.1f}ms total, {r.latency_ms/batch_size:.1f}ms/item, {r.tokens_per_sec:.1f} tok/s")

    # Stress test results
    if stress_result:
        print(f"\n{'='*60}")
        print("Stress Test Results:")
        print(f"{'='*60}")
        print(f"  Total Requests: {stress_result.total_requests}")
        print(f"  Successful: {stress_result.successful_requests}")
        print(f"  Failed: {stress_result.failed_requests}")
        print(f"  Total Time: {stress_result.total_time_ms:.1f}ms")
        print()
        print("  Latency Percentiles:")
        print(f"    P50: {stress_result.p50_latency_ms:.1f}ms")
        print(f"    P95: {stress_result.p95_latency_ms:.1f}ms")
        print(f"    P99: {stress_result.p99_latency_ms:.1f}ms")
        print(f"    Avg: {stress_result.avg_latency_ms:.1f}ms")
        print(f"    Min: {stress_result.min_latency_ms:.1f}ms")
        print(f"    Max: {stress_result.max_latency_ms:.1f}ms")
        print()
        print(f"  Throughput: {stress_result.requests_per_second:.2f} requests/sec")

        if stress_result.errors:
            print()
            print("  Sample Errors:")
            for err in stress_result.errors[:5]:
                print(f"    - {err[:80]}")


def generate_comparison_table(
    results: list[BenchmarkResult],
    output_file: str | None = None,
) -> str:
    """Generate a comparison table for local vs cloud benchmarks."""
    successful = [r for r in results if r.success]

    lines = []
    lines.append("# Cloud MT Benchmark Results")
    lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**Model**: MADLAD-400 3B (Modal Cloud)")
    lines.append("**GPU**: NVIDIA A10G (Modal)")
    lines.append("")
    lines.append("## Comparison Table (for Local vs Cloud)")
    lines.append("")
    lines.append("| Test Case | Cloud Latency | Cloud Tokens/s | Local Latency | Local Tokens/s |")
    lines.append("|-----------|---------------|----------------|---------------|----------------|")

    for r in successful:
        display = r.input[:40] + "..." if len(r.input) > 40 else r.input
        lines.append(f"| `{display}` ({r.input_lang}→{r.output_lang}) | {r.latency_ms:.1f}ms | {r.tokens_per_sec:.1f} | - | - |")

    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    if successful:
        avg_latency = sum(r.latency_ms for r in successful) / len(successful)
        avg_tokens = sum(r.tokens_per_sec for r in successful) / len(successful)

        lines.append("| Metric | Cloud | Local |")
        lines.append("|--------|-------|-------|")
        lines.append(f"| Avg Latency | {avg_latency:.1f}ms | - |")
        lines.append(f"| Avg Tokens/sec | {avg_tokens:.1f} | - |")
        lines.append(f"| Total Tests | {len(successful)} | - |")

    content = "\n".join(lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nReport saved to: {output_file}")

    return content


# =============================================================================
# Main Entry Point
# =============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud MT Performance Benchmark")
    parser.add_argument("--stress", action="store_true", help="Run stress tests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests for stress test")
    parser.add_argument("--requests", type=int, default=50, help="Total requests for stress test")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch tests")
    parser.add_argument("--output", type=str, help="Output file for comparison report")
    args = parser.parse_args()

    print("=" * 60)
    print("=== Cloud MT Performance Benchmark ===")
    print("=" * 60)
    print()
    print(f"Translate URL: {MODAL_TRANSLATE_URL}")
    print(f"Health URL: {MODAL_HEALTH_URL}")
    print()

    async with CloudMTClient(timeout=WARMUP_TIMEOUT) as client:
        # Check health
        print("Checking service health...")
        try:
            health = await client.health_check()
            print(f"  Status: {health.status}")
            print(f"  Model: {health.model_id}")
            print(f"  Model Loaded: {health.model_loaded}")
            print(f"  GPU Available: {health.gpu_available}")
            if health.gpu_memory_total_gb > 0:
                print(f"  GPU Memory: {health.gpu_memory_used_gb:.2f}/{health.gpu_memory_total_gb:.2f} GB")
            print(f"  Warm: {health.warm}")
        except Exception as e:
            print(f"  Health check failed: {e}")
            health = None

        # Warmup
        if not args.skip_warmup:
            await run_warmup(client)

    # Use shorter timeout for benchmarks
    async with CloudMTClient(timeout=REQUEST_TIMEOUT) as client:
        # Run single translation benchmarks
        test_cases = get_test_cases()
        results = await run_single_benchmark(client, test_cases)

        # Run batch benchmarks
        batch_results = None
        if not args.skip_batch:
            batch_results = await run_batch_benchmark(client)

        # Run stress test
        stress_result = None
        if args.stress:
            stress_result = await run_stress_test(
                client,
                num_requests=args.requests,
                concurrency=args.concurrency,
            )

        # Print summary
        print_summary(results, batch_results, stress_result, health)

        # Generate comparison report
        if args.output:
            generate_comparison_table(results, args.output)

    print(f"\n{'='*60}")
    print("=== Benchmark Complete ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
