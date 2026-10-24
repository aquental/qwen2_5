import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import ollama
import pytest

from qwen import bench


def test_default_model_configuration():
    assert bench.MODEL == "qwen3.8:27b"
    assert bench.NUM_CTX == 4096


def response(**overrides):
    return ollama.GenerateResponse(
        **(
            {
                "response": "",
                "done": True,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 500_000_000,
                "eval_count": 10,
                "eval_duration": 2_000_000_000,
                "total_duration": 3_000_000_000,
                "load_duration": 100_000_000,
            }
            | overrides
        )
    )


@pytest.fixture
def streaming(monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(bench.ollama, "Client", factory)
    ticks = iter([10.0, 10.1, 10.25, 10.5, 13.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))

    def setup(chunks):
        def generate():
            yield from chunks

        stream = generate()
        client.generate.return_value = stream
        return stream

    return setup, client, factory


def test_stream_metrics(streaming):
    setup, client, factory = streaming
    stream = setup(
        [
            response(done=False),
            response(response="Hello", done=False),
            response(response=" world", done=False),
            response(),
        ]
    )
    result = bench.run_stream("prompt", "test")
    assert result["stalled"] is False
    assert result["ttft"] == pytest.approx(0.25)
    assert result["chunks"] == 2
    assert result["chars"] == 11
    for key, expected in {
        "prompt_eval_s": 0.5,
        "eval_s": 2,
        "total_s": 3,
        "load_s": 0.1,
        "prompt_tok_s": 40,
        "gen_tok_s": 5,
        "per_token_ms": 200,
    }.items():
        assert result[key] == expected
    assert stream.gi_frame is None
    factory.assert_called_once_with(
        host="http://127.0.0.1:11434", timeout=bench.STALL_TIMEOUT
    )
    client.generate.assert_called_once_with(
        model=bench.MODEL,
        prompt="prompt",
        stream=True,
        options={"num_ctx": bench.NUM_CTX, "temperature": 0.2, "seed": 42},
    )


@pytest.mark.parametrize("value", [0, None])
def test_missing_metrics(streaming, capsys, value):
    setup, _, _ = streaming
    setup(
        [
            response(
                prompt_eval_count=value,
                prompt_eval_duration=value,
                eval_count=value,
                eval_duration=value,
                total_duration=value,
            )
        ]
    )
    result = bench.run_stream("prompt", "test")
    assert result["stalled"] is False
    assert result["ttft"] is None
    assert all(
        result[key] is None for key in ("prompt_tok_s", "gen_tok_s", "per_token_ms")
    )
    bench.report(result, "empty")
    assert "N/A" in capsys.readouterr().out


@pytest.mark.parametrize("chunks", [[], [response(response="partial", done=False)]])
def test_incomplete_stream(streaming, chunks):
    setup, _, _ = streaming
    stream = setup(chunks)
    result = bench.run_stream("prompt", "test")
    assert result["stalled"] is True
    assert result["chunks"] == len(chunks)
    assert stream.gi_frame is None


@pytest.mark.parametrize("during_iteration", [False, True])
def test_stream_errors(streaming, capsys, during_iteration):
    _, client, _ = streaming
    if during_iteration:

        def broken():
            yield response(response="partial", done=False)
            raise TimeoutError("read timed out")

        stream = broken()
        client.generate.return_value = stream
    else:
        client.generate.side_effect = ConnectionError("connection refused")
    result = bench.run_stream("prompt", "failure")
    assert result["stalled"] is True
    assert result["chunks"] == int(during_iteration)
    assert "[failure] EXCEÇÃO" in capsys.readouterr().out
    if during_iteration:
        assert stream.gi_frame is None


def test_gpu_baseline(monkeypatch):
    run = Mock(
        return_value=SimpleNamespace(stdout=" 1024, 7168, 75, 120.5, 62, 1500\n")
    )
    monkeypatch.setattr(bench.subprocess, "run", run)
    assert bench.baseline() == {
        "mem_used": 1024,
        "mem_free": 7168,
        "util": 75,
        "watts": 120.5,
        "temp": 62,
        "sm_mhz": 1500,
    }
    run.assert_called_once_with(
        [
            "nvidia-smi",
            f"--query-gpu={bench.GPU_FIELDS}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    run.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
    with pytest.raises(subprocess.CalledProcessError):
        bench.baseline()


def test_sampler_peaks_and_errors(monkeypatch):
    sampler = bench.GpuSampler(interval=0)
    samples = iter(
        [
            ["bad"],
            ["100", "900", "30", "40.5", "50", "1000"],
            ["200", "800", "70", "60.5", "60", "1200"],
        ]
    )

    def sample():
        values = next(samples)
        if values[0] == "200":
            sampler._stop_event.set()
        return values

    ticks = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr(bench, "sample_gpu", sample)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    assert sampler.peak_window() is None
    sampler.run()
    assert len(sampler.samples) == 2
    assert sampler.samples[0] == (0.5, 100, 900, 30, 40.5, 50, 1000)
    assert (
        sampler.peak_mem,
        sampler.peak_util,
        sampler.peak_watts,
        sampler.peak_temp,
    ) == (
        200,
        70,
        60.5,
        60,
    )
    assert sampler.peak_window() == (1.0, 1.0)


def test_sampler_thread_shutdown(monkeypatch):
    sampled = threading.Event()

    def sample():
        sampled.set()
        return ["100", "900", "50", "40", "50", "1000"]

    monkeypatch.setattr(bench, "sample_gpu", sample)
    sampler = bench.GpuSampler(interval=60)
    sampler.start()
    try:
        assert sampled.wait(timeout=2)
    finally:
        sampler.stop()
    assert not sampler.is_alive()
    assert len(sampler.samples) == 1
    sampler.stop()


def test_report(streaming, capsys):
    setup, _, _ = streaming
    setup([response(response="text")])
    gpu = bench.GpuSampler()
    gpu.peak_mem = 100
    gpu.samples = [(0.2, 100, 900, 60, 40, 50, 1000)]
    bench.report(bench.run_stream("prompt", "test"), "success", gpu)
    output = capsys.readouterr().out
    assert "gen_tok_s" in output
    assert "pico_vram_mib" in output
    assert "janela_quente_s" in output
    bench.report({"stalled": True, "ttft": 0.0, "chunks": 1}, "failure")
    output = capsys.readouterr().out
    assert "ESTOL/ERRO" in output
    assert "0.000 s" in output


@pytest.fixture
def main_setup(monkeypatch):
    model = ollama.ProcessResponse.Model(
        model=bench.MODEL, size_vram=123, context_length=bench.NUM_CTX
    )
    client = Mock()
    client.ps.return_value = SimpleNamespace(models=[model])
    monkeypatch.setattr(bench.ollama, "Client", Mock(return_value=client))
    monkeypatch.setattr(
        bench, "sample_gpu", lambda: ["100", "900", "50", "40", "50", "1000"]
    )
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    gpu = Mock(peak_mem=100, peak_util=50, peak_watts=40, peak_temp=50, samples=[])
    gpu.peak_window.return_value = None
    factory = Mock(return_value=gpu)
    monkeypatch.setattr(bench, "GpuSampler", factory)
    success = {
        "stalled": False,
        "ttft": 0.1,
        "chunks": 2,
        "chars": 10,
        "prompt_eval_count": 20,
        "prompt_eval_s": 0.5,
        "eval_count": 10,
        "eval_s": 2,
        "total_s": 3,
        "load_s": 0.1,
        "prompt_tok_s": 40,
        "gen_tok_s": 5,
        "per_token_ms": 200,
    }
    run = Mock(return_value=success)
    monkeypatch.setattr(bench, "run_stream", run)
    return run, gpu, factory, success


def test_main_success(main_setup, capsys):
    run, gpu, _, success = main_setup
    run.side_effect = [
        success,
        success,
        success | {"gen_tok_s": 4},
        success | {"gen_tok_s": 6},
        success | {"gen_tok_s": 8},
    ]
    assert bench.main() == 0
    assert [call.args[1] for call in run.call_args_list] == [
        "warmup",
        "medicao",
        "rep2",
        "rep3",
        "rep4",
    ]
    gpu.start.assert_called_once()
    gpu.stop.assert_called_once()
    output = capsys.readouterr().out
    assert "size_vram_bytes" in output
    assert "mediana             = 6.0" in output
    assert "desvio_padrao       = 2.0" in output


def test_warmup_failure(main_setup):
    run, _, factory, _ = main_setup
    run.return_value = {"stalled": True}
    assert bench.main() == 1
    assert run.call_count == 1
    factory.assert_not_called()


def test_measurement_failure(main_setup):
    run, gpu, _, success = main_setup
    run.side_effect = [success, {"stalled": True, "ttft": None, "chunks": 0}]
    assert bench.main() == 1
    assert run.call_count == 2
    gpu.stop.assert_called_once()


def test_sampler_cleanup_on_error(main_setup):
    run, gpu, _, success = main_setup
    run.side_effect = [success, RuntimeError("unexpected")]
    with pytest.raises(RuntimeError, match="unexpected"):
        bench.main()
    gpu.stop.assert_called_once()


def test_failed_repetitions_excluded(main_setup, capsys):
    run, _, _, success = main_setup
    run.side_effect = [
        success,
        success,
        {"stalled": True},
        success | {"gen_tok_s": None},
        success,
    ]
    assert bench.main() == 0
    output = capsys.readouterr().out
    assert "mediana             = 5.0" in output
    assert "desvio_padrao" not in output
