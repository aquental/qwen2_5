"""Benchmark de latência de geração para modelos servidos pelo Ollama local.

Usa o SDK oficial (ollama.Client) em vez de urlopen cru, o que resolve o
tratamento de Transfer-Encoding: chunked que quebrava o readline() manual.

Princípios de medição:
  * Fase de warmup: força o carregamento do modelo e o preenchimento do
    prompt cache ANTES de medir, para que o TTFT não capture I/O de disco
    nem a busca de prefixo do llama.cpp (n_slots=1 serializa tudo).
  * TTFT do cliente (primeiro chunk recebido) E prompt_eval do servidor são
    reportados separadamente - o segundo exclui o overhead de rede/busca.
  * tokens/s vem de eval_count / eval_duration do servidor, nunca da contagem
    de chunks (um chunk pode conter vários tokens).
  * Watchdog por leitura: se nenhum chunk chegar em STALL_TIMEOUT s, aborta
    com diagnóstico em vez de pendurar a conexão.
"""

import statistics
import subprocess
import threading
import time
from datetime import datetime

import ollama

MODEL = "qwen2.5-coder:7b"
NUM_CTX = 4096
STALL_TIMEOUT = 180.0  # s sem nenhum chunk => aborta
SAMPLE_INTERVAL = 0.2  # 5 Hz

PROMPT = (
    "Você é um engenheiro de sistemas especialista em concorrência. "
    "Explique em detalhes, com código Python executável, como implementar um "
    "thread pool com fila limitada, backpressure, cancelamento cooperativo de "
    "tarefas, timeout por tarefa e shutdown gracioso que drene a fila. "
    "Discuta também as condições de corrida que podem surgir entre o cancelamento "
    "e a execução de uma tarefa, e como evitá-las com locks e estados explícitos. "
    "Finalize com uma tabela comparando ThreadPoolExecutor, ProcessPoolExecutor e "
    "um pool baseado em asyncio. Seja exaustivo."
)

GPU_FIELDS = (
    "memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm"
)


def sample_gpu():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    return [x.strip() for x in out.split(",")]


class GpuSampler(threading.Thread):
    """Amostra a GPU em thread separada; nunca bloqueia o caminho de inferência."""

    def __init__(self, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.samples = []  # (t_rel, mem_used, mem_free, util, watts, temp, sm_mhz)
        self.peak_mem = 0
        self.peak_util = 0
        self.peak_watts = 0.0
        self.peak_temp = 0
        self._t0 = None

    def run(self):
        self._t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                mem_used, mem_free, util, watts, temp, sm = sample_gpu()
                t = time.perf_counter() - self._t0
                rec = (t, int(mem_used), int(mem_free), int(util), float(watts), int(temp), int(sm))
                self.samples.append(rec)
                self.peak_mem = max(self.peak_mem, rec[1])
                self.peak_util = max(self.peak_util, rec[3])
                self.peak_watts = max(self.peak_watts, rec[4])
                self.peak_temp = max(self.peak_temp, rec[5])
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

    def peak_window(self):
        """Janela de tempo em que a VRAM esteve no pico (util >= 50% ou mem no pico)."""
        hot = [s for s in self.samples if s[3] >= 50 or s[1] >= self.peak_mem]
        if not hot:
            return None
        return hot[0][0], hot[-1][0]


def baseline():
    mem_used, mem_free, util, watts, temp, sm = sample_gpu()
    return {
        "mem_used": int(mem_used),
        "mem_free": int(mem_free),
        "util": int(util),
        "watts": float(watts),
        "temp": int(temp),
        "sm_mhz": int(sm),
    }


def run_stream(prompt, label):
    """Uma passada de streaming. Retorna dict de métricas ou marcador de estol."""
    client = ollama.Client(host="http://127.0.0.1:11434")
    stream = client.generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        options={"num_ctx": NUM_CTX, "temperature": 0.2, "seed": 42},
    )

    t0 = time.perf_counter()
    ttft = None
    chunk_count = 0
    chars = 0
    last_activity = t0
    final = None
    stalled = False

    try:
        for chunk in stream:
            now = time.perf_counter()
            if chunk.response:
                if ttft is None:
                    ttft = now - t0
                chunk_count += 1
                chars += len(chunk.response)
            last_activity = now
            if now - last_activity > STALL_TIMEOUT:
                stalled = True
                break
            if chunk.done:
                final = chunk
                break
    except Exception as exc:  # noqa: BLE001 - diagnóstico do benchmark
        print(f"  [{label}] EXCEÇÃO durante o streaming: {type(exc).__name__}: {exc}")
        stalled = True

    if final is None and not stalled:
        stalled = True

    if stalled or final is None:
        return {"stalled": True, "ttft": ttft, "chunks": chunk_count, "chars": chars}

    def sec(v):
        return v / 1e9 if v is not None else None

    pe, pd = final.prompt_eval_count, sec(final.prompt_eval_duration)
    ec, ed = final.eval_count, sec(final.eval_duration)

    return {
        "stalled": False,
        "ttft": ttft,
        "prompt_eval_count": pe,
        "prompt_eval_s": pd,
        "eval_count": ec,
        "eval_s": ed,
        "total_s": sec(final.total_duration),
        "load_s": sec(final.load_duration),
        "prompt_tok_s": (pe / pd) if pe and pd else None,
        "gen_tok_s": (ec / ed) if ec and ed else None,
        "per_token_ms": (ed / ec * 1000) if ec and ed else None,
        "chunks": chunk_count,
        "chars": chars,
    }


def report(r, label, gpu=None):
    print(f"\n### {label}")
    if r["stalled"]:
        print("  RESULTADO: ESTOL - nenhum chunk novo dentro do watchdog")
        if r.get("ttft"):
            print(f"  ttft antes do estol   = {r['ttft']:.3f} s")
        print(f"  chunks recebidos      = {r['chunks']}")
        return
    print(f"  ttft_cliente_s        = {r['ttft']:.3f}   (primeiro chunk; inclui rede + prefix search)")
    print(f"  prompt_eval_count     = {r['prompt_eval_count']} tokens")
    print(f"  prompt_eval_s         = {r['prompt_eval_s']:.3f}")
    if r["prompt_tok_s"]:
        print(f"  prompt_tok_s          = {r['prompt_tok_s']:.1f}   (prefill puro, sem overhead de cliente)")
    print(f"  eval_count            = {r['eval_count']} tokens gerados")
    print(f"  eval_s                = {r['eval_s']:.3f}")
    if r["gen_tok_s"]:
        print(f"  gen_tok_s             = {r['gen_tok_s']:.1f}")
    if r["per_token_ms"]:
        print(f"  per_token_ms          = {r['per_token_ms']:.1f}")
    print(f"  total_s               = {r['total_s']:.3f}")
    if r["load_s"]:
        print(f"  load_s                = {r['load_s']:.3f}")
    print(f"  stream_chunks         = {r['chunks']}  (NAO e contagem de tokens)")
    if gpu is not None:
        print(f"  pico_vram_mib         = {gpu.peak_mem}")
        print(f"  pico_util_pct         = {gpu.peak_util}")
        print(f"  pico_watts            = {gpu.peak_watts:.1f}")
        print(f"  pico_temp_c           = {gpu.peak_temp}")
        w = gpu.peak_window()
        if w:
            print(f"  janela_quente_s       = {w[0]:.2f} -> {w[1]:.2f}  (dur {w[1] - w[0]:.2f}s)")


def main():
    print(f"timestamp             = {datetime.now().isoformat(timespec='seconds')}")
    print(f"modelo                = {MODEL}   (num_ctx={NUM_CTX})")
    print(f"prompt_chars          = {len(PROMPT)}")
    print(f"stall_timeout_s       = {STALL_TIMEOUT}")

    client = ollama.Client(host="http://127.0.0.1:11434")
    loaded = [m for m in client.ps().models if MODEL in m.model]
    print(f"modelo_ja_carregado   = {bool(loaded)}")
    if loaded:
        m = loaded[0]
        print(f"  processor           = {m.processor}  ctx={m.context_length}")

    b = baseline()
    print(f"baseline_pre_warmup   = vram {b['mem_used']} MiB usados / {b['mem_free']} livres | "
          f"gpu {b['util']}% | {b['watts']:.1f} W | {b['temp']} C | sm {b['sm_mhz']} MHz")

    # ---- FASE 1: WARMUP (nao medido) ------------------------------------
    # Garante modelo residente e prefix cache populado, para que a medicao
    # nao capture a busca de prefixo do llama.cpp nem I/O de disco.
    print("\n>>> warmup: residente + prompt cache (nao medido)")
    warm = run_stream(PROMPT, "warmup")
    if warm["stalled"]:
        print("  WARMUP ESTOLOU - o servidor nao respondeu. Diagnostico:")
        print("    journalctl -u ollama -n 50 --no-pager")
        print("    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
        return 1
    print(f"  warmup ok: ttft={warm['ttft']:.3f}s  gen_tok_s={warm['gen_tok_s']:.1f}  "
          f"prompt_tok_s={warm['prompt_tok_s']:.1f}")
    time.sleep(1.0)

    # ---- FASE 2: MEDICAO com amostragem ---------------------------------
    print("\n>>> medicao com amostragem de GPU a 5 Hz")
    gpu = GpuSampler()
    gpu.start()
    res = run_stream(PROMPT, "medicao")
    gpu.stop()
    report(res, "MEDICAO (prompt cache quente)", gpu)

    mem_used, mem_free, util, watts, temp, sm = sample_gpu()
    print(f"\npos_medicao           = vram {mem_used} MiB usados / {mem_free} livres | "
          f"gpu {util}% | {watts:.1f} W | {temp} C | sm {sm} MHz")

    # ---- FASE 3: REPETICOES ---------------------------------------------
    reps = []
    for i in range(2, 5):
        r = run_stream(PROMPT, f"rep{i}")
        if not r["stalled"] and r["gen_tok_s"]:
            reps.append(r["gen_tok_s"])
        time.sleep(0.5)
    if reps:
        print(f"\nreps gen_tok_s        = {[round(x, 1) for x in reps]}")
        print(f"  mediana             = {statistics.median(reps):.1f}")
        if len(reps) > 1:
            print(f"  desvio_padrao       = {statistics.stdev(reps):.1f}")

    print(f"\namostras_gpu          = {len(gpu.samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
