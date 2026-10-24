# qwen2_5

Benchmark de latência do `qwen3.8:27b` via Ollama local, com
amostragem de métricas de uma GPU NVIDIA.

## Requisitos

- Python 3.14 ou superior e [uv](https://docs.astral.sh/uv/).
- Para o benchmark: Ollama em `http://127.0.0.1:11434`, modelo
  `qwen3.8:27b` baixado e `nvidia-smi` no PATH.
- A consulta atual espera uma única GPU NVIDIA.
- Os testes não precisam de GPU, servidor Ollama ou acesso à rede.

## Instalação

```bash
git clone https://github.com/aquental/qwen2_5.git
cd qwen2_5
uv sync --locked
```

O `uv` usa `.python-version` e instala as dependências de execução e
desenvolvimento registradas em `uv.lock`. Para instalar apenas as de execução,
use `uv sync --locked --no-dev`.

## Executar

Inicie o Ollama (`ollama serve`, se não estiver rodando como serviço), depois:

```bash
ollama pull qwen3.8:27b
nvidia-smi
uv run qwen-bench
```

Alternativa: `uv run python -m qwen.bench`. O comando `uv run qwen` é apenas o
exemplo inicial “Hello from qwen!”, não o benchmark.

São executados um warmup, uma medição com amostragem de GPU a 5 Hz e três
repetições. Modelo, prompt, contexto (`4096`), timeout (`180 s`) e intervalo de
amostragem são constantes em `src/qwen/bench.py`.

### Métricas e limitações
O modelo padrão tem aproximadamente 18 GB de download e exige memória adicional
para inferência. Em uma GPU de 11 GB, a execução exige offload para CPU/RAM e
pode ser significativamente mais lenta. O contexto permanece em 4096 tokens.

O Qwen3.8 pode emitir chunks de raciocínio (`thinking`). Este benchmark mantém
o comportamento padrão do modelo: o TTFT mede apenas o primeiro chunk de
resposta textual, ignorando os chunks de raciocínio. As métricas de geração
são as fornecidas pelo servidor e não se limitam ao texto contado pelo cliente.
Não compare diretamente esse TTFT com modelos sem raciocínio.

- **TTFT**: tempo do cliente até o primeiro chunk com texto, incluindo preparação
  da requisição e espera pela resposta.
- **prompt_eval_s / prompt_tok_s**: duração e velocidade de avaliação do prompt
  reportadas pelo servidor.
- **gen_tok_s / per_token_ms**: derivados dos tokens e duração do servidor,
  não da quantidade de chunks.
- **GPU**: picos de VRAM, utilização, potência e temperatura. A janela quente
  vai da primeira à última amostra com utilização >= 50% ou VRAM no pico.
- **Repetições**: mediana e desvio padrão amostral da velocidade de geração.

O warmup tenta reduzir o impacto do carregamento e aquecer o cache de prompt;
não garante cache hit em todas as configurações. Tempos são expressos em
segundos, memória em MiB, potência em W e temperatura em °C.
Métricas temporais ausentes aparecem como `N/A`.

O timeout do SDK limita operações de rede, incluindo leituras; não é um limite
para a duração total da geração. Timeout, falhas e ausência do chunk final são
reportados como estol/erro. Falhas no warmup ou na medição principal retornam
código 1. Repetições com falha são excluídas das estatísticas. Falhas em amostras
individuais de GPU são ignoradas; o baseline exige `nvidia-smi` funcionando.
O benchmark não valida a qualidade do texto gerado.

## Desenvolvimento

```bash
uv run pytest --cov=qwen.bench --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv build
```

Os testes simulam o SDK Ollama, relógio, falhas de leitura e `nvidia-smi`,
cobrindo métricas, streams incompletos, encerramento da thread e fluxo principal.