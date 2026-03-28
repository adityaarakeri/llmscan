# LLM Scan

> A fast CLI that scans your hardware and tells you which local LLMs will actually run on your machine.

<!-- badges -->
[![CI](https://github.com/adityaarakeri/llmscan/actions/workflows/ci.yml/badge.svg)](https://github.com/adityaarakeri/llmscan/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmscan)](https://pypi.org/project/llmscan/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/llmscan)](https://pypi.org/project/llmscan/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Auto-detects your hardware** -OS, CPU, RAM, and GPU memory in seconds
- **Multi-vendor GPU support** -NVIDIA (`nvidia-smi`), AMD ROCm (`rocm-smi`), Intel Arc (`xpu-smi`/`clinfo`), Apple Silicon (unified memory), Windows (`Get-CimInstance`/`wmic`)
- **Smart fitness scoring** -Rates every model as `great`, `ok`, `tight`, or `no` based on your VRAM, RAM, and multi-GPU setup
- **45+ bundled models** -Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, CodeLlama, StarCoder, and more
- **Hugging Face search** -Find GGUF models on Hugging Face and add them to your local catalog
- **Auto-computed VRAM** -Adds models with formula-derived memory requirements from parameter count and quantization
- **Multi-GPU aware** -Accounts for tensor parallelism across multiple GPUs
- **CPU-only inference** -Recognizes when you have enough RAM to run models without a GPU
- **Beautiful terminal UI** -Rich tables, color-coded ratings, ASCII banner
- **Custom catalogs** -Bring your own model list as a JSON file
- **JSON output** -Pipe results into scripts or dashboards

## Install

```bash
# With pip
pip install llmscan

# With pipx (isolated install)
pipx install llmscan

# With uv
uv pip install llmscan

# From source
git clone https://github.com/adityaarakeri/llmscan.git
cd llmscan
pip install -e .
```

The installed command remains `llmscan`.

## Quick Start

```bash
# Show banner + hardware summary + compatible models
llmscan

# Check version
llmscan --version
```

## Usage

### `scan` -Inspect your hardware

```bash
llmscan scan              # Rich formatted hardware details
llmscan scan --json       # Machine-readable JSON output
```

### `list` -List compatible models

```bash
llmscan list                          # Show models rated "tight" and above
llmscan list --min-rating great       # Only show "great" fits
llmscan list --min-rating no          # Show all models including non-fits
llmscan list --json                   # JSON output with machine profile + models
llmscan list --catalog my_models.json # Use a custom catalog file
```

### `explain` -Deep-dive on a specific model

```bash
llmscan explain llama-3.1-8b-instruct              # Why does this model fit?
llmscan explain qwen2.5-72b-instruct               # Why doesn't it?
llmscan explain my-model --catalog my_models.json   # Explain from a custom catalog
```

### `search` -Find GGUF models on Hugging Face

```bash
llmscan search llama              # Search for Llama GGUF models
llmscan search "codellama 13b"    # More specific search
llmscan search mistral --limit 5  # Limit results
llmscan search qwen --json        # JSON output
```

### `add` -Add a model to your local catalog

VRAM/RAM requirements are auto-computed from the parameter count and quantization type.

```bash
# Add by specifying params and quant manually
llmscan add my-model --params-b 7 --quant Q4_K_M --family Llama
llmscan add my-model --params-b 7 --quant Q4_K_M --family Llama --notes "My fine-tune"

# Add from a Hugging Face repo (auto-detects params and quant)
llmscan add TheBloke/Llama-2-7B-GGUF

# Overwrite an existing entry
llmscan add my-model --params-b 7 --quant Q8_0 --family Llama --force

# JSON output
llmscan add my-model --params-b 7 --quant Q4_K_M --json
```

Supported quantization types: `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `IQ2_XS`, `IQ3_XS`

Models are saved to `~/.llmscan/catalog.json` and automatically merged with the bundled catalog.

### `remove` -Remove a model from your local catalog

```bash
llmscan remove my-model   # Only removes user-added models, not bundled ones
```

### Example Output

```
llmscan list --min-rating ok
```

```
┃ Model                        ┃ Family  ┃ Params ┃ Fit   ┃ Min VRAM ┃ Rec VRAM ┃ Rec RAM ┃
┃ llama-3.1-8b-instruct        ┃ Llama   ┃ 8B     ┃ great ┃ 5.0 GB   ┃ 6.2 GB   ┃ 10 GB   ┃
┃ mistral-7b-instruct-v0.3     ┃ Mistral ┃ 7B     ┃ great ┃ 4.4 GB   ┃ 5.5 GB   ┃ 8 GB    ┃
┃ phi-4-mini                   ┃ Phi     ┃ 4B     ┃ great ┃ 2.5 GB   ┃ 3.1 GB   ┃ 6 GB    ┃
...
```

## Rating System

| Rating  | Meaning |
|---------|---------|
| `great` | GPU VRAM and RAM both meet recommended targets -best experience |
| `ok`    | Meets minimum requirements, may need moderate context limits or uses CPU-only inference |
| `tight` | Runs with CPU offload, reduced context, or multi-GPU splitting -expect slower performance |
| `no`    | Hardware is below practical minimums |

## Supported Hardware

| Vendor | Detection Method | Notes |
|--------|-----------------|-------|
| NVIDIA | `nvidia-smi` | Discrete GPUs with CUDA |
| AMD | `rocm-smi` | GPUs with ROCm drivers |
| Intel | `xpu-smi`, `clinfo` | Arc and Data Center GPUs |
| Apple | `sysctl` | Apple Silicon unified memory (65% GPU estimate) |
| Windows | `Get-CimInstance`, `wmic` | Fallback for any GPU on Windows |

## Custom Catalogs

Create a JSON file with your own model entries:

```json
[
  {
    "id": "my-custom-model-7b",
    "family": "Custom",
    "params_b": 7,
    "quant": "Q4_K_M",
    "min_vram_gb": 5,
    "recommended_vram_gb": 8,
    "recommended_ram_gb": 16,
    "notes": "My fine-tuned model"
  }
]
```

Then pass it with `--catalog`:

```bash
llmscan list --catalog my_models.json
llmscan explain my-custom-model-7b --catalog my_models.json
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique model identifier |
| `family` | string | Model family name (e.g., "LLaMA", "Mistral") |
| `params_b` | number | Parameter count in billions |
| `quant` | string | Quantization format (e.g., "Q4_K_M", "Q5_K_M") |
| `min_vram_gb` | number | Minimum GPU VRAM in GB |
| `recommended_vram_gb` | number | Recommended GPU VRAM in GB |
| `recommended_ram_gb` | number | Recommended system RAM in GB |
| `notes` | string | Additional notes about the model |

## Development

```bash
git clone https://github.com/adityaarakeri/llmscan.git
cd llmscan

# Install with dev dependencies
uv pip install -e ".[test,dev]"

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy llmscan/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Ensure all checks pass: `uv run pytest && uv run ruff check . && uv run mypy llmscan/`
5. Open a pull request

## License

MIT -see [LICENSE](LICENSE) for details.
