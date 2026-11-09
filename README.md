# LLMViz Studio — Interactive Neural Network Architecture Diagrams

Transform Hugging Face model configurations and SafeTensors checkpoints into unified, publication-grade interactive architecture diagrams with progressive disclosure. No weight downloads, GPUs, API keys, or backend servers required. 100% client-side in the browser, deployable anywhere as a static site.

- **Repository**: [https://github.com/kaushikharsh99/llm-viz-studio](https://github.com/kaushikharsh99/llm-viz-studio)
- **Live Deployment**: [https://llm-viz-studio.vercel.app/](https://llm-viz-studio.vercel.app/)

## Key Capabilities

- **Unified Interactive Diagram Engine**: Beautiful publication-quality architecture diagrams (Raschka family style) with progressive disclosure.
- **Detail Slider (1 — 2 — 3)**:
  - **Preset 1 (Overview)**: Faithful, publication-grade high-level diagram (Embeddings, Repeated blocks, Attention block, FFN / MoE block, Norms, Residuals, Output Head).
  - **Preset 2 (Module Detail)**: Expands inline primary internal structure (Q/K/V/O projections, RoPE, Gate/Up/Activation/Down paths, MoE router and experts).
  - **Preset 3 (Deep Detail)**: Detailed mathematical operations (MatMul, Causal Mask, Softmax, Hadamard product $\otimes$) with exact verified SafeTensors shapes and dtypes.
- **Manual Section Expansion**: Click any section directly inside the diagram to expand or collapse it on demand. Dynamic layout reflow ensures zero overlapping components and perfectly routed residual bypass rails.
- **Focus Mode**: Study individual mechanisms (such as Multi-Head Attention, DeepSeek MLA, or MoE Routers) in isolated, publication-quality illustrations with breadcrumb navigation back to the full model.
- **Advanced 4-Mode Export System**:
  - Export Preset 1 (Overview SVG / PNG)
  - Export Preset 2 (Module Detail SVG / PNG)
  - Export Preset 3 (Deep Detail SVG / PNG)
  - Export Current Workspace (custom layout with your exact expansion arrangement)
- **Model-Specific Architecture Adapters**: Verified support for Llama / SmolLM2, Qwen 2 / 2.5, Qwen 3 / 3.5, and DeepSeek MLA & MoE.
- **In-Diagram Component Search & Auto-Zoom**: Search any operator, tensor, or projection layer with instant animated zoom-to-node and contextual inspection.
- **Direct Model URLs & 1-Click Workspace Sharing**: Open any model directly via URL pathname (e.g. `https://llm-viz-studio.vercel.app/Qwen/Qwen2.5-1.5B-Instruct`) and share your exact expansion arrangement via URL.
- **Side-by-Side Architecture Comparison**: Compare any two models side-by-side with color-coded structural diffs across attention, parameters, MoE routing, and layer dimensions.
- **100% Client-Side & Serverless**: Safe bounded HTTP Range requests directly to Hugging Face, running entirely in your browser with zero backend latency.

## Quick Start (Development)

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Production Build

```bash
npm run build
```

Generates the static build into `frontend/dist/`.

## Running Tests

```bash
npm test
```

Runs comprehensive unit and validation suites (Vitest + Pytest) across all model families, detail presets, expansion states, and export modes.

## License

Open source under the [Apache-2.0 License](LICENSE).
