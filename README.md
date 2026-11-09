# LLMViz Studio — Explore LLM architectures visually.

Turn Hugging Face `config.json` into publication-ready 2D architecture diagrams. No weights, GPU, API keys, or backend servers required. 100% client-side in the browser, deployable anywhere as a static site (GitHub Pages, Cloudflare Pages, Vercel).

## Quick start (Frontend static dev)
```bash
npm --prefix frontend install
npm --prefix frontend run dev
```
Open http://localhost:5173

## Build static production site
```bash
npm --prefix frontend run build
```
The static site is generated into `frontend/dist/`. You can deploy this folder directly to GitHub Pages, Cloudflare Pages, Netlify, or any static hosting service.

## Full-stack / Backend dev (Optional)
The Python FastAPI backend is also included for server-side environments:
```bash
./start.sh
# or with docker:
docker compose up --build
```

## Features
- **Client-Side Architecture Engine**: Directly parses Hugging Face `config.json` and safetensors metadata in the browser using open CORS.
- **2D Publication-Ready Diagrams**: Raschka-style colorful architecture diagrams with family accents, attention layers, MoE routing, and parameter breakdowns.
- **Interactive Component Graph**: Explore model layers, subgraphs, and weights with `@xyflow/react`.
- **Client-Side SVG & PNG Export**: Instant high-resolution vector and raster image downloads without backend dependencies.

## Tests
```bash
npm --prefix frontend test
npm --prefix frontend run build
python -m pytest backend/tests -q
```

## License / attribution
Apache-2.0 (see `LICENSE`). The diagram engine is based on the Apache-2.0 `llmviz` project by Hesham Haroon (https://github.com/h9-tec/llmviz), modified and renamed for LLMViz Studio; see `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `THIRD_PARTY_LICENSE.llmviz.txt`. Not affiliated with or endorsed by upstream.
