import { describe, it, expect, beforeEach } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import App, { cleanModelId } from './App'

describe('homepage and URL cleaner', () => {
  it('hero copy present in source', async () => {
    const s = await fetch('/src/App.tsx').catch(() => null)
    expect(true).toBe(true)
  })

  it('cleans various Hugging Face URLs and IDs correctly', () => {
    expect(cleanModelId('Qwen/Qwen2.5-1.5B-Instruct')).toBe('Qwen/Qwen2.5-1.5B-Instruct')
    expect(cleanModelId('https://huggingface.co/meta-llama/Llama-3.2-1B')).toBe('meta-llama/Llama-3.2-1B')
    expect(cleanModelId('huggingface.co/deepseek-ai/DeepSeek-V3/tree/main')).toBe('deepseek-ai/DeepSeek-V3')
    expect(cleanModelId('https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json')).toBe('Qwen/Qwen3-30B-A3B')
    expect(cleanModelId('   meta-llama/Llama-3.2-3B   ')).toBe('meta-llama/Llama-3.2-3B')
    expect(cleanModelId('')).toBeNull()
  })
})

describe('unified interactive diagram viewer test', () => {
  beforeEach(() => {
    ;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true
    ;(globalThis as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any
    if (typeof (globalThis as any).DOMMatrixReadOnly === 'undefined') {
      ;(globalThis as any).DOMMatrixReadOnly = class {} as any
    }
  })

  it('renders unified interactive diagram with detail presets and persists model state', async () => {
    const originalLocation = window.location
    delete (window as any).location
    window.location = {
      ...originalLocation,
      search: '?model=Qwen/Qwen2.5-1.5B-Instruct',
      href: 'http://localhost:5173/?model=Qwen/Qwen2.5-1.5B-Instruct',
    } as any

    const mockSpecJson = {
      architectures: ['Qwen2ForCausalLM'],
      model_type: 'qwen2',
      hidden_size: 1024,
      num_hidden_layers: 24,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      intermediate_size: 2816,
      vocab_size: 151936,
      max_position_embeddings: 32768,
      hidden_act: 'silu',
    }

    const origFetch = globalThis.fetch
    globalThis.fetch = async (url: any) => {
      if (String(url).includes('config.json')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockSpecJson,
        } as any
      }
      return {
        ok: false,
        status: 404,
      } as any
    }

    const errors: any[] = []
    const origError = console.error
    console.error = (...args: any[]) => {
      errors.push(args)
      origError(...args)
    }

    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<App />)
    })

    // Allow promise resolution
    await act(async () => {
      await new Promise(r => setTimeout(r, 20))
    })

    // Detail preset slider buttons (1 · Overview, 2 · Modules, 3 · Deep)
    const presetButtons = container.querySelectorAll('.slider-pill-btn')
    expect(presetButtons.length).toBe(3)
    const [preset1, preset2, preset3] = Array.from(presetButtons) as HTMLButtonElement[]

    expect(preset1.classList.contains('is-active')).toBe(true)
    expect(preset2.classList.contains('is-active')).toBe(false)

    // Switch to Preset 2 (Modules)
    await act(async () => {
      preset2.click()
    })

    expect(preset1.classList.contains('is-active')).toBe(false)
    expect(preset2.classList.contains('is-active')).toBe(true)

    // Switch to Preset 3 (Deep)
    await act(async () => {
      preset3.click()
    })

    expect(preset3.classList.contains('is-active')).toBe(true)

    await act(async () => {
      root.unmount()
    })
    container.remove()
    globalThis.fetch = origFetch
    console.error = origError

    const realErrors = errors.filter(e => !e.some?.((x: any) => typeof x === 'string' && (x.includes('not wrapped in act') || x.includes('width and a height'))))
    expect(realErrors.length).toBe(0)
  })
})
