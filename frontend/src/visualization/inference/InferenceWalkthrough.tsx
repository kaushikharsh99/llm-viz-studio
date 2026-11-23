import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import {
  BLOCK_COUNT, evaluation, EXAMPLE_PROMPT, HEADS, MODEL_WIDTH, parameterCount, traceInference, VOCABULARY,
  type ChannelTensor,
} from '../../engine/inference/tinyRecall'
import InferenceScene, { type InferenceTransition } from './InferenceScene'
import './inference.css'

function TensorTable({ tensor, position }: { tensor: ChannelTensor; position: number }) {
  const { values, rowLabels, columnLabels, probability } = tensor
  const max = Math.max(1e-9, ...values.flat().filter(Number.isFinite).map(Math.abs))
  const columns = values[0].map((_, i) => columnLabels?.[i] ?? String(i))
  return (
    <div className="inference-matrix-scroll" tabIndex={0} aria-label="Scrollable tensor values">
      <table className="inference-matrix">
        <caption>{tensor.name} · {values.length} × {columns.length}</caption>
        <thead><tr><th scope="col">Position</th>{columns.map(column => <th key={column} scope="col">{column}</th>)}</tr></thead>
        <tbody>
          {values.map((row, i) => (
            <tr key={i} className={i === position ? 'inference-selected-row' : ''}>
              <th scope="row">{rowLabels?.[i] ?? i}</th>
              {row.map((n, j) => {
                const intensity = Math.abs(n) / (probability ? 1 : max)
                const finite = Number.isFinite(n)
                const color = n < 0 ? '171, 86, 43' : '25, 112, 135'
                return (
                  <td
                    key={j}
                    className={finite ? undefined : 'inference-masked-cell'}
                    title={finite ? String(n) : 'Masked: negative infinity'}
                    style={finite ? { backgroundColor: `rgba(${color}, ${intensity * 0.28})` } : undefined}
                  >{!finite ? '−∞' : tensor.name === 'IDs' ? n : n.toFixed(3)}</td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function InferenceWalkthrough({ onClose }: { onClose: () => void }) {
  const transitionId = useRef(0)
  const [transition, setTransition] = useState<InferenceTransition | null>(null)
  const [playing, setPlaying] = useState(false)
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPT)
  const [trace, setTrace] = useState(() => traceInference(EXAMPLE_PROMPT))
  const [error, setError] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [tensorIndex, setTensorIndex] = useState(0)
  const [position, setPosition] = useState(trace.tokens.length - 1)
  const step = trace.steps[stepIndex]
  const tensor = step.tensors[tensorIndex]
  const finalStep = stepIndex === trace.steps.length - 1

  const goTo = useCallback((index: number) => {
    if (index === stepIndex || index < 0 || index >= trace.steps.length) return
    setTransition({ id: ++transitionId.current, from: stepIndex, to: index, departure: tensor })
    setStepIndex(index)
    setTensorIndex(0)
    if (index === trace.steps.length - 1) setPlaying(false)
  }, [stepIndex, tensor, trace.steps.length])

  useEffect(() => {
    if (!playing) return
    const timer = window.setTimeout(() => goTo(stepIndex + 1), 1550)
    return () => window.clearTimeout(timer)
  }, [playing, stepIndex, goTo])

  const togglePlayback = () => {
    if (playing) {
      setPlaying(false)
    } else {
      if (finalStep) goTo(0)
      setPlaying(true)
    }
  }

  const run = (event: FormEvent) => {
    event.preventDefault()
    try {
      const next = traceInference(prompt)
      setTrace(next)
      setPosition(next.tokens.length - 1)
      setError('')
      setPlaying(false)
      setTransition(null)
      setStepIndex(0)
      setTensorIndex(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not run inference.')
    }
  }

  return (
    <section className="inference-workspace" aria-labelledby="inference-title">
      <header className="inference-header">
        <div>
          <h2 id="inference-title">Live inference</h2>
          <p>{parameterCount.toLocaleString()} learned parameters · {BLOCK_COUNT} decoder blocks · {HEADS} heads per block · {MODEL_WIDTH} channels</p>
        </div>
        <button type="button" className="toolbar-btn" onClick={onClose}>Back to architecture</button>
      </header>
      <div className="inference-workbench">
        <InferenceScene
          steps={trace.steps} tensor={tensor} tokens={trace.tokens} step={stepIndex}
          position={position} transition={transition} onStep={goTo}
          playback={(
            <div className="inference-step-buttons">
              <button type="button" className="toolbar-btn" disabled={stepIndex === 0} onClick={() => goTo(stepIndex - 1)}>Previous</button>
              <button type="button" className="inference-primary" onClick={togglePlayback}>
                {playing ? 'Pause' : finalStep ? 'Replay' : 'Play'}
              </button>
              <button type="button" className="toolbar-btn" disabled={finalStep} onClick={() => goTo(stepIndex + 1)}>Next</button>
            </div>
          )}
        />
        <aside className="inference-inspector" aria-label="Inference controls and tensor values">
          <form className="inference-prompt" onSubmit={run}>
            <label htmlFor="inference-prompt">Recall prompt</label>
            <div className="inference-input-row">
              <input id="inference-prompt" aria-describedby="inference-prompt-hint" aria-invalid={Boolean(error)}
                value={prompt} onChange={event => setPrompt(event.target.value)} maxLength={60} autoComplete="off" spellCheck={false} />
              <button type="submit" className="inference-primary">Run inference</button>
            </div>
            <p id="inference-prompt-hint">Assign A, B, and C values from 1–3, then ask for one key. Example · {EXAMPLE_PROMPT}</p>
            {error && <p className="inference-error" role="alert">{error}</p>}
          </form>
          <div className="inference-token-strip" role="group" aria-label="Token positions">
            {trace.tokens.map((token, i) => (
              <button type="button" key={i} aria-pressed={i === position} onClick={() => setPosition(i)}
                aria-label={`Position ${i}, token ${token}, ID ${trace.ids[i]}`}>
                <strong>{token}</strong><small>{trace.ids[i]}</small>
              </button>
            ))}
            <span className="inference-token-caption">Token IDs</span>
          </div>
          <nav className="inference-playback" aria-label="Inference steps">
            <div className="inference-step-picker">
              <label htmlFor="inference-step">Step {stepIndex + 1} of {trace.steps.length}</label>
              <select id="inference-step" value={stepIndex} onChange={event => goTo(Number(event.target.value))}>
                {trace.steps.map((item, i) => <option key={item.title} value={i}>{item.title}</option>)}
              </select>
            </div>
          </nav>
          <section className="inference-stage" aria-labelledby="inference-step-title">
            <div className="inference-step-heading" aria-live={playing ? 'off' : 'polite'}>
              <h3 id="inference-step-title">{step.title}</h3>
              <p>{step.description}</p>
            </div>
            {step.tensors.length > 1 && (
              <div className="inference-tensor-select">
                <label htmlFor="inference-tensor">Tensor</label>
                <select id="inference-tensor" value={tensorIndex} onChange={event => {
                  setTransition(null)
                  setTensorIndex(Number(event.target.value))
                }}>
                  {step.tensors.map((item, i) => <option key={item.name} value={i}>{item.name}</option>)}
                </select>
              </div>
            )}
            <TensorTable tensor={tensor} position={position} />
            {finalStep && (
              <section className="inference-prediction" aria-label="Next-token distribution">
                <h4>Predicted next token <output>{trace.prediction}</output></h4>
                <div className="inference-distribution">
                  {VOCABULARY.map((token, i) => ({ token, p: trace.finalProbabilities[i] }))
                    .sort((a, b) => b.p - a.p)
                    .map(({ token, p }) => (
                      <div key={token}>
                        <code>{token}</code>
                        <meter min={0} max={1} value={p} aria-label={`Probability of ${token}`} />
                        <span>{(p * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                </div>
              </section>
            )}
          </section>
          <p className="inference-footer">
            Tiny Recall runs locally on a synthetic recall task. Held-out accuracy · {evaluation.correct}/{evaluation.total} prompts.
          </p>
        </aside>
      </div>
    </section>
  )
}
