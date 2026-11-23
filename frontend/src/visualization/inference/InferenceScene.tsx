import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { DEFAULT_CAMERA, orbitProjection, ROTATION_KEYS, rotateCamera } from '../../engine/semantic/orbit'
import type { ChannelTensor, InferenceStep } from '../../engine/inference/tinyRecall'
import AcceleratorBoard from './AcceleratorBoard'

const WIDTH = 940
const HEIGHT = 780
const MID = 385
const LEVELS = [100, 232, 364, 496, 628]
const LABELS = ['Token IDs', 'Embeddings', 'Decoder block 1', 'Decoder block 2', 'Readout']
const REPRESENTATIVE_STEPS = [0, 1, 6, 11, 13]

function deck(step: number) {
  if (step < 2) return step
  if (step < 7) return 2
  if (step < 12) return 3
  return 4
}

export interface InferenceTransition {
  id: number
  from: number
  to: number
  departure: ChannelTensor
}

interface SceneProps {
  steps: InferenceStep[]
  tensor: ChannelTensor
  tokens: readonly string[]
  step: number
  position: number
  transition: InferenceTransition | null
  onStep: (step: number) => void
  playback: ReactNode
}

function format(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : '−∞'
}

export default function InferenceScene({ steps, tensor, tokens, step, position, transition, onStep, playback }: SceneProps) {
  const [camera, setCamera] = useState(DEFAULT_CAMERA)
  const packets = useRef<SVGGElement>(null)
  useLayoutEffect(() => {
    packets.current?.querySelectorAll<SVGAnimationElement>('animate, animateMotion').forEach(animation => animation.beginElement?.())
  }, [transition?.id])
  const drag = useRef<{ id: number; x: number; y: number; camera: typeof camera } | null>(null)
  const project = orbitProjection(camera, MID, HEIGHT / 2)
  const xy = ({ x, y }: ReturnType<typeof project>) => `${x},${y}`
  const active = deck(step)
  const boards = LEVELS.map((level, i) => ({
    level,
    index: i,
    tensor: i === active ? tensor : steps[REPRESENTATIVE_STEPS[i]].tensors[0],
  })).sort((a, b) => project([MID, a.level, 0]).depth - project([MID, b.level, 0]).depth)
  const route = LEVELS.map(y => xy(project([MID, y, 0]))).join(' ')

  return (
    <section className="inference-scene" aria-labelledby="inference-scene-title">
      <div className="inference-scene-header">
        <div><p className="inference-eyebrow">TINY RECALL</p><h3 id="inference-scene-title">Inference in 3D</h3></div>
        <button type="button" className="toolbar-btn" onClick={() => setCamera(DEFAULT_CAMERA)}>Reset angle</button>
      </div>
      <svg
        className="inference-scene-canvas"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="group"
        aria-label="Rotatable inference canvas"
        aria-describedby="inference-orbit-help"
        tabIndex={0}
        onPointerDown={event => {
          if (!event.isPrimary || event.button !== 0) return
          drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, camera }
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={event => {
          const start = drag.current
          if (start?.id === event.pointerId) {
            setCamera(rotateCamera(start.camera, (event.clientX - start.x) * 0.42, (event.clientY - start.y) * 0.42))
          }
        }}
        onLostPointerCapture={() => { drag.current = null }}
        onKeyDown={event => {
          const delta = ROTATION_KEYS.get(event.key)
          if (delta && event.target === event.currentTarget) {
            event.preventDefault()
            setCamera(camera => rotateCamera(camera, delta[0], delta[1]))
          }
        }}
      >
        <polyline points={route} className="inference-spine" />
        {boards.map(({ level, index, tensor: board }) => {
          const columns = board.values[0].length
          const selected = index === active
          const cellWidth = Math.min(39, 300 / columns)
          const xStart = MID - columns * cellWidth / 2
          const rowDepth = 13
          const zStart = -tokens.length * rowDepth / 2
          const range = Math.max(1e-9, ...board.values.flat().filter(Number.isFinite).map(Math.abs))
          return (
            <AcceleratorBoard key={index} project={project} center={MID} level={level}
              selected={selected} surfaceWidth={columns * cellWidth}>
              {board.values.flatMap((row, r) => row.map((value, c) => {
                const x = xStart + c * cellWidth + 1, z = zStart + r * rowDepth + 1
                const points = [[x, level - 0.3, z], [x + cellWidth - 2, level - 0.3, z],
                  [x + cellWidth - 2, level - 0.3, z + rowDepth - 2], [x, level - 0.3, z + rowDepth - 2]] as [number, number, number][]
                return (
                  <polygon key={`${r}-${c}`} points={points.map(project).map(xy).join(' ')}
                    fill={!Number.isFinite(value) ? '#535c72' : value < 0 ? '#faac65' : '#53d6bd'}
                    fillOpacity={!Number.isFinite(value) ? 0.3 : 0.11 + 0.84 * Math.abs(value) / range}
                    stroke={selected && r === position ? '#e1fff9' : 'none'} strokeWidth="0.8">
                    <title>{`Position ${r} · ${board.columnLabels?.[c] ?? `column ${c}`} · ${format(value)}`}</title>
                  </polygon>
                )
              }))}
            </AcceleratorBoard>
          )
        })}
        {LEVELS.map((y, index) => {
          const p = project([MID, y, 0])
          return (
            <g key={index} className={index === active ? 'inference-deck-caption is-current' : 'inference-deck-caption'}>
              <line x1={Math.min(585, Math.max(...[-205, 205].flatMap(x => [-76, 76].map(z => project([MID + x, y, z]).x))) + 8)} y1={p.y} x2={590} y2={p.y} />
              <text x={606} y={p.y - 4}>{LABELS[index]}</text>
              {index === active && <text className="inference-deck-detail" x={606} y={p.y + 17}>
                {tensor.name} · {tensor.values.length} × {tensor.values[0].length}
              </text>}
            </g>
          )
        })}
        {transition && transition.from !== transition.to && (
          <g key={transition.id} ref={packets} className="inference-moving-packets" data-testid="moving-token-packets" aria-hidden="true">
            {tokens.map((token, row) => {
              const from = deck(transition.from), to = deck(transition.to)
              const z = (row - (tokens.length - 1) / 2) * 13
              const x = MID - 55 + row * 13
              const start = project([x, LEVELS[from], z])
              const end = project([x, LEVELS[to], z])
              const middle = project([x - (from === to ? 155 : 62), (LEVELS[from] + LEVELS[to]) / 2 - 32, z])
              const path = `M ${xy(start)} Q ${xy(middle)} ${xy(end)}`
              return (
                <g key={row} className={row === position ? 'packet packet-selected' : 'packet'}>
                  <animateMotion begin="indefinite" path={path} dur="850ms" fill="freeze" />
                  <animate begin="indefinite" attributeName="opacity" values="1;1;0" keyTimes="0;0.85;1" dur="970ms" fill="freeze" />
                  <circle r={row === position ? 8 : 4} />
                  {row === position && (
                    <text x="13" y="-11">
                      <tspan>{token} · col 0 · {format(transition.departure.values[row][0])}
                        <animate begin="indefinite" attributeName="opacity" values="1;1;0" keyTimes="0;0.72;1" dur="820ms" fill="freeze" />
                      </tspan>
                      <tspan x="13" opacity="0">{token} · col 0 · {format(tensor.values[row][0])}
                        <animate begin="indefinite" attributeName="opacity" values="0;0;1" keyTimes="0;0.75;1" dur="820ms" fill="freeze" />
                      </tspan>
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        )}
      </svg>
      <div className="inference-scene-playback">{playback}</div>
      <div className="inference-scene-footer">
        <p id="inference-orbit-help">Drag or use arrow keys to orbit</p>
        <div className="inference-layer-jumps" aria-label="Jump to layer">
          {LABELS.map((label, i) => (
            <button type="button" onClick={() => onStep(REPRESENTATIVE_STEPS[i])} key={label}
              aria-label={`Show ${label}`} aria-pressed={i === active}>{i + 1}</button>
          ))}
        </div>
        <p>Teal = positive · amber = negative</p>
      </div>
    </section>
  )
}
