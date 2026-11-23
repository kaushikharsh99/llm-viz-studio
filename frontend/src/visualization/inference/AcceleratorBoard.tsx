import type { ReactNode } from 'react'
import type { orbitProjection } from '../../engine/semantic/orbit'

type Point = [number, number, number]
type Project = ReturnType<typeof orbitProjection>
type Projected = ReturnType<Project>
type Material = { top: string; long: string; short: string; bottom: string }

const GRAPHITE = { top: '#29343d', long: '#111c27', short: '#374653', bottom: '#22313d' }
const ALUMINUM = { top: '#a2b1b9', long: '#475966', short: '#607784', bottom: '#586c78' }
const SILICON = { top: '#455662', long: '#263846', short: '#192936', bottom: '#243645' }
const SOCKET = { top: '#253542', long: '#142331', short: '#354c5c', bottom: '#263846' }
const DIE = { top: '#132831', long: '#163940', short: '#23454d', bottom: '#17323b' }

const outline = (points: Projected[]) => points.map(({ x, y }) => `${x},${y}`).join(' ')

function faceArea(points: Projected[]) {
  return points.reduce((area, point, i) => {
    const next = points[(i + 1) % points.length]
    return area + point.x * next.y - next.x * point.y
  }, 0)
}

function footprint(x: number, y: number, z: number, width: number, depth: number, bevel: number): Point[] {
  const w = width / 2, d = depth / 2
  if (!bevel) return [[x - w, y, z - d], [x + w, y, z - d], [x + w, y, z + d], [x - w, y, z + d]]
  return [
    [x - w + bevel, y, z - d], [x + w - bevel, y, z - d],
    [x + w, y, z - d + bevel], [x + w, y, z + d - bevel],
    [x + w - bevel, y, z + d], [x - w + bevel, y, z + d],
    [x - w, y, z + d - bevel], [x - w, y, z - d + bevel],
  ]
}

interface Part {
  id: string
  x: number
  z: number
  y: number
  width: number
  depth: number
  height: number
  material: Material
  bevel?: number
}

function BoardPart({ part, project }: { part: Part; project: Project }) {
  const { x, y, z, width, depth, height, material, bevel = 0 } = part
  const top = footprint(x, y, z, width, depth, bevel).map(project)
  const bottom = footprint(x, y + height, z, width, depth, bevel).map(project)
  const faces = [
    { points: top, fill: material.top },
    { points: bottom.slice().reverse(), fill: material.bottom },
    ...top.map((point, i) => {
      const next = (i + 1) % top.length
      const isLong = i === 0 || i === top.length / 2
      return { points: [top[next], point, bottom[i], bottom[next]], fill: isLong ? material.long : material.short }
    }),
  ]
  return (
    <g className={`accelerator-part accelerator-${part.id.split('-')[0]}`}>
      {faces.map((face, index) => faceArea(face.points) > 0.001 && (
        <polygon key={index} points={outline(face.points)} fill={face.fill} />
      ))}
    </g>
  )
}

interface Props {
  project: Project
  center: number
  level: number
  selected: boolean
  surfaceWidth: number
  children: ReactNode
}

export default function AcceleratorBoard({ project, center, level, selected, surfaceWidth, children }: Props) {
  const surfaceVisible = faceArea(footprint(center, level, 0, 12, 12, 0).map(project)) > 0
  const part = (id: string, x: number, y: number, z: number, width: number, depth: number, height: number,
    material: Material, bevel = 0): Part =>
    ({ id, x: center + x, y: level + y, z, width, depth, height, material, bevel })
  const led = selected ? '#8bffe0' : '#4b9f91'
  const parts = [
    part('substrate', 0, 8, 0, 410, 151, 10, GRAPHITE, 13),
    part('backplate', 0, 18, 0, 344, 115, 4, ALUMINUM, 12),
  ]
  const components = [
    part('carrier', 0, 4, 0, 324, 120, 4, ALUMINUM, 8),
    part('die', 0, 0, 0, surfaceWidth + 12, 114, 4, DIE, 5),
    part('connector-left', -184, 0, 0, 30, 88, 8, SOCKET, 4),
    part('connector-right', 184, -1, 0, 31, 88, 9, SOCKET, 4),
    ...[-1, 1].flatMap(side => [-1, 1].flatMap(edge => [
      part(`fastener-${side}-${edge}`, side * 183, 5, edge * 60, 12, 12, 3, ALUMINUM, 4),
      part(`contact-${side}-${edge}`, side * 131, 5, edge * 67, 43, 5, 3, SILICON),
    ])),
    ...Array.from({ length: 7 }, (_, i) =>
      part(`fin-${i}`, 184, -5, (i - 3) * 10, 23, 3, 4, SILICON)),
    ...Array.from({ length: 6 }, (_, i) =>
      part(`pin-${i}`, -187, -2, (i - 2.5) * 11, 14, 3, 2, { ...SOCKET, top: '#ba9261' })),
    ...Array.from({ length: 3 }, (_, i) =>
      part(`light-${i}`, 94 + i * 13, 3, 64, 5, 5, 2, { ...SOCKET, top: led })),
  ]
  return (
    <g className={`inference-board accelerator-board${selected ? ' is-current' : ''}`}>
      {parts.sort((a, b) => project([a.x, a.y, a.z]).depth - project([b.x, b.y, b.z]).depth)
        .map(part => <BoardPart key={part.id} part={part} project={project} />)}
      {surfaceVisible && (
        <>
          {components.map(part => <BoardPart key={part.id} part={part} project={project} />)}
          {children}
          {selected && <polygon className="accelerator-active-outline"
            points={outline(footprint(center, -0.4 + level, 0, surfaceWidth + 10, 112, 5).map(project))} />}
        </>
      )}
    </g>
  )
}
