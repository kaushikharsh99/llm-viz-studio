export interface DiagramCamera {
  yaw: number
  pitch: number
}

export const DEFAULT_CAMERA: DiagramCamera = { yaw: -32, pitch: -23 }

export const ROTATION_KEYS = new Map([
  ['ArrowLeft', [-8, 0]],
  ['ArrowRight', [8, 0]],
  ['ArrowUp', [0, -8]],
  ['ArrowDown', [0, 8]],
])

export function rotateCamera(camera: DiagramCamera, yaw: number, pitch: number): DiagramCamera {
  return {
    yaw: ((camera.yaw + yaw) % 360 + 360) % 360,
    pitch: Math.max(-55, Math.min(55, camera.pitch + pitch)),
  }
}

export function orbitProjection(camera: DiagramCamera, centerX: number, centerY: number) {
  const yaw = camera.yaw * Math.PI / 180
  const pitch = camera.pitch * Math.PI / 180
  const sinYaw = Math.sin(yaw), cosYaw = Math.cos(yaw)
  const sinPitch = Math.sin(pitch), cosPitch = Math.cos(pitch)
  return ([x, y, z]: [number, number, number]) => {
    const horizontal = x - centerX
    const vertical = y - centerY
    const depth = -horizontal * sinYaw + z * cosYaw
    return {
      x: centerX + horizontal * cosYaw + z * sinYaw,
      y: centerY + vertical * cosPitch - depth * sinPitch,
      depth: vertical * sinPitch + depth * cosPitch,
    }
  }
}
