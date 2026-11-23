import checkpoint from './tinyRecall.weights.json'

export const VOCABULARY = ['A', 'B', 'C', '1', '2', '3', '?'] as const
export const EXAMPLE_PROMPT = 'A 2 B 1 C 3 ? A'
export const MODEL_WIDTH = 16
export const HEADS = 2
const HEAD_WIDTH = MODEL_WIDTH / HEADS

type Matrix = number[][]
type Token = typeof VOCABULARY[number]

export interface ChannelTensor {
  name: string
  values: Matrix
  rowLabels?: string[]
  columnLabels?: readonly string[]
  probability?: boolean
}

export interface InferenceStep {
  title: string
  description: string
  tensors: ChannelTensor[]
}

const { blocks, ...weights } = checkpoint.weights
export const BLOCK_COUNT = blocks.length

export const evaluation = checkpoint.evaluation
export const parameterCount = [...Object.values(weights), ...blocks.flatMap(block => Object.values(block))].reduce(
  (count, matrix) => count + matrix.length * matrix[0].length, 0,
)

export function tokenize(prompt: string): Token[] {
  const words = prompt.trim().toUpperCase().split(/\s+/)
  const keys = [words[0], words[2], words[4]]
  if (words.length !== 8 || words[6] !== '?' || new Set(keys).size !== 3 ||
    !keys.every(key => ['A', 'B', 'C'].includes(key)) || !keys.includes(words[7]) ||
    ![words[1], words[3], words[5]].every(value => /^[1-3]$/.test(value))) {
    throw new Error('Enter three key–value pairs using A, B, C and 1–3, then ? and a key.')
  }
  return words as Token[]
}

function matmul(a: Matrix, b: Matrix): Matrix {
  return a.map(row => b[0].map((_, column) =>
    row.reduce((sum, value, i) => sum + value * b[i][column], 0)))
}

function transpose(matrix: Matrix): Matrix {
  return matrix[0].map((_, column) => matrix.map(row => row[column]))
}

function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]))
}

function layerNorm(matrix: Matrix): Matrix {
  return matrix.map(row => {
    const mean = row.reduce((sum, n) => sum + n, 0) / row.length
    const variance = row.reduce((sum, n) => sum + (n - mean) ** 2, 0) / row.length
    return row.map(n => (n - mean) / Math.sqrt(variance + 1e-5))
  })
}

function softmax(row: number[]): number[] {
  const peak = Math.max(...row)
  const exp = row.map(n => Math.exp(n - peak))
  const total = exp.reduce((sum, n) => sum + n, 0)
  return exp.map(n => n / total)
}

export function traceInference(prompt: string) {
  const tokens = tokenize(prompt)
  const ids = tokens.map(token => VOCABULARY.indexOf(token))
  const positions = tokens.map((token, i) => `${i} · ${token}`)
  const tensor = (name: string, values: Matrix, columnLabels?: readonly string[]): ChannelTensor =>
    ({ name, values, rowLabels: positions, columnLabels })

  const embeddings = ids.map(id => weights.token[id])
  const initial = add(embeddings, weights.position)
  const decoderSteps: InferenceStep[] = []
  let residual = initial
  for (const [layer, weights] of blocks.entries()) {
    const input = residual
    const normalized = layerNorm(input)
    const query = matmul(normalized, weights.q)
    const key = matmul(normalized, weights.k)
    const value = matmul(normalized, weights.v)

    const heads = Array.from({ length: HEADS }, (_, head) => {
      const split = (matrix: Matrix) => matrix.map(row => row.slice(head * HEAD_WIDTH, (head + 1) * HEAD_WIDTH))
      const q = split(query), k = split(key), v = split(value)
      const scores = matmul(q, transpose(k)).map(row => row.map(s => s / Math.sqrt(HEAD_WIDTH)))
      const masked = scores.map((row, i) => row.map((s, j) => j > i ? -Infinity : s))
      const weights = masked.map(softmax)
      const context = matmul(weights, v)
      return { q, k, v, scores, masked, weights, context }
    })
    const merged = tokens.map((_, i) => heads.flatMap(head => head.context[i]))
    const projection = matmul(merged, weights.out)
    const attentionResidual = add(input, projection)
    const mlpInput = layerNorm(attentionResidual)
    const expanded = matmul(mlpInput, weights.up)
    const activated = expanded.map(row => row.map(n => Math.max(0, n)))
    const reduced = matmul(activated, weights.down)
    const mlpResidual = add(attentionResidual, reduced)
    const attentionStep = (head: typeof heads[number], index: number): InferenceStep => ({
      title: `L${layer + 1} · Attention head ${index + 1}`,
      description: 'Queries score earlier keys using scaled dot products. The causal mask blocks future positions. Softmax gives weights for mixing the value vectors.',
      tensors: [
        { ...tensor('Attention weights', head.weights, positions), probability: true },
        tensor('Queries (Q)', head.q), tensor('Keys (K)', head.k), tensor('Values (V)', head.v),
        tensor('Raw scores', head.scores, positions), tensor('Masked scores', head.masked, positions),
        tensor('Weighted values', head.context),
      ],
    })

    decoderSteps.push(
      {
        title: `L${layer + 1} · Normalize and project`,
        description: 'Normalize each token independently, then project its channels into queries, keys, and values. Each of the two heads receives eight channels.',
        tensors: [tensor('Normalized input', normalized), tensor('Queries', query), tensor('Keys', key), tensor('Values', value)],
      },
      ...heads.map(attentionStep),
      {
        title: `L${layer + 1} · Attention output`,
        description: 'Concatenate the two heads, apply the learned output projection, and add the result to the input residual stream.',
        tensors: [tensor('After residual add', attentionResidual), tensor('Concatenated heads', merged), tensor('Output projection', projection)],
      },
      {
        title: `L${layer + 1} · Feed-forward`,
        description: 'Normalize the residual stream, expand from 16 to 32 channels, apply ReLU, project back to 16 channels, and add the residual.',
        tensors: [
          tensor('After residual add', mlpResidual), tensor('Normalized input', mlpInput),
          tensor('Expanded channels', expanded), tensor('ReLU activations', activated), tensor('Down projection', reduced),
        ],
      },
    )
    residual = mlpResidual
  }
  const finalHidden = layerNorm(residual)
  const logits = matmul(finalHidden, weights.readout)
  const probabilities = logits.map(softmax)
  const finalProbabilities = probabilities[probabilities.length - 1]
  const prediction = VOCABULARY[finalProbabilities.indexOf(Math.max(...finalProbabilities))]

  const steps: InferenceStep[] = [
    {
      title: 'Token IDs',
      description: 'This model has a seven-token vocabulary. Each symbol is converted to its integer ID before embedding lookup.',
      tensors: [tensor('IDs', ids.map(id => [id]), ['ID'])],
    },
    {
      title: 'Embeddings',
      description: 'Look up each token’s learned vector and add its learned position vector. The resulting 16 channels enter the first decoder block.',
      tensors: [tensor('Token + position', initial), tensor('Token embeddings', embeddings), tensor('Position embeddings', weights.position)],
    },
    ...decoderSteps,
    {
      title: 'Token logits',
      description: 'After final normalization, the learned readout projects each position to one score per vocabulary token. The last row scores the next token for the whole prompt.',
      tensors: [tensor('Logits', logits, VOCABULARY), tensor('Final hidden state', finalHidden)],
    },
    {
      title: 'Next-token probabilities',
      description: 'Softmax converts the logits into probabilities. The token with the highest probability in the last row is the model’s greedy prediction.',
      tensors: [{ ...tensor('Probabilities', probabilities, VOCABULARY), probability: true }],
    },
  ]

  return { tokens, ids, steps, prediction, finalProbabilities, logits }
}
