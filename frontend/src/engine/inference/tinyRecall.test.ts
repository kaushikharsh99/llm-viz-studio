import { describe, expect, it } from 'vitest'
import checkpoint from './tinyRecall.weights.json'
import { EXAMPLE_PROMPT, traceInference } from './tinyRecall'

describe('Tiny Recall inference', () => {
  it('matches the final logits from the independent PyTorch implementation', () => {
    const trace = traceInference(EXAMPLE_PROMPT)
    const last = trace.logits[trace.logits.length - 1]
    last.forEach((value, i) => expect(value).toBeCloseTo(checkpoint.referenceFinalLogits[i], 4))
    expect(trace.prediction).toBe('2')
    expect(traceInference('C 2 A 3 B 1 ? C').prediction).toBe('2')
    expect(traceInference('C 1 A 3 B 2 ? C').prediction).toBe('1')
  })

  it('masks all future tokens and produces normalized attention and output rows', () => {
    const trace = traceInference(EXAMPLE_PROMPT)
    for (const step of trace.steps.filter(step => step.title.includes('Attention head'))) {
      const attention = step.tensors[0].values
      attention.forEach((row, i) => {
        expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1)
        expect(row.slice(i + 1).every(value => value === 0)).toBe(true)
      })
    }
    expect(trace.finalProbabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1)
    expect(() => traceInference('A 2 B 1 C 3 A')).toThrow(/three key–value pairs/)
  })
})
