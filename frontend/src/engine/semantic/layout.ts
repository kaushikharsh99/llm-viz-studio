import type { ArchSpec } from '../spec'
import type { SemanticNode, WorkspaceState, DiagramLayout, LayoutedNode, DetailPreset } from './types'
import { fmtCount } from '../render/svg'

const TCX = 470
const OUTER_X = 150
const OUTER_W = 640
const BLK_X = 215
const BLK_W = 510
const RAIL_X = BLK_X + BLK_W - 55

export function isNodeExpanded(
  nodeId: string,
  nodeKind: string,
  preset: DetailPreset,
  expandedIds: Set<string>,
  collapsedIds: Set<string>
): boolean {
  if (collapsedIds.has(nodeId)) return false
  if (expandedIds.has(nodeId)) return true

  // Preset-based defaults
  if (preset === 1) {
    return false // Overview: modules collapsed by default
  }
  if (preset === 2) {
    // Expand primary modules
    return nodeKind === 'attention' || nodeKind === 'ffn' || nodeKind === 'moe'
  }
  if (preset === 3) {
    // Expand down to operations
    return (
      nodeKind === 'attention' ||
      nodeKind === 'ffn' ||
      nodeKind === 'moe' ||
      nodeKind === 'q_proj' ||
      nodeKind === 'gate_proj' ||
      nodeKind === 'moe_expert_group'
    )
  }
  return false
}

function vocabStr(n: number): string {
  return `${Math.round(n / 1000)}k`
}

function ctxStr(n: number): string {
  if (n >= 16384 && n % 1024 === 0) {
    return `${Math.floor(n / 1024)}k`
  }
  return n.toLocaleString()
}

/**
 * Compute the dynamic layout of the architectural diagram
 */
export function computeDiagramLayout(
  root: SemanticNode,
  spec: ArchSpec,
  state: WorkspaceState
): DiagramLayout {
  const allNodes = new Map<string, LayoutedNode>()
  const { preset, expandedIds, collapsedIds, focusedNodeId } = state

  // =========================================================================
  // FOCUS MODE (ISOLATED VIEW OF A SELECTED SUB-ARCHITECTURE)
  // =========================================================================
  if (focusedNodeId) {
    let focusTarget: SemanticNode | null = null

    function findNode(n: SemanticNode): void {
      if (n.id === focusedNodeId) {
        focusTarget = n
        return
      }
      if (n.children) {
        n.children.forEach(findNode)
      }
    }
    findNode(root)

    if (focusTarget) {
      return computeFocusLayout(focusTarget, spec, state)
    }
  }

  // =========================================================================
  // FULL MODEL DIAGRAM LAYOUT (PRESETS 1, 2, 3 + MANUAL EXPANSION)
  // =========================================================================
  const L = state.selectedLayer || 0

  // Find layer block in root
  let layerBlock: SemanticNode | undefined
  function findLayer(n: SemanticNode): void {
    if (n.kind === 'layer_block') {
      layerBlock = n
      return
    }
    if (n.children) {
      for (const ch of n.children) {
        findLayer(ch)
        if (layerBlock) return
      }
    }
  }
  findLayer(root)

  const childrenList: SemanticNode[] = layerBlock?.children || []
  const norm1Node = childrenList.find(c => c.kind === 'norm_1')
  const attnNode = childrenList.find(c => c.kind === 'attention')
  const res1Node = childrenList.find(c => c.kind === 'residual_add_1')
  const norm2Node = childrenList.find(c => c.kind === 'norm_2')
  const ffnNode = childrenList.find(c => c.kind === 'ffn' || c.kind === 'moe')
  const res2Node = childrenList.find(c => c.kind === 'residual_add_2')

  const isAttnExpanded = attnNode
    ? isNodeExpanded(attnNode.id, attnNode.kind, preset, expandedIds, collapsedIds)
    : false
  const isFfnExpanded = ffnNode
    ? isNodeExpanded(ffnNode.id, ffnNode.kind, preset, expandedIds, collapsedIds)
    : false

  // Layout from top to bottom
  const cx = TCX
  let y = 140

  // 1. Title is drawn at y=74
  // 2. Linear Output Layer (LM Head)
  const headNode = root.children?.find(c => c.kind === 'lm_head')
  const headW = 310, headH = 52
  const { children: _hCh, ...headProps } = headNode || {
    id: 'lm_head',
    kind: 'lm_head' as const,
    label: 'Linear output layer',
    is_expandable: true,
    default_detail_level: 1 as const,
  }
  const layoutedHead: LayoutedNode = {
    ...headProps,
    x: cx - headW / 2,
    y: 261,
    width: headW,
    height: headH,
    isExpanded: false,
    cx,
    topY: 261,
    bottomY: 261 + headH,
    leftX: cx - headW / 2,
    rightX: cx + headW / 2,
  }
  allNodes.set(layoutedHead.id, layoutedHead)

  // 3. Final Normalization
  const finNorm = root.children?.find(c => c.kind === 'final_norm')
  const finY = 261 + headH + 38
  const finW = 260, finH = 50
  const { children: _fnCh, ...finProps } = finNorm || {
    id: 'final_norm',
    kind: 'final_norm' as const,
    label: spec.norm_type === 'rmsnorm' ? 'Final RMSNorm' : 'Final LayerNorm',
    is_expandable: false,
    default_detail_level: 1 as const,
  }
  const layoutedFinNorm: LayoutedNode = {
    ...finProps,
    x: cx - finW / 2,
    y: finY,
    width: finW,
    height: finH,
    isExpanded: false,
    cx,
    topY: finY,
    bottomY: finY + finH,
    leftX: cx - finW / 2,
    rightX: cx + finW / 2,
  }
  allNodes.set(layoutedFinNorm.id, layoutedFinNorm)

  // 4. Repeated Transformer Block Container
  const blkTop = finY + finH + 42
  y = blkTop + 34

  // Inside Layer Block:
  // Plus 2 (Residual Add 2)
  const plus2Y = y + 16
  const { children: _r2Ch, ...res2Props } = res2Node || {
    id: `layer_${L}_res2`,
    kind: 'residual_add_2' as const,
    label: '⊕ Residual Connection 2',
    is_expandable: false,
    default_detail_level: 1 as const,
  }
  const layoutedRes2: LayoutedNode = {
    ...res2Props,
    x: cx - 15,
    y: plus2Y - 15,
    width: 30,
    height: 30,
    isExpanded: false,
    cx,
    topY: plus2Y - 15,
    bottomY: plus2Y + 15,
    leftX: cx - 15,
    rightX: cx + 15,
  }
  allNodes.set(layoutedRes2.id, layoutedRes2)
  y = plus2Y + 36

  // FFN / MoE Block
  const ffnTop = y
  let ffnH = 48
  let ffnW = 210
  const ffnChildrenLayouted: LayoutedNode[] = []

  if (isFfnExpanded && ffnNode?.children) {
    ffnW = 420
    let childY = ffnTop + 24
    for (const ch of ffnNode.children) {
      const isDeep = ch.default_detail_level === 3
      if (isDeep && preset < 3 && !expandedIds.has(ch.id)) {
        continue
      }
      const chH = 44
      const chW = 380
      const { children: _cch, ...chProps } = ch
      const lch: LayoutedNode = {
        ...chProps,
        x: cx - chW / 2,
        y: childY,
        width: chW,
        height: chH,
        isExpanded: false,
        cx,
        topY: childY,
        bottomY: childY + chH,
        leftX: cx - chW / 2,
        rightX: cx + chW / 2,
      }
      ffnChildrenLayouted.push(lch)
      allNodes.set(lch.id, lch)
      childY += chH + 12
    }
    ffnH = childY - ffnTop + 14
  }

  const { children: _fch, ...ffnProps } = ffnNode || {
    id: `layer_${L}_ffn`,
    kind: spec.moe ? 'moe' : 'ffn',
    label: spec.moe ? 'MoE' : 'Feed forward',
    is_expandable: true,
    default_detail_level: 1 as const,
  }
  const layoutedFfn: LayoutedNode = {
    ...ffnProps,
    x: cx - ffnW / 2,
    y: ffnTop,
    width: ffnW,
    height: ffnH,
    isExpanded: isFfnExpanded,
    children: ffnChildrenLayouted,
    cx,
    topY: ffnTop,
    bottomY: ffnTop + ffnH,
    leftX: cx - ffnW / 2,
    rightX: cx + ffnW / 2,
  }
  allNodes.set(layoutedFfn.id, layoutedFfn)
  y = ffnTop + ffnH + 34

  // Norm 2
  const norm2H = 48, norm2W = 240
  const { children: _n2Ch, ...norm2Props } = norm2Node || {
    id: `layer_${L}_norm2`,
    kind: 'norm_2' as const,
    label: spec.norm_type === 'rmsnorm' ? 'RMSNorm 2' : 'LayerNorm 2',
    is_expandable: false,
    default_detail_level: 1 as const,
  }
  const layoutedNorm2: LayoutedNode = {
    ...norm2Props,
    x: cx - norm2W / 2,
    y,
    width: norm2W,
    height: norm2H,
    isExpanded: false,
    cx,
    topY: y,
    bottomY: y + norm2H,
    leftX: cx - norm2W / 2,
    rightX: cx + norm2W / 2,
  }
  allNodes.set(layoutedNorm2.id, layoutedNorm2)
  y += norm2H + 28

  // Plus 1 (Residual Add 1)
  const plus1Y = y + 16
  const { children: _r1Ch, ...res1Props } = res1Node || {
    id: `layer_${L}_res1`,
    kind: 'residual_add_1' as const,
    label: '⊕ Residual Connection 1',
    is_expandable: false,
    default_detail_level: 1 as const,
  }
  const layoutedRes1: LayoutedNode = {
    ...res1Props,
    x: cx - 15,
    y: plus1Y - 15,
    width: 30,
    height: 30,
    isExpanded: false,
    cx,
    topY: plus1Y - 15,
    bottomY: plus1Y + 15,
    leftX: cx - 15,
    rightX: cx + 15,
  }
  allNodes.set(layoutedRes1.id, layoutedRes1)
  y = plus1Y + 36

  // Attention Block
  const attnTop = y
  let attnH = 96
  let attnW = 310
  const attnChildrenLayouted: LayoutedNode[] = []

  if (isAttnExpanded && attnNode?.children) {
    attnW = 420
    let childY = attnTop + 24
    for (const ch of attnNode.children) {
      const isDeep = ch.default_detail_level === 3
      if (isDeep && preset < 3 && !expandedIds.has(ch.id)) {
        continue
      }
      const chH = 44
      const chW = 380
      const { children: _ach, ...chProps } = ch
      const lch: LayoutedNode = {
        ...chProps,
        x: cx - chW / 2,
        y: childY,
        width: chW,
        height: chH,
        isExpanded: false,
        cx,
        topY: childY,
        bottomY: childY + chH,
        leftX: cx - chW / 2,
        rightX: cx + chW / 2,
      }
      attnChildrenLayouted.push(lch)
      allNodes.set(lch.id, lch)
      childY += chH + 12
    }
    attnH = childY - attnTop + 14
  }

  const { children: _achTop, ...attnProps } = attnNode || {
    id: `layer_${L}_attention`,
    kind: 'attention' as const,
    label: 'Self-Attention',
    is_expandable: true,
    default_detail_level: 1 as const,
  }
  const layoutedAttn: LayoutedNode = {
    ...attnProps,
    x: cx - attnW / 2,
    y: attnTop,
    width: attnW,
    height: attnH,
    isExpanded: isAttnExpanded,
    children: attnChildrenLayouted,
    cx,
    topY: attnTop,
    bottomY: attnTop + attnH,
    leftX: cx - attnW / 2,
    rightX: cx + attnW / 2,
  }
  allNodes.set(layoutedAttn.id, layoutedAttn)
  y = attnTop + attnH + 34

  // Norm 1
  const norm1H = 48, norm1W = 240
  const { children: _n1Ch, ...norm1Props } = norm1Node || {
    id: `layer_${L}_norm1`,
    kind: 'norm_1' as const,
    label: spec.norm_type === 'rmsnorm' ? 'RMSNorm 1' : 'LayerNorm 1',
    is_expandable: false,
    default_detail_level: 1 as const,
  }
  const layoutedNorm1: LayoutedNode = {
    ...norm1Props,
    x: cx - norm1W / 2,
    y,
    width: norm1W,
    height: norm1H,
    isExpanded: false,
    cx,
    topY: y,
    bottomY: y + norm1H,
    leftX: cx - norm1W / 2,
    rightX: cx + norm1W / 2,
  }
  allNodes.set(layoutedNorm1.id, layoutedNorm1)
  const norm1Bot = y + norm1H
  y += norm1H + 28

  const blkBot = y
  const blkH = blkBot - blkTop

  // 5. Token Embedding Layer
  const embNode = root.children?.find(c => c.kind === 'embedding')
  const embTop = blkBot + 52
  const embW = 340, embH = 52
  const { children: _eCh, ...embProps } = embNode || {
    id: 'embed_tokens',
    kind: 'embedding' as const,
    label: 'Token embedding layer',
    is_expandable: true,
    default_detail_level: 1 as const,
  }
  const layoutedEmb: LayoutedNode = {
    ...embProps,
    x: cx - embW / 2,
    y: embTop,
    width: embW,
    height: embH,
    isExpanded: false,
    cx,
    topY: embTop,
    bottomY: embTop + embH,
    leftX: cx - embW / 2,
    rightX: cx + embW / 2,
  }
  allNodes.set(layoutedEmb.id, layoutedEmb)

  const outerTop = 165
  const outerBot = embTop + embH + 44
  const outerH = outerBot - outerTop

  // 6. Tokenized Text & Sample Input
  const tokY = outerBot + 66
  const tokW = 250, tokH = 50
  const layoutedTok: LayoutedNode = {
    id: 'tokens_io',
    kind: 'tokenized_text',
    label: 'Tokenized text',
    is_expandable: false,
    default_detail_level: 1,
    x: cx - tokW / 2,
    y: tokY,
    width: tokW,
    height: tokH,
    isExpanded: false,
    cx,
    topY: tokY,
    bottomY: tokY + tokH,
    leftX: cx - tokW / 2,
    rightX: cx + tokW / 2,
  }
  allNodes.set(layoutedTok.id, layoutedTok)

  const totalHeight = tokY + tokH + 140
  const isMoeWithInset = spec.moe && !isFfnExpanded
  const totalWidth = isMoeWithInset ? 1540 : 1100

  // Residual bypass rails
  const residualRails = [
    {
      startCx: cx,
      startY: norm1Bot + 20,
      railX: RAIL_X,
      endY: plus1Y,
      endCx: cx + 17,
    },
    {
      startCx: cx,
      startY: layoutedNorm2.bottomY + 18,
      railX: RAIL_X,
      endY: plus2Y,
      endCx: cx + 17,
    },
  ]

  // Callouts
  const callouts: DiagramLayout['callouts'] = []

  // Vocab callout
  if (spec.vocab_size) {
    callouts.push({
      x: cx + 190,
      y: 148,
      lines: [[['Vocabulary size of ', ''], [vocabStr(spec.vocab_size), 'accent']]],
      targetX: cx + 140,
      targetY: 258,
    })
  }

  // Attention heads callout
  const a = spec.attention
  const headLines: [string, string][][] = [[[`${a.num_heads} `, 'accent'], ['heads', '']]]
  if (a.kind === 'GQA' || a.kind === 'MQA') {
    headLines.push([[`${a.num_kv_heads} `, 'accent'], ['KV groups', '']])
  }
  if (a.sliding_window) {
    headLines.push([['sliding window of ', ''], [fmtCount(a.sliding_window), 'accent'], [' tokens', '']])
  }
  if (a.qk_norm) {
    headLines.push([['QK-Norm on queries & keys', '']])
  }
  if (spec.positional === 'alibi') {
    headLines.push([['ALiBi positional bias', '']])
  }
  callouts.push({
    x: cx + 330,
    y: embTop + 44,
    lines: headLines,
    targetX: cx + 152,
    targetY: layoutedAttn.bottomY - 8,
  })

  // Context length callout
  if (spec.context_length) {
    callouts.push({
      x: 18,
      y: blkBot + 66,
      lines: [
        [['Supported', '']],
        [['context length', '']],
        [['of ', ''], [ctxStr(spec.context_length), 'accent'], [' tokens', '']],
      ],
      targetX: spec.positional === 'rope' ? 58 : cx - 184,
      targetY: spec.positional === 'rope' ? layoutedAttn.y + layoutedAttn.height / 2 + 20 : embTop + 30,
    })
  }

  return {
    width: totalWidth,
    height: totalHeight,
    viewBox: `0 0 ${totalWidth} ${totalHeight}`,
    rootNodes: [
      layoutedHead,
      layoutedFinNorm,
      layoutedRes2,
      layoutedFfn,
      layoutedNorm2,
      layoutedRes1,
      layoutedAttn,
      layoutedNorm1,
      layoutedEmb,
      layoutedTok,
    ],
    allNodes,
    residualRails,
    callouts,
    outerContainer: {
      x: OUTER_X,
      y: outerTop,
      w: OUTER_W,
      h: outerH,
    },
    repeatedBlockContainer: {
      x: BLK_X,
      y: blkTop,
      w: BLK_W,
      h: blkH,
      numLayers: spec.num_layers,
    },
  }
}

/**
 * Focus Mode layout: renders an isolated subsystem in deep detail
 */
function computeFocusLayout(
  focusNode: SemanticNode,
  spec: ArchSpec,
  state: WorkspaceState
): DiagramLayout {
  const allNodes = new Map<string, LayoutedNode>()
  const cx = 450
  let y = 140
  const children = focusNode.children || []

  const layoutedChildren: LayoutedNode[] = []
  for (const ch of children) {
    const chH = 52
    const chW = 420
    const { children: _cch, ...chProps } = ch
    const lch: LayoutedNode = {
      ...chProps,
      x: cx - chW / 2,
      y,
      width: chW,
      height: chH,
      isExpanded: false,
      cx,
      topY: y,
      bottomY: y + chH,
      leftX: cx - chW / 2,
      rightX: cx + chW / 2,
    }
    layoutedChildren.push(lch)
    allNodes.set(lch.id, lch)
    y += chH + 28
  }

  const focusW = 560
  const focusH = y - 110
  const { children: _fCh, ...focusProps } = focusNode
  const rootLayouted: LayoutedNode = {
    ...focusProps,
    x: cx - focusW / 2,
    y: 110,
    width: focusW,
    height: focusH,
    isExpanded: true,
    children: layoutedChildren,
    cx,
    topY: 110,
    bottomY: 110 + focusH,
    leftX: cx - focusW / 2,
    rightX: cx + focusW / 2,
  }
  allNodes.set(rootLayouted.id, rootLayouted)

  const totalWidth = 900
  const totalHeight = Math.max(650, y + 100)

  return {
    width: totalWidth,
    height: totalHeight,
    viewBox: `0 0 ${totalWidth} ${totalHeight}`,
    rootNodes: [rootLayouted],
    allNodes,
    residualRails: [],
    callouts: [],
  }
}
