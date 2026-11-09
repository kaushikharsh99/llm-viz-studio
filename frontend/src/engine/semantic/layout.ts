import type { ArchSpec } from '../spec'
import type { SemanticNode, WorkspaceState, DiagramLayout, LayoutedNode, DetailPreset, DataflowEdge, MagnifiedInset } from './types'
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
 * Layout internal computational graph for Attention (GQA / MQA / MHA / MLA)
 */
function layoutAttentionInternals(
  attnNode: SemanticNode,
  spec: ArchSpec,
  cx: number,
  topY: number,
  preset: DetailPreset,
  expandedIds: Set<string>,
  allNodes: Map<string, LayoutedNode>,
  width: number = 700
): { height: number; width: number; children: LayoutedNode[]; edges: DataflowEdge[] } {
  const children = attnNode.children || []
  const layoutedChildren: LayoutedNode[] = []
  const edges: DataflowEdge[] = []
  const isDeep = preset === 3 || children.some(c => expandedIds.has(c.id))

  function addNode(node: SemanticNode, nCx: number, nY: number, nW: number, nH: number): LayoutedNode {
    const { children: _c, ...props } = node
    const ln: LayoutedNode = {
      ...props,
      x: nCx - nW / 2,
      y: nY,
      width: nW,
      height: nH,
      isExpanded: false,
      cx: nCx,
      topY: nY,
      bottomY: nY + nH,
      leftX: nCx - nW / 2,
      rightX: nCx + nW / 2,
    }
    layoutedChildren.push(ln)
    allNodes.set(ln.id, ln)
    return ln
  }

  const qNode = children.find(c => c.kind === 'q_proj')
  const kNode = children.find(c => c.kind === 'k_proj')
  const vNode = children.find(c => c.kind === 'v_proj')
  const oNode = children.find(c => c.kind === 'o_proj')
  const ropeNode = children.find(c => c.kind === 'rope_op')
  const qkNormNode = children.find(c => c.kind === 'qk_norm')
  const scoresNode = children.find(c => c.kind === 'attn_scores')
  const softmaxNode = children.find(c => c.kind === 'softmax_op')
  const ctxNode = children.find(c => c.kind === 'attn_context')

  const isMLA = spec.attention?.kind === 'MLA'
  if (isMLA) {
    const qaNode = children.find(c => c.id.includes('_qa'))
    const qbNode = children.find(c => c.id.includes('_qb'))
    const kvaNode = children.find(c => c.id.includes('_kva'))
    const kvbNode = children.find(c => c.id.includes('_kvb'))

    let curY = topY + 16
    let oLn: LayoutedNode | undefined
    if (oNode) {
      oLn = addNode(oNode, cx, curY, 630, 46)
      curY += 46 + 22
    }
    let scoresLn: LayoutedNode | undefined
    if (scoresNode) {
      scoresLn = addNode(scoresNode, cx, curY, 630, 42)
      curY += 42 + 22
    }
    let qbLn: LayoutedNode | undefined, kvbLn: LayoutedNode | undefined
    if (qbNode && kvbNode) {
      qbLn = addNode(qbNode, cx - 160, curY, 305, 46)
      kvbLn = addNode(kvbNode, cx + 160, curY, 305, 46)
      curY += 46 + 22
    }
    let qaLn: LayoutedNode | undefined, kvaLn: LayoutedNode | undefined
    if (qaNode && kvaNode) {
      qaLn = addNode(qaNode, cx - 160, curY, 305, 46)
      kvaLn = addNode(kvaNode, cx + 160, curY, 305, 46)
      curY += 46 + 18
    }
    const height = curY - topY + 16

    if (qaLn && qbLn) edges.push({ fromId: qaLn.id, toId: qbLn.id, fromX: qaLn.cx, fromY: qaLn.topY, toX: qbLn.cx, toY: qbLn.bottomY, label: 'Latent Q' })
    if (kvaLn && kvbLn) edges.push({ fromId: kvaLn.id, toId: kvbLn.id, fromX: kvaLn.cx, fromY: kvaLn.topY, toX: kvbLn.cx, toY: kvbLn.bottomY, label: 'Latent KV' })
    if (qbLn && scoresLn) edges.push({ fromId: qbLn.id, toId: scoresLn.id, fromX: qbLn.cx, fromY: qbLn.topY, toX: scoresLn.cx - 100, toY: scoresLn.bottomY })
    if (kvbLn && scoresLn) edges.push({ fromId: kvbLn.id, toId: scoresLn.id, fromX: kvbLn.cx, fromY: kvbLn.topY, toX: scoresLn.cx + 100, toY: scoresLn.bottomY })
    if (scoresLn && oLn) edges.push({ fromId: scoresLn.id, toId: oLn.id, fromX: scoresLn.cx, fromY: scoresLn.topY, toX: oLn.cx, toY: oLn.bottomY })

    return { height, width, children: layoutedChildren, edges }
  }

  if (!isDeep) {
    let curY = topY + 16
    let oLn: LayoutedNode | undefined
    if (oNode) {
      oLn = addNode(oNode, cx, curY, 630, 48)
      curY += 48 + 32
    }
    let ropeLn: LayoutedNode | undefined
    if (ropeNode) {
      ropeLn = addNode(ropeNode, cx - 105, curY, 415, 42)
      curY += 42 + 32
    } else if (qkNormNode) {
      ropeLn = addNode(qkNormNode, cx - 105, curY, 415, 42)
      curY += 42 + 32
    }

    const qLn = qNode ? addNode(qNode, cx - 212, curY, 204, 48) : undefined
    const kLn = kNode ? addNode(kNode, cx, curY, 204, 48) : undefined
    const vLn = vNode ? addNode(vNode, cx + 212, curY, 204, 48) : undefined

    const height = (qLn?.bottomY || curY + 48) - topY + 18

    if (qLn && ropeLn) edges.push({ fromId: qLn.id, toId: ropeLn.id, fromX: qLn.cx, fromY: qLn.topY, toX: ropeLn.cx - 80, toY: ropeLn.bottomY, label: 'Q: [B, H, S, d]' })
    if (kLn && ropeLn) edges.push({ fromId: kLn.id, toId: ropeLn.id, fromX: kLn.cx, fromY: kLn.topY, toX: ropeLn.cx + 80, toY: ropeLn.bottomY, label: 'K: [B, H_kv, S, d]' })
    if (ropeLn && oLn) edges.push({ fromId: ropeLn.id, toId: oLn.id, fromX: ropeLn.cx, fromY: ropeLn.topY, toX: oLn.cx - 80, toY: oLn.bottomY, label: 'Attn Matrix' })
    if (vLn && oLn) edges.push({ fromId: vLn.id, toId: oLn.id, fromX: vLn.cx, fromY: vLn.topY, toX: oLn.cx + 120, toY: oLn.bottomY, label: 'V: [B, H_kv, S, d]' })

    return { height, width, children: layoutedChildren, edges }
  }

  // Preset 3: Deep Computational Subgraph (Full mathematical breakdown)
  let curY = topY + 16
  let oLn: LayoutedNode | undefined
  if (oNode) {
    oLn = addNode(oNode, cx, curY, 630, 46)
    curY += 46 + 22
  }
  let ctxLn: LayoutedNode | undefined
  if (ctxNode) {
    ctxLn = addNode(ctxNode, cx, curY, 540, 40)
    curY += 40 + 20
  }
  let softmaxLn: LayoutedNode | undefined
  if (softmaxNode) {
    softmaxLn = addNode(softmaxNode, cx - 105, curY, 415, 40)
    curY += 40 + 20
  }
  let scoresLn: LayoutedNode | undefined
  if (scoresNode) {
    scoresLn = addNode(scoresNode, cx - 105, curY, 415, 40)
    curY += 40 + 20
  }
  let ropeLn: LayoutedNode | undefined
  if (ropeNode) {
    ropeLn = addNode(ropeNode, cx - 105, curY, 415, 40)
    curY += 40 + 24
  } else if (qkNormNode) {
    ropeLn = addNode(qkNormNode, cx - 105, curY, 415, 40)
    curY += 40 + 24
  }

  const qLn = qNode ? addNode(qNode, cx - 212, curY, 204, 48) : undefined
  const kLn = kNode ? addNode(kNode, cx, curY, 204, 48) : undefined
  const vLn = vNode ? addNode(vNode, cx + 212, curY, 204, 48) : undefined

  const height = (qLn?.bottomY || curY + 48) - topY + 18

  if (qLn && ropeLn) edges.push({ fromId: qLn.id, toId: ropeLn.id, fromX: qLn.cx, fromY: qLn.topY, toX: ropeLn.cx - 80, toY: ropeLn.bottomY, label: 'Q [B, H, S, d]' })
  if (kLn && ropeLn) edges.push({ fromId: kLn.id, toId: ropeLn.id, fromX: kLn.cx, fromY: kLn.topY, toX: ropeLn.cx + 80, toY: ropeLn.bottomY, label: 'K [B, H_kv, S, d]' })
  if (ropeLn && scoresLn) edges.push({ fromId: ropeLn.id, toId: scoresLn.id, fromX: ropeLn.cx, fromY: ropeLn.topY, toX: scoresLn.cx, toY: scoresLn.bottomY, label: 'Rotary Q, K' })
  if (scoresLn && softmaxLn) edges.push({ fromId: scoresLn.id, toId: softmaxLn.id, fromX: scoresLn.cx, fromY: scoresLn.topY, toX: scoresLn.cx, toY: scoresLn.bottomY, label: 'Scores [H, S, S]' })
  if (softmaxLn && ctxLn) edges.push({ fromId: softmaxLn.id, toId: ctxLn.id, fromX: softmaxLn.cx, fromY: softmaxLn.topY, toX: ctxLn.cx - 100, toY: ctxLn.bottomY, label: 'Weights [H, S, S]' })
  if (vLn && ctxLn) edges.push({ fromId: vLn.id, toId: ctxLn.id, fromX: vLn.cx, fromY: vLn.topY, toX: ctxLn.cx + 120, toY: ctxLn.bottomY, label: 'Values [B, H_kv, S, d]' })
  if (ctxLn && oLn) edges.push({ fromId: ctxLn.id, toId: oLn.id, fromX: ctxLn.cx, fromY: ctxLn.topY, toX: oLn.cx, toY: oLn.bottomY, label: 'Context [B, S, D]' })

  return { height, width, children: layoutedChildren, edges }
}

/**
 * Layout internal computational graph for FFN (SwiGLU / MoE)
 */
function layoutFfnInternals(
  ffnNode: SemanticNode,
  spec: ArchSpec,
  cx: number,
  topY: number,
  preset: DetailPreset,
  expandedIds: Set<string>,
  allNodes: Map<string, LayoutedNode>,
  width: number = 700
): { height: number; width: number; children: LayoutedNode[]; edges: DataflowEdge[] } {
  const children = ffnNode.children || []
  const layoutedChildren: LayoutedNode[] = []
  const edges: DataflowEdge[] = []
  const isDeep = preset === 3 || children.some(c => expandedIds.has(c.id))

  function addNode(node: SemanticNode, nCx: number, nY: number, nW: number, nH: number): LayoutedNode {
    const { children: _c, ...props } = node
    const ln: LayoutedNode = {
      ...props,
      x: nCx - nW / 2,
      y: nY,
      width: nW,
      height: nH,
      isExpanded: false,
      cx: nCx,
      topY: nY,
      bottomY: nY + nH,
      leftX: nCx - nW / 2,
      rightX: nCx + nW / 2,
    }
    layoutedChildren.push(ln)
    allNodes.set(ln.id, ln)
    return ln
  }

  if (spec.moe) {
    const routerNode = children.find(c => c.kind === 'moe_router')
    const expNode = children.find(c => c.kind === 'moe_expert_group')
    const sharedNode = children.find(c => c.kind === 'moe_shared')

    let curY = topY + 16
    let expLn: LayoutedNode | undefined
    if (expNode) {
      expLn = addNode(expNode, cx, curY, 630, 50)
      curY += 50 + 24
    }
    let sharedLn: LayoutedNode | undefined
    if (sharedNode) {
      sharedLn = addNode(sharedNode, cx, curY, 630, 44)
      curY += 44 + 24
    }
    let routerLn: LayoutedNode | undefined
    if (routerNode) {
      routerLn = addNode(routerNode, cx, curY, 630, 44)
      curY += 44 + 18
    }
    const height = curY - topY + 16

    if (routerLn && expLn) {
      edges.push({
        fromId: routerLn.id,
        toId: expLn.id,
        fromX: routerLn.cx,
        fromY: routerLn.topY,
        toX: expLn.cx,
        toY: expLn.bottomY,
        label: 'Routing Probabilities',
      })
    }

    return { height, width, children: layoutedChildren, edges }
  }

  // Dense SwiGLU FFN
  const gateNode = children.find(c => c.kind === 'gate_proj')
  const upNode = children.find(c => c.kind === 'up_proj')
  const downNode = children.find(c => c.kind === 'down_proj')
  const actNode = children.find(c => c.kind === 'act_op')
  const hadamardNode = children.find(c => c.kind === 'hadamard_op')

  if (!isDeep) {
    let curY = topY + 16
    let downLn: LayoutedNode | undefined
    if (downNode) {
      downLn = addNode(downNode, cx, curY, 630, 48)
      curY += 48 + 32
    }
    const gateLn = gateNode ? addNode(gateNode, cx - 160, curY, 305, 48) : undefined
    const upLn = upNode ? addNode(upNode, cx + 160, curY, 305, 48) : undefined

    const height = (gateLn?.bottomY || curY + 48) - topY + 18

    if (gateLn && downLn) {
      edges.push({
        fromId: gateLn.id,
        toId: downLn.id,
        fromX: gateLn.cx,
        fromY: gateLn.topY,
        toX: downLn.cx - 100,
        toY: downLn.bottomY,
        label: 'Gate [B, S, I]',
      })
    }
    if (upLn && downLn) {
      edges.push({
        fromId: upLn.id,
        toId: downLn.id,
        fromX: upLn.cx,
        fromY: upLn.topY,
        toX: downLn.cx + 100,
        toY: downLn.bottomY,
        label: 'Up [B, S, I]',
      })
    }

    return { height, width, children: layoutedChildren, edges }
  }

  // Deep SwiGLU FFN
  let curY = topY + 16
  let downLn: LayoutedNode | undefined
  if (downNode) {
    downLn = addNode(downNode, cx, curY, 630, 46)
    curY += 46 + 24
  }
  let hadamardLn: LayoutedNode | undefined
  if (hadamardNode) {
    hadamardLn = addNode(hadamardNode, cx, curY, 420, 40)
    curY += 40 + 22
  }
  const actLn = actNode ? addNode(actNode, cx - 160, curY, 305, 40) : undefined
  const upLn = upNode ? addNode(upNode, cx + 160, curY, 305, 48) : undefined

  curY += 40 + 22
  const gateLn = gateNode ? addNode(gateNode, cx - 160, curY, 305, 48) : undefined

  const height = (gateLn?.bottomY || curY + 48) - topY + 18

  if (gateLn && actLn) {
    edges.push({
      fromId: gateLn.id,
      toId: actLn.id,
      fromX: gateLn.cx,
      fromY: gateLn.topY,
      toX: actLn.cx,
      toY: actLn.bottomY,
      label: 'Gate [B, S, I]',
    })
  }
  if (actLn && hadamardLn) {
    edges.push({
      fromId: actLn.id,
      toId: hadamardLn.id,
      fromX: actLn.cx,
      fromY: actLn.topY,
      toX: hadamardLn.cx - 80,
      toY: hadamardLn.bottomY,
      label: 'SiLU(G)',
    })
  }
  if (upLn && hadamardLn) {
    edges.push({
      fromId: upLn.id,
      toId: hadamardLn.id,
      fromX: upLn.cx,
      fromY: upLn.topY,
      toX: hadamardLn.cx + 80,
      toY: hadamardLn.bottomY,
      label: 'Up [B, S, I]',
    })
  }
  if (hadamardLn && downLn) {
    edges.push({
      fromId: hadamardLn.id,
      toId: downLn.id,
      fromX: hadamardLn.cx,
      fromY: hadamardLn.topY,
      toX: downLn.cx,
      toY: downLn.bottomY,
      label: 'SwiGLU [B, S, I]',
    })
  }

  return { height, width, children: layoutedChildren, edges }
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

  const internalEdges: DataflowEdge[] = []
  const magnifiedInsets: MagnifiedInset[] = []
  const insetX = 840
  const insetW = 700
  const insetCx = insetX + insetW / 2 // 1190

  // Repeated block dimensions (stable tower)
  const blkW = BLK_W
  const outerW = OUTER_W
  const blkX = cx - blkW / 2
  const outerX = cx - outerW / 2
  const railX = blkX + blkW - 45

  // FFN / MoE Block in Central Tower (always compact in tower)
  const ffnTop = y
  const ffnH = 48
  const ffnW = 240

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
    children: [],
    cx,
    topY: ffnTop,
    bottomY: ffnTop + ffnH,
    leftX: cx - ffnW / 2,
    rightX: cx + ffnW / 2,
  }
  allNodes.set(layoutedFfn.id, layoutedFfn)
  y = ffnTop + ffnH + 34

  // If FFN is expanded: layout into spacious Magnified Side Inset
  if (isFfnExpanded && ffnNode?.children) {
    const ffnInsetTopY = 240
    const ffnRes = layoutFfnInternals(ffnNode, spec, insetCx, ffnInsetTopY + 36, preset, expandedIds, allNodes, insetW)
    const insetH = ffnRes.height + 44
    magnifiedInsets.push({
      id: 'ffn_magnified_inset',
      kind: spec.moe ? 'moe' : 'ffn',
      title: spec.moe
        ? `▲ Mixture-of-Experts (${preset === 3 ? 'Deep Operations' : 'Expanded'})`
        : `▲ SwiGLU Feed-Forward Network (${preset === 3 ? 'Deep Operations' : 'Expanded'})`,
      sourceNodeId: layoutedFfn.id,
      sourceX: blkX + blkW,
      sourceTopY: ffnTop,
      sourceBottomY: ffnTop + ffnH,
      x: insetX,
      y: ffnInsetTopY,
      width: insetW,
      height: insetH,
      children: ffnRes.children,
      edges: ffnRes.edges,
    })
    internalEdges.push(...ffnRes.edges)
  }

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

  // Attention Block in Central Tower (always compact in tower)
  const attnTop = y
  const attnH = 96
  const attnW = 310

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
    children: [],
    cx,
    topY: attnTop,
    bottomY: attnTop + attnH,
    leftX: cx - attnW / 2,
    rightX: cx + attnW / 2,
  }
  allNodes.set(layoutedAttn.id, layoutedAttn)
  y = attnTop + attnH + 34

  // If Attention is expanded: layout into spacious Magnified Side Inset
  if (isAttnExpanded && attnNode?.children) {
    const ffnInset = magnifiedInsets.find(i => i.kind === 'ffn' || i.kind === 'moe')
    const attnInsetTopY = ffnInset ? ffnInset.y + ffnInset.height + 36 : 560
    const attnRes = layoutAttentionInternals(attnNode, spec, insetCx, attnInsetTopY + 36, preset, expandedIds, allNodes, insetW)
    const insetH = attnRes.height + 44
    magnifiedInsets.push({
      id: 'attn_magnified_inset',
      kind: 'attention',
      title: `▲ Self-Attention Mechanism (${preset === 3 ? 'Deep Operations' : 'Expanded'})`,
      sourceNodeId: layoutedAttn.id,
      sourceX: blkX + blkW,
      sourceTopY: attnTop,
      sourceBottomY: attnTop + attnH,
      x: insetX,
      y: attnInsetTopY,
      width: insetW,
      height: insetH,
      children: attnRes.children,
      edges: attnRes.edges,
    })
    internalEdges.push(...attnRes.edges)
  }

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

  const hasMagnified = magnifiedInsets.length > 0
  const isMoeOverview = spec.moe && !isFfnExpanded
  const totalWidth = hasMagnified ? 1600 : (isMoeOverview ? 1540 : 1100)

  let maxInsetBottom = 0
  for (const mi of magnifiedInsets) {
    maxInsetBottom = Math.max(maxInsetBottom, mi.y + mi.height)
  }

  const towerBottom = tokY + tokH + 120
  const totalHeight = hasMagnified
    ? Math.max(towerBottom + 160, maxInsetBottom + 80)
    : towerBottom

  // Residual bypass rails
  const residualRails = [
    {
      startCx: cx,
      startY: norm1Bot + 20,
      railX: railX,
      endY: plus1Y,
      endCx: cx + 17,
    },
    {
      startCx: cx,
      startY: layoutedNorm2.bottomY + 18,
      railX: railX,
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

  if (hasMagnified) {
    // Left-anchored callout so it never collides with side magnified insets
    callouts.push({
      x: 18,
      y: 460,
      lines: headLines,
      targetX: blkX,
      targetY: layoutedAttn.y + 20,
    })
  } else {
    callouts.push({
      x: cx + 330,
      y: embTop + 44,
      lines: headLines,
      targetX: cx + 152,
      targetY: layoutedAttn.bottomY - 8,
    })
  }

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
    internalEdges,
    magnifiedInsets,
    callouts,
    outerContainer: {
      x: outerX,
      y: outerTop,
      w: outerW,
      h: outerH,
    },
    repeatedBlockContainer: {
      x: blkX,
      y: blkTop,
      w: blkW,
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
  const isAttn = focusNode.kind === 'attention'
  const isFfn = focusNode.kind === 'ffn' || focusNode.kind === 'moe'

  if (isAttn) {
    const res = layoutAttentionInternals(focusNode, spec, cx, 140, 3, state.expandedIds, allNodes, 700)
    const focusW = 700
    const focusH = res.height
    const { children: _fCh, ...focusProps } = focusNode
    const rootLayouted: LayoutedNode = {
      ...focusProps,
      x: cx - focusW / 2,
      y: 110,
      width: focusW,
      height: focusH + 30,
      isExpanded: true,
      children: res.children,
      cx,
      topY: 110,
      bottomY: 110 + focusH + 30,
      leftX: cx - focusW / 2,
      rightX: cx + focusW / 2,
    }
    allNodes.set(rootLayouted.id, rootLayouted)
    const totalWidth = 900
    const totalHeight = Math.max(650, 110 + focusH + 80)
    return {
      width: totalWidth,
      height: totalHeight,
      viewBox: `0 0 ${totalWidth} ${totalHeight}`,
      rootNodes: [rootLayouted],
      allNodes,
      residualRails: [],
      internalEdges: res.edges,
      callouts: [],
    }
  }

  if (isFfn) {
    const res = layoutFfnInternals(focusNode, spec, cx, 140, 3, state.expandedIds, allNodes, 700)
    const focusW = 700
    const focusH = res.height
    const { children: _fCh, ...focusProps } = focusNode
    const rootLayouted: LayoutedNode = {
      ...focusProps,
      x: cx - focusW / 2,
      y: 110,
      width: focusW,
      height: focusH + 30,
      isExpanded: true,
      children: res.children,
      cx,
      topY: 110,
      bottomY: 110 + focusH + 30,
      leftX: cx - focusW / 2,
      rightX: cx + focusW / 2,
    }
    allNodes.set(rootLayouted.id, rootLayouted)
    const totalWidth = 900
    const totalHeight = Math.max(650, 110 + focusH + 80)
    return {
      width: totalWidth,
      height: totalHeight,
      viewBox: `0 0 ${totalWidth} ${totalHeight}`,
      rootNodes: [rootLayouted],
      allNodes,
      residualRails: [],
      internalEdges: res.edges,
      callouts: [],
    }
  }

  // Fallback vertical stack for other nodes
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
    internalEdges: [],
    callouts: [],
  }
}
