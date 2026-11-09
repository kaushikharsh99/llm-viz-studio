import { useEffect, useState } from 'react'
import DiagramView from './visualization/diagram/DiagramView'
import GraphView from './visualization/graph/GraphView'
import { fetchJSON } from './shared/api/api'

const EXAMPLES = [
 {label:'Qwen dense', id:'Qwen/Qwen2.5-1.5B-Instruct'},
 {label:'Llama-family dense', id:'HuggingFaceTB/SmolLM2-1.7B'},
 {label:'DeepSeek MoE', id:'deepseek-ai/DeepSeek-V3'},
 {label:'Qwen MoE', id:'Qwen/Qwen3-30B-A3B'},
]
function useDark(){const [d,setD]=useState(()=>localStorage.getItem('theme')==='dark');useEffect(()=>{localStorage.setItem('theme',d?'dark':'light')},[d]);return [d,setD] as const}
export default function App(){
 const params=new URLSearchParams(location.search)
 const model=params.get('model')
 const [dark,setDark]=useDark()
 return <div className={`app-shell${dark?' dark':''}`}>
  <header className="nav"><a href="/" className="brand"><span className="brand-mark" aria-hidden="true">◈</span> LLMViz Studio</a><nav className="nav-links"><a href="/">Explore</a><a href="https://github.com/kaushikharsh99/llm-viz-studio" target="_blank" rel="noreferrer">GitHub</a><button className="theme-toggle" aria-label="Theme toggle" onClick={()=>setDark(!dark)}>{dark?'☀':'☾'}</button></nav></header>
  {model? <Viewer model={model}/> : <Home/>}
  <footer>LLMViz Studio — Explore LLM architectures visually. Open source under Apache-2.0.</footer>
 </div>
}
function Home(){
 const [q,setQ]=useState(''); const [sug,setSug]=useState<any[]>([]); const [idx,setIdx]=useState(-1); const [err,setErr]=useState(''); const [loading,setLoading]=useState(false)
 const go=(id:string)=>{location.href=`/visualize?model=${encodeURIComponent(id)}`}
 useEffect(()=>{ if(!q||q.includes('/')){setSug([]);return} const c=new AbortController(); setLoading(true); const t=setTimeout(()=>{fetchJSON(`/api/v1/models/search?q=${encodeURIComponent(q)}`,c.signal).then(setSug).catch(()=>{}).finally(()=>setLoading(false))},300); return ()=>{clearTimeout(t);c.abort()}},[q])
 const submit=(e:any)=>{e.preventDefault(); const id=(idx>=0&&sug[idx])?sug[idx].id:q.trim(); if(!id)return; if(!/^[^/]+\/[^/]+$/.test(id)){setErr('Enter a full model ID like Qwen/Qwen2.5-1.5B-Instruct, or pick a suggestion.');return} go(id)}
 return <main className="home-page">
  <section className="hero"><p className="eyebrow">MODEL ARCHITECTURE · OPEN SOURCE</p><h1>Visualize any LLM architecture.</h1><p className="hero-copy">Turn Hugging Face model configurations into detailed, publication-ready 2D diagrams. No downloads, GPUs, or model weights required.</p>
  <form className="search" onSubmit={submit} role="search">
   <input aria-label="Model search" placeholder="Search models or enter a Hugging Face model ID..." value={q} onChange={e=>{setQ(e.target.value);setIdx(-1);setErr('')}} onKeyDown={e=>{if(e.key==='ArrowDown'){e.preventDefault();setIdx(Math.min(idx+1,sug.length-1))} if(e.key==='ArrowUp'){e.preventDefault();setIdx(Math.max(idx-1,-1))} if(e.key==='Escape'){setSug([])}}} />
   <button type="submit">{loading?'Searching…':'Visualize'}</button>
   {sug.length>0 && <ul className="sug" role="listbox">{sug.map((s,i)=><li key={s.id} role="option" aria-selected={i===idx} className={i===idx?'sel':''} onMouseDown={()=>go(s.id)}><b>{s.id}</b><small>{s.author||''} {s.pipeline_tag||''}</small></li>)}</ul>}
  </form>
  {err&&<p className="err" role="alert">{err}</p>}
  <div className="chips">{EXAMPLES.map(x=><button key={x.id} onClick={()=>go(x.id)}>{x.label}: <code>{x.id}</code></button>)}</div>
  </section>
  <section className="cards">
   <article><h3>Real configs, real diagrams</h3><p>We parse the model's public <code>config.json</code> with our built-in engine — embeddings, attention variant, MoE routing, norms and parameter estimates.</p></article>
   <article><h3>Zoom, pan, export</h3><p>Inspect the SVG at any scale, then download full-resolution SVG or PNG for papers and slides.</p></article>
   <article><h3>Open &amp; self-hostable</h3><p>Apache-2.0, no weights, no API keys, no GPU. Run locally with Docker Compose.</p></article>
  </section>
 </main>
}
// Viewer: tab shell only. Each tab is an independent module with its own state;
// switching tabs keeps the selected model (from URL) and never resets the other view's logic.
function Viewer({model}:{model:string}){
 const [tab,setTab]=useState<'diagram'|'graph'>('diagram')
 return <main className="viewer">
  <div className="toolbar" role="tablist" aria-label="Visualization mode">
   <button role="tab" aria-selected={tab==='diagram'} onClick={()=>setTab('diagram')}>Diagram</button>
   <button role="tab" aria-selected={tab==='graph'} onClick={()=>setTab('graph')}>Graph</button>
  </div>
  {tab==='diagram' ? <DiagramView key={model} model={model}/> : <GraphView key={model} model={model}/>}
 </main>
}
