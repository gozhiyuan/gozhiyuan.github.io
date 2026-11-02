"""Build original, editable SVG figures for the illustrative Part 4 trace."""
from pathlib import Path
from html import escape

OUT = Path(__file__).parent
INK = '#18283d'
BLUE = '#e8f1ff'
GREEN = '#e2f5ec'
AMBER = '#fff2d9'
GRAY = '#edf0f5'

def text(x, y, s, size=19, weight='400', fill=INK, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'

def box(x,y,w,h,label,sub='',fill=BLUE):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#b6c4d4"/>'
            +text(x+16,y+30,label,20,'600')+(text(x+16,y+58,sub,18) if sub else ''))

def arrow(x1,y1,x2,y2,color='#52647b'):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'

def save(name,w,h,body,title):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#52647b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="system-ui, -apple-system, sans-serif">{body}</g></svg>'''
    (OUT/name).write_text(svg)

b=text(28,38,'B’s cached prefix and two new query rows',25,'700')
b+=text(28,69,'Illustrative addresses · allocator page size 2 · one TP rank',18)
labels=['logical pos','token ID','physical slot']
vals=[list(range(6)),[10,11,20,21,40,41],[0,1,2,3,6,7]]
for r,label in enumerate(labels):
    y=105+r*47
    b+=text(28,y+27,label,18,'600')
    for c,v in enumerate(vals[r]):
        x=200+c*103
        b+=f'<rect x="{x}" y="{y}" width="91" height="38" rx="5" fill="{GREEN if c<4 else BLUE}"/>'+text(x+45,y+26,str(v),20,'500',anchor='middle')
b+=text(200,268,'Prefix K/V already valid',18,'600')+text(612,268,'Write new K/V',18,'600')
b+=text(28,316,'Allowed attention reads',21,'700')
for r,(pos,n) in enumerate([(4,5),(5,6)]):
    y=340+r*62
    b+=text(28,y+27,f'query pos {pos}',19,'600')
    for c in range(6):
        x=200+c*103
        b+=f'<rect x="{x}" y="{y}" width="91" height="43" rx="5" fill="{GREEN if c<n else GRAY}"/>'
        b+=text(x+45,y+28,f'slot {vals[2][c]}' if c<n else 'masked',17,anchor='middle')
b+=text(28,495,'Only [40, 41] enter the model. Both queries can read the cached prefix.',18)
save('engine-kv-trace.svg',860,520,b,'Packed suffix queries write two K/V slots and read the causally allowed prefix')

b=text(28,38,'One Qwen3 block: two rows stay two rows',25,'700')
b+=text(28,70,'Illustrative shape fixture · TP 1 · hidden 256 · 4 Q heads / 2 KV heads',18)
b+=box(180,100,480,68,'Token embeddings','[2] → [2, 256]')
b+=f'<rect x="130" y="190" width="580" height="440" rx="14" fill="#fafbfd" stroke="#bac5d3" stroke-dasharray="6 4"/>'
b+=text(150,218,'× 2 decoder layers',18,'600')
b+=arrow(420,168,420,235)
b+=box(180,235,480,65,'RMSNorm → merged QKV','[2, 256] → [2, 512]')
b+=arrow(420,300,420,325)
b+=box(180,325,480,75,'Q/K norm + RoPE → attention','Q [2, 4, 64] · K/V each [2, 2, 64]',fill=GREEN)
b+=arrow(420,400,420,425)
b+=box(180,425,480,65,'Output projection → add residual','[2, 256] + [2, 256]')
b+=arrow(420,490,420,515)
b+=box(180,515,480,85,'RMSNorm → gated MLP','[2, 256] → [2, 1024] → [2, 512] → [2, 256]')
b+=f'<path d="M180,130 H90 V455 H177" fill="none" stroke="#52647b" stroke-width="2" marker-end="url(#arrow)"/>'
b+=text(28,318,'residual',17)
b+=f'<path d="M660,455 H750 V685 H663" fill="none" stroke="#52647b" stroke-width="2" marker-end="url(#arrow)"/>'
b+=text(692,632,'residual',17)
b+=arrow(420,600,420,650)
b+=box(180,650,480,70,'Final add + RMSNorm → last row → LM head','[2, 256] → [1, 256] → logits [1, 64]',fill=AMBER)
b+=text(28,758,'MLP residual addition is fused into the following norm (or the final norm).',18)
save('engine-qwen3-shapes.svg',860,785,b,'Qwen3 block shape flow and residual path for two suffix tokens')

b=text(28,38,'CUDA graphs reuse work; token and cache inputs change',24,'700')
b+=text(28,70,'Decode only · illustrative capture buckets [1, 2, 4, 8]',18)
b+=box(28,105,375,90,'This scheduling turn','3 real requests → pad to bucket 4')
b+=box(455,105,375,90,'Persistent GPU input buffers','input_ids · positions · out_loc',fill=GREEN)
b+=arrow(403,150,455,150)
b+=box(455,225,375,90,'Backend replay preparation','Update lengths and cache addresses',fill=GREEN)
b+=arrow(642,195,642,225)
b+=box(455,345,375,90,'Replay captured model forward','4 rows → persistent logits buffer')
b+=arrow(642,315,642,345)
b+=box(28,345,375,90,'After replay: sampler','Keep 3 real rows → 3 next-token IDs',fill=AMBER)
b+=arrow(455,390,403,390)
b+=text(28,250,'The dummy row uses a reserved',18)+text(28,277,'request row and dummy cache page.',18)
b+=text(28,480,'Sampling, scheduling and CPU result handling stay outside the captured model.',18)
save('engine-graph-contract.svg',860,510,b,'Decode graph padding, persistent buffers and sampling outside graph replay')
print('Generated 3 SVG figures')
