#!/usr/bin/env python3
"""麻将标注器：浏览器打开，直接修正 YOLO 预标注。

用法：
  training/venv/bin/python training/tools/annotator.py --dataset <数据集目录> [--port 8899]

操作：
  点击框选中 / Del 删除 / 拖动移动 / 拖四角收紧 / 空白处拖拽画新框
  ←→ 切换图片（自动保存） / Ctrl+S 保存 / 滚轮缩放
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from PIL import Image

NAMES = ([f"wan{i}" for i in range(1, 10)] + [f"tong{i}" for i in range(1, 10)]
         + [f"tiao{i}" for i in range(1, 10)] + ["back"])

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>麻将标注器</title>
<style>
 body{margin:0;font:14px system-ui;background:#1e1e1e;color:#eee;display:flex;flex-direction:column;height:100vh}
 #bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#2b2b2b;flex-wrap:wrap}
 #bar b{color:#8fd}
 canvas{cursor:crosshair;flex:1;display:block;background:#111}
 .cls{padding:3px 8px;border-radius:4px;border:1px solid #555;cursor:pointer;font-size:12px}
 .cls.sel{outline:2px solid #fff;font-weight:600}
 button{padding:5px 12px;border-radius:4px;border:0;background:#3d8fd1;color:#fff;cursor:pointer}
 #classes{display:flex;gap:4px;flex-wrap:wrap;padding:4px 12px;background:#262626}
</style></head><body>
<div id="bar">
  <button onclick="nav(-1)">← 上一张</button>
  <b id="idx"></b>
  <button onclick="nav(1)">下一张 →</button>
  <span id="dirty" style="color:#f66"></span>
  <span style="flex:1"></span>
  <button onclick="save(true)" style="background:#2e9e5b">保存 (Ctrl+S)</button>
  <span id="cnt"></span>
</div>
<div id="classes"></div>
<canvas id="cv"></canvas>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let names = [], photos = [], cur = -1, img = null;
let boxes = [], sel = -1, scale = 1, offX = 0, offY = 0;
let drag = null, curCls = 27, dirty = false;

fetch('/api/meta').then(r=>r.json()).then(m=>{
  names = m.names; photos = m.photos;
  const cdiv = document.getElementById('classes');
  names.forEach((n,i)=>{
    const s = document.createElement('span');
    s.className='cls'; s.textContent=n; s.onclick=()=>{curCls=i;
      document.querySelectorAll('.cls').forEach((e,j)=>e.classList.toggle('sel', j===i));};
    if(i===27) s.classList.add('sel');
    cdiv.appendChild(s);
  });
  load(0);
});
cv.addEventListener('wheel', e=>{
  e.preventDefault();
  const f = e.deltaY<0?1.15:0.87;
  const mx=e.offsetX,my=e.offsetY;
  offX = mx-(mx-offX)*f; offY = my-(my-offY)*f; scale*=f; draw();
});
function toImg(mx,my){ return {x:(mx-offX)/scale, y:(my-offY)/scale}; }
cv.addEventListener('mousedown', e=>{
  const m = toImg(e.offsetX,e.offsetY);
  let picked = -1;
  for(let i=boxes.length-1;i>=0;i--){
    const b=boxes[i];
    if(m.x>b.x&&m.x<b.x+b.w&&m.y>b.y&&m.y<b.y+b.h){picked=i;break;}
  }
  if(picked>=0){
    sel=picked; const b=boxes[picked];
    const t=9/scale;
    let c='';
    if(Math.abs(m.x-b.x)<t)c+='l'; if(Math.abs(m.x-b.x-b.w)<t)c+='r';
    if(Math.abs(m.y-b.y)<t)c+='t'; if(Math.abs(m.y-b.y-b.h)<t)c+='b';
    drag = c ? {type:'resize',i:picked,c,anchor:{...b}}
             : {type:'move',i:picked,sx:m.x,sy:m.y,ox:b.x,oy:b.y};
  } else {
    drag={type:'new',sx:m.x,sy:m.y,ex:m.x,ey:m.y}; sel=-1;
  }
  dirty=true; upd(); draw();
});
cv.addEventListener('mousemove', e=>{
  if(!drag) return;
  const m = toImg(e.offsetX,e.offsetY);
  if(drag.type==='new'){drag.ex=m.x;drag.ey=m.y;}
  else if(drag.type==='move'){const b=boxes[drag.i];
    b.x=drag.ox+(m.x-drag.sx); b.y=drag.oy+(m.y-drag.sy);}
  else if(drag.type==='resize'){const b=boxes[drag.i],a=drag.anchor;
    // 以按下时的原始框为基准，按角/边重算
    let x1=a.x,y1=a.y,x2=a.x+a.w,y2=a.y+a.h;
    if(drag.c.includes('l'))x1=m.x; if(drag.c.includes('r'))x2=m.x;
    if(drag.c.includes('t'))y1=m.y; if(drag.c.includes('b'))y2=m.y;
    b.x=Math.min(x1,x2);b.y=Math.min(y1,y2);b.w=Math.abs(x2-x1);b.h=Math.abs(y2-y1);}
  draw();
});
window.addEventListener('mouseup', ()=>{
  if(drag&&drag.type==='new'){
    const x=Math.min(drag.sx,drag.ex),y=Math.min(drag.sy,drag.ey);
    const w=Math.abs(drag.ex-drag.sx),h=Math.abs(drag.ey-drag.sy);
    if(w>8&&h>8){ boxes.push({x,y,w,h,c:curCls}); sel=boxes.length-1; }
  }
  drag=null; upd(); draw();
});
window.addEventListener('keydown', e=>{
  if(e.key==='Delete'||e.key==='Backspace'){if(sel>=0){boxes.splice(sel,1);sel=-1;dirty=true;draw();}}
  else if(e.key==='ArrowRight') nav(1);
  else if(e.key==='ArrowLeft') nav(-1);
  else if((e.metaKey||e.ctrlKey)&&e.key==='s'){e.preventDefault();save();}
});
function draw(){
  cv.width=cv.clientWidth; cv.height=cv.clientHeight;
  ctx.fillStyle='#111'; ctx.fillRect(0,0,cv.width,cv.height);
  if(!img) return;
  ctx.save(); ctx.translate(offX,offY); ctx.scale(scale,scale);
  ctx.drawImage(img,0,0);
  ctx.lineWidth=2/scale;
  boxes.forEach((b,i)=>{
    ctx.strokeStyle = b.c===27?'#f0c040':'#40d070';
    if(i===sel){ctx.strokeStyle='#ff5050';ctx.lineWidth=3/scale;}
    ctx.strokeRect(b.x,b.y,b.w,b.h);
    ctx.font=(12/scale)+'px system-ui';
    ctx.fillStyle=ctx.strokeStyle;
    ctx.fillText(names[b.c], b.x, b.y-4/scale);
  });
  if(drag&&drag.type==='new'){
    ctx.strokeStyle='#50a0ff'; ctx.lineWidth=1.5/scale;
    ctx.strokeRect(Math.min(drag.sx,drag.ex),Math.min(drag.sy,drag.ey),
                   Math.abs(drag.ex-drag.sx),Math.abs(drag.ey-drag.sy));
  }
  ctx.restore();
}

function load(i){
  if(i<0||i>=photos.length) return;
  if(dirty&&!confirm('有未保存修改，放弃？')) return;
  cur=i; dirty=false; boxes=[]; sel=-1;
  img=new Image();
  img.onload=()=>{
    scale=Math.min(cv.clientWidth/img.width, (cv.clientHeight-4)/img.height, 1.5);
    offX=(cv.width-img.width*scale)/2; offY=(cv.height-img.height*scale)/2;
    draw();
  };
  img.src='/api/photo?n='+encodeURIComponent(photos[i]);
  fetch('/api/label?n='+encodeURIComponent(photos[i])).then(r=>r.json()).then(l=>{
    boxes=l; draw(); upd();
  });
  upd();
}
function nav(d){ save(); load(cur+d); }
function upd(){
  document.getElementById('idx').textContent=(cur+1)+' / '+photos.length;
  document.getElementById('dirty').textContent=dirty?'● 未保存':'';
  document.getElementById('cnt').textContent=boxes.length+' 框';
}
async function save(keep){
  if(cur<0) return;
  const l=boxes.map(b=>({x:b.x,y:b.y,w:b.w,h:b.h,c:b.c}));
  await fetch('/api/label?n='+encodeURIComponent(photos[cur]),{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({W:img.width,H:img.height,boxes:l})});
  dirty=false; upd();
  if(keep===true) load(cur+1);
}
window.addEventListener('resize',()=>{if(img){scale=Math.min(cv.clientWidth/img.width,(cv.clientHeight-4)/img.height,1.5);offX=(cv.width-img.width*scale)/2;offY=(cv.height-img.height*scale)/2;draw();}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    ds = Path(args.dataset).resolve()
    app = Flask(__name__)

    @app.get("/api/meta")
    def meta():
        photos = sorted(p.stem for p in (ds / "images/train").glob("*.jpg"))
        return jsonify({"names": NAMES, "photos": photos})

    @app.get("/api/photo")
    def photo():
        name = urllib.parse.unquote(request.args["n"])
        return send_file(ds / f"images/train/{name}.jpg")

    @app.get("/api/label")
    def get_label():
        name = urllib.parse.unquote(request.args["n"])
        lbl = ds / f"labels/train/{name}.txt"
        W, H = Image.open(ds / f"images/train/{name}.jpg").size
        boxes = []
        if lbl.exists() and lbl.read_text().strip():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                boxes.append({"x": (cx - bw / 2) * W, "y": (cy - bh / 2) * H,
                              "w": bw * W, "h": bh * H, "c": cid})
        return jsonify(boxes)

    @app.post("/api/label")
    def post_label():
        name = urllib.parse.unquote(request.args["n"])
        body = request.get_json()
        W, H = body["W"], body["H"]
        lines = []
        for b in body["boxes"]:
            if b["w"] < 4 or b["h"] < 4:
                continue
            cx = (b["x"] + b["w"] / 2) / W
            cy = (b["y"] + b["h"] / 2) / H
            lines.append(f"{b['c']} {cx:.6f} {cy:.6f} {b['w']/W:.6f} {b['h']/H:.6f}")
        (ds / f"labels/train/{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return jsonify({"saved": len(lines)})

    @app.get("/")
    def index():
        return HTML

    print(f"标注器: http://localhost:{args.port}  （数据集 {ds.name}）")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
