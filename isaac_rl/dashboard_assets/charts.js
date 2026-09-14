'use strict';
const $ = id => document.getElementById(id);
const colors = ['#76dfb5','#82b4ff','#ffbd78','#d9a0ff'];
const specs = [
 ['Total PPO loss','updates',['loss']],
 ['Policy loss','updates',['policy_loss']],
 ['Value loss','updates',['value_loss']],
 ['Episode return','episodes',['r']],
 ['Policy entropy','updates',['entropy']],
 ['Approximate KL divergence','updates',['kl']],
 ['Value explained variance','updates',['explained_variance']],
 ['Gradient norm','updates',['grad_norm']],
 ['Collection throughput · steps/s','updates',['rollout_sps']],
 ['Measured phase time · seconds','updates',['timing.collection_s','timing.update_s','timing.inference_s','timing.reset_s']],
 ['PPO update time · seconds','updates',['timing.update_s']],
 ['Windows committed memory · %','updates',['commit_memory.percent']],
 ['Episode length · steps','episodes',['l']],
 ['Episode combat & exploration','episodes',['kills','combat_clears','rooms']],
 ['Episode damage','episodes',['damage_dealt','damage_taken']],
 ['Episode success & boss encounter · fraction','episodes',['success','boss_seen']]
];
const labels = {r:'Return',l:'Length',success:'Floor success',boss_seen:'Boss seen'};
let snapshot=null, paused=false;
const charts = specs.map(([title,source,keys]) => {
 const card=document.createElement('article'), heading=document.createElement('h2'), legend=document.createElement('div'), plot=document.createElement('div'), canvas=document.createElement('canvas'), tip=document.createElement('div');
 heading.textContent=title;legend.className='legend';plot.className='plot';tip.className='tip';canvas.setAttribute('role','img');canvas.setAttribute('aria-label',title);canvas.tabIndex=0;
 keys.forEach((key,i)=>{const label=document.createElement('span');label.style.color=colors[i];label.textContent=labels[key]||key.split('.').pop().replaceAll('_',' ');legend.append(label)});
 plot.append(canvas,tip);card.append(heading,legend,plot);$('charts').append(card);
 const chart={canvas,tip,source,keys,title,series:[]};
 canvas.addEventListener('pointermove',e=>hover(chart,e.offsetX));canvas.addEventListener('pointerleave',()=>tip.style.display='none');
 canvas.addEventListener('focus',()=>hover(chart,canvas.clientWidth-18));canvas.addEventListener('blur',()=>tip.style.display='none');return chart;
});
function number(n,d=2){return typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'—'}
function value(row,key){const v=key.split('.').reduce((o,k)=>o?.[k],row);return typeof v==='boolean'?Number(v):typeof v==='number'&&Number.isFinite(v)?v:null}
function short(n){return new Intl.NumberFormat(undefined,{notation:'compact',maximumFractionDigits:2}).format(n)}
function seriesFor(rows,key){
 const count=Number($('smooth').value),axis=$('axis').value;let queue=[],sum=0,prev=null,segment=0;
 return rows.map(row=>{
  if(prev&&(row.session_id!==prev.session_id||row.steps<prev.steps)){queue=[];sum=0;segment++}
  prev=row;const raw=value(row,key),x=value(row,axis);
  if(raw===null||x===null){queue=[];sum=0;segment++;return null}
  queue.push(raw);sum+=raw;if(queue.length>count)sum-=queue.shift();return {x,y:sum/queue.length,raw,segment,steps:row.steps,time:row.time};
 }).filter(Boolean);
}
// Pixel buckets retain extrema (not every-N sampling); full data remains in memory.
function envelope(points,xpos){
 const output=[];let bucket=[],last=null;
 function flush(){if(!bucket.length)return;let lo=0,hi=0;bucket.forEach((p,i)=>{if(p.y<bucket[lo].y)lo=i;if(p.y>bucket[hi].y)hi=i});[...new Set([0,lo,hi,bucket.length-1])].sort((a,b)=>a-b).forEach(i=>output.push(bucket[i]));bucket=[]}
 for(const p of points){const key=`${p.segment}:${Math.floor(xpos(p.x))}`;if(key!==last){flush();last=key}bucket.push(p)}flush();return output;
}
function draw(chart){
 const {canvas,keys,source}=chart,ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=245,dpr=window.devicePixelRatio||1;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.scale(dpr,dpr);ctx.font='11px system-ui';ctx.clearRect(0,0,w,h);
 const all=snapshot[source]||[],latest=all.reduce((m,r)=>Math.max(m,Number(r.steps)||0),snapshot.state.steps||0),range=$('range').value;
 chart.series=keys.map(key=>seriesFor(all,key).filter(p=>range==='all'||p.steps>=latest-Number(range)));
 const points=chart.series.flat();if(!points.length){ctx.fillStyle='#99a9bf';ctx.fillText('No recorded values in this range',18,110);return}
 let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(const p of points){xmin=Math.min(xmin,p.x);xmax=Math.max(xmax,p.x);ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y)}
 if(xmin===xmax){xmin-=1;xmax+=1}const padding=(ymax-ymin)*.08||Math.max(Math.abs(ymin)*.08,.1);ymin-=padding;ymax+=padding;
 const left=65,right=w-18,top=12,bottom=h-35,xpos=x=>left+(x-xmin)/(xmax-xmin)*(right-left),ypos=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
 chart.bounds={left,right,xmin,xmax};ctx.lineWidth=1;
 for(let i=0;i<5;i++){const y=top+(bottom-top)*i/4;ctx.strokeStyle='#263348';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#99a9bf';ctx.textAlign='right';ctx.fillText(number(ymax-(ymax-ymin)*i/4,3),left-9,y+4)}
 for(let i=0;i<4;i++){const x=xmin+(xmax-xmin)*i/3;ctx.textAlign=i===0?'left':i===3?'right':'center';ctx.fillText($('axis').value==='time'?new Date(x*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}):short(x),xpos(x),h-12)}
 chart.series.forEach((series,i)=>{ctx.strokeStyle=colors[i];ctx.fillStyle=colors[i];ctx.lineWidth=1.6;ctx.beginPath();let last=null;for(const p of envelope(series,xpos)){if(last===null||p.segment!==last)ctx.moveTo(xpos(p.x),ypos(p.y));else ctx.lineTo(xpos(p.x),ypos(p.y));last=p.segment}ctx.stroke();if(series.length===1){ctx.beginPath();ctx.arc(xpos(series[0].x),ypos(series[0].y),3,0,2*Math.PI);ctx.fill()}});
 canvas.setAttribute('aria-label',`${chart.title}. ${points.length} values. Latest: ${keys.map((k,i)=>k+' '+number(chart.series[i].at(-1)?.y,4)).join(', ')}. Horizontal axis ${$('axis').value}.`);
}
function hover(chart,pixel){if(!chart.bounds)return;const b=chart.bounds,x=b.xmin+(pixel-b.left)/(b.right-b.left)*(b.xmax-b.xmin);let text=[];
 chart.series.forEach((series,i)=>{let closest=null;for(const p of series)if(!closest||Math.abs(p.x-x)<Math.abs(closest.x-x))closest=p;if(closest){if(!text.length)text.push(`Step ${number(closest.steps,0)} · ${new Date(closest.time*1000).toLocaleString()}`);text.push(`${labels[chart.keys[i]]||chart.keys[i]}: ${number(closest.y,5)} (raw ${number(closest.raw,5)})`)}});
 chart.tip.textContent=text.join('\n');chart.tip.style.display=text.length?'block':'none';
}
function render(){if(!snapshot)return;const s=snapshot.state;
 $('badge').textContent=paused?'DISPLAY PAUSED':`${snapshot.lifecycle} / ${snapshot.phase}`;$('badge').className=!paused&&snapshot.lifecycle==='ALIVE'?'':'warn';
 $('identity').textContent=`${snapshot.run} · ${s.device||'unknown device'} · learner PID ${s.pid||'unknown'} · ${s.session_id||'no session'}`;
 for(const [id,val] of Object.entries({steps:number(s.steps,0),workers:number(s.num_envs,0),sps:number(s.rollout_sps??s.sps),updates:number(s.updates,0),commit:s.commit_memory?number(s.commit_memory.percent,1)+'%':'—'}))$(id).textContent=val;
 $('workerRows').replaceChildren();for(const worker of s.workers||[]){const tr=document.createElement('tr');for(const key of ['port','game_seed','episode_steps','episode_reward','health','rooms','combat_clears','boss_seen']){const td=document.createElement('td');td.textContent=typeof worker[key]==='number'?number(worker[key]):String(worker[key]??'—');tr.append(td)}$('workerRows').append(tr)}
 $('coverage').textContent=`${snapshot.updates.length.toLocaleString()} recorded updates · ${snapshot.episodes.length.toLocaleString()} recorded episodes. All available JSONL history is retained; older metrics that were never logged cannot be reconstructed. Smoothing is a trailing record mean, reset at session changes. Lines break on missing data or checkpoint rollback. Hover for exact steps and raw values. Success curves describe training, not held-out evaluation.`;
 $('lifecycle').textContent=`Dashboard service PID ${snapshot.server.pid}: alive · Learner: ${snapshot.lifecycle} · heartbeat ${number(snapshot.heartbeat_age,1)}s old · checkpoint ${number(s.checkpoint_steps,0)} steps. This read-only history service remains available after learner exit; it never starts or reconnects a learner. Closing this page does not stop training.`;
 charts.forEach(draw);
}
async function refresh(){if(!paused){try{const response=await fetch('/api/history',{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error(`HTTP ${response.status}`);snapshot=await response.json();$('connection').textContent='';render();$('refresh').textContent=`Updated ${new Date().toLocaleTimeString()} · every 5s`}catch(error){$('connection').textContent=`Dashboard connection lost. Charts are frozen; learner liveness cannot be verified. ${error.message}`;$('badge').textContent='DISPLAY OFFLINE';$('badge').className='warn'}}setTimeout(refresh,5000)}
for(const id of ['range','smooth','axis'])$(id).addEventListener('change',render);
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'Resume chart refresh':'Pause chart refresh';$('connection').textContent=paused?'Chart refresh paused. Training continues; displayed liveness is the last observation.':'';if(paused){$('badge').textContent='DISPLAY PAUSED';$('badge').className='warn'}});
window.addEventListener('resize',()=>{if(snapshot)charts.forEach(draw)});refresh();
