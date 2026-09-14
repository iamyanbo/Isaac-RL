// Pure chart logic tests with a synthetic canvas; no browser automation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = new Proxy({}, {get: (o,k) => o[k] || (() => {})});
function element(){return {style:{},clientWidth:600,append(){},setAttribute(){},addEventListener(){},replaceChildren(){},getContext(){return context}}}
const elements = Object.fromEntries(['charts','smooth','axis','range'].map(k=>[k,element()]));
elements.smooth.value='10';elements.axis.value='steps';elements.range.value='all';
const sandbox={console,Intl,Date,Math,Number,Set,window:{devicePixelRatio:1,addEventListener(){}},document:{getElementById:id=>elements[id]||(elements[id]=element()),createElement:element},setTimeout(){},AbortSignal,fetch:()=>new Promise(()=>{})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('isaac_rl/dashboard_assets/charts.js','utf8'),sandbox);
function evaluate(code){return vm.runInContext(code,sandbox)}
assert.equal(evaluate("seriesFor([{steps:1000001,loss:2},{steps:1234567890,loss:4}],'loss')[1].y"),3);
assert.equal(evaluate("seriesFor([{steps:5,loss:2,session_id:'a'},{steps:6,loss:4,session_id:'b'}],'loss')[1].y"),4);
assert.equal(evaluate("seriesFor([{steps:5,loss:2},{steps:4,loss:4}],'loss')[1].segment"),1);
assert.equal(evaluate("seriesFor([{steps:5,loss:2},{steps:6,loss:null},{steps:7,loss:4}],'loss')[1].y"),4);
assert.equal(evaluate("seriesFor([{steps:5,success:false},{steps:6,success:true}],'success')[1].y"),.5);
assert.equal(evaluate("envelope([{x:1,y:2,segment:0},{x:2,y:99,segment:0},{x:3,y:-4,segment:0},{x:4,y:3,segment:0}],()=>1).some(p=>p.y===99)"),true);
// More records than JavaScript's argument-spread limit; no 1M step boundary.
evaluate("snapshot={state:{steps:1234567890},updates:Array.from({length:150000},(_,i)=>({steps:1000000+i,loss:i%10})),episodes:[]}; draw(charts[0]);");
assert.equal(evaluate('charts[0].series[0].length'),150000);
console.log('Chart logic: 7 checks passed, including 150,000 records and billion-step counters.');
