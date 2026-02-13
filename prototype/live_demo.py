import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Gestellence", page_icon="🤙", layout="centered")
st.title("🤙 Gestellence – Live Gesture Translation")
st.caption("Academic Project · MUJ 2026")
st.info("**Gestures:** ✋ STOP | ✊ YES | ✌️ NO | 🤙 CALL ME | 👌 OK", icon="ℹ️")

components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  *{box-sizing:border-box;}
  body{margin:0;padding:8px;background:#0e1117;font-family:'Segoe UI',sans-serif;display:flex;flex-direction:column;align-items:center;}
  #wrap{position:relative;width:100%;max-width:620px;}
  video,canvas{width:100%;border-radius:10px;display:block;}
  canvas{position:absolute;top:0;left:0;pointer-events:none;}
  #box{margin-top:10px;padding:12px;background:#1e2130;border-radius:10px;text-align:center;width:100%;}
  #gest{font-size:2rem;font-weight:700;color:#00e676;}
  #sub{font-size:0.8rem;color:#888;margin-top:4px;}
  #btn{margin-top:10px;padding:12px 40px;background:#ff4b4b;color:#fff;border:none;
       border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;display:block;
       width:200px;margin-left:auto;margin-right:auto;}
  #btn:hover{background:#cc3333;}
</style>
</head>
<body>
<div id="wrap">
  <video id="v" autoplay playsinline muted></video>
  <canvas id="c"></canvas>
</div>
<div id="box">
  <div id="gest">No Hand</div>
  <div id="sub">Click START to begin</div>
</div>
<button id="btn">▶ START</button>

<script type="module">
import{HandLandmarker,FilesetResolver}from"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const v=document.getElementById("v"),c=document.getElementById("c");
const x=c.getContext("2d"),gest=document.getElementById("gest");
const sub=document.getElementById("sub"),btn=document.getElementById("btn");

const CONN=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
            [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
const CLR={STOP:"#ff4b4b",YES:"#00e676",NO:"#ff9800","CALL ME":"#29b6f6",OK:"#ce93d8"};

let hl,running=false,ts=0;

function gesture(lm){
  const up=(t,b)=>t.y<b.y,dn=(t,b)=>t.y>b.y;
  if(up(lm[8],lm[6])&&up(lm[12],lm[10])&&up(lm[16],lm[14])&&up(lm[20],lm[18]))return"STOP";
  if(dn(lm[8],lm[6])&&dn(lm[12],lm[10])&&dn(lm[16],lm[14])&&dn(lm[20],lm[18]))return"YES";
  if(up(lm[8],lm[6])&&up(lm[12],lm[10])&&dn(lm[16],lm[14])&&dn(lm[20],lm[18]))return"NO";
  if(lm[4].y<lm[6].y&&up(lm[20],lm[18])&&dn(lm[8],lm[6])&&dn(lm[12],lm[10])&&dn(lm[16],lm[14]))return"CALL ME";
  if(Math.hypot(lm[4].x-lm[8].x,lm[4].y-lm[8].y)<0.05&&up(lm[12],lm[10])&&up(lm[16],lm[14])&&up(lm[20],lm[18]))return"OK";
  return"UNKNOWN";
}

function draw(lm,W,H){
  x.strokeStyle="rgba(0,200,200,0.8)";x.lineWidth=2;
  for(const[a,b]of CONN){x.beginPath();x.moveTo(lm[a].x*W,lm[a].y*H);x.lineTo(lm[b].x*W,lm[b].y*H);x.stroke();}
  for(const p of lm){x.beginPath();x.arc(p.x*W,p.y*H,5,0,6.28);x.fillStyle="white";x.fill();}
}

function loop(){
  if(!running)return;
  c.width=v.videoWidth;c.height=v.videoHeight;
  const W=c.width,H=c.height,now=performance.now();
  if(now===ts){requestAnimationFrame(loop);return;}ts=now;
  x.save();x.scale(-1,1);x.drawImage(v,-W,0,W,H);x.restore();
  const r=hl.detectForVideo(v,now);
  if(r.landmarks?.length){
    const g=gesture(r.landmarks[0]);
    for(const lm of r.landmarks)draw(lm.map(p=>({...p,x:1-p.x})),W,H);
    gest.textContent=g;gest.style.color=CLR[g]||"#fff";
    sub.textContent=r.landmarks.length+" hand(s) detected";
  }else{
    gest.textContent="No Hand";gest.style.color="#555";
    sub.textContent="Show your hand to the camera";
  }
  requestAnimationFrame(loop);
}

btn.addEventListener("click",async()=>{
  if(running){
    running=false;btn.textContent="▶ START";
    v.srcObject?.getTracks().forEach(t=>t.stop());v.srcObject=null;return;
  }
  if(!hl){
    sub.textContent="Loading model...";btn.disabled=true;
    const vision=await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    hl=await HandLandmarker.createFromOptions(vision,{
      baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",delegate:"GPU"},
      runningMode:"VIDEO",numHands:2,
      minHandDetectionConfidence:0.7,minHandPresenceConfidence:0.7,minTrackingConfidence:0.7
    });
    btn.disabled=false;
  }
  v.srcObject=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
  await v.play();running=true;btn.textContent="⏹ STOP";
  sub.textContent="Camera running...";
  requestAnimationFrame(loop);
});
</script>
</body>
</html>
""", height=700, scrolling=False)