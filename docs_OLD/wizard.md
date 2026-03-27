# Getting Started Wizard

Answer a few questions to get personalised setup instructions for your data.

<style>
.wcard {
  border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
  border-radius: 8px; padding: 2rem; margin: 1.5rem 0;
  background: var(--md-default-bg-color, #fff);
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.wprog { display: flex; align-items: center; margin-bottom: 2rem; }
.wbub {
  width: 32px; height: 32px; border-radius: 50%;
  border: 2px solid var(--md-default-fg-color--lightest, #e0e0e0);
  background: var(--md-default-bg-color, #fff);
  color: var(--md-default-fg-color--light, #757575);
  font-size: .8rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; transition: all .2s;
}
.wbub.won  { border-color: var(--md-primary-fg-color, #00897b); background: var(--md-primary-fg-color, #00897b); color: #fff; }
.wbub.wdone{ border-color: var(--md-primary-fg-color, #00897b); background: var(--md-primary-fg-color, #00897b); color: #fff; }
.wcon { flex: 1; height: 2px; background: var(--md-default-fg-color--lightest, #e0e0e0); transition: background .2s; }
.wcon.wdone { background: var(--md-primary-fg-color, #00897b); }

.wpage2 { display: none; }
.wpage2.won { display: block; }

.wqt { font-size: 1.05rem; font-weight: 600; margin-bottom: .4rem; }
.wqh { font-size: .87rem; color: var(--md-default-fg-color--light, #757575); margin-bottom: 1.1rem; line-height: 1.5; }

.wchoices { display: flex; flex-direction: column; gap: .5rem; list-style: none; padding: 0; margin: 0; }
.wchoice {
  display: flex; align-items: flex-start; gap: .75rem;
  padding: .65rem 1rem;
  border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
  border-radius: 6px; cursor: pointer; user-select: none;
  transition: border-color .15s, background .15s;
}
.wchoice:hover { border-color: var(--md-primary-fg-color, #00897b); background: rgba(0,137,123,.06); }
.wchoice.wsel  { border-color: var(--md-primary-fg-color, #00897b); background: rgba(0,137,123,.06); }
.wchoice input { margin-top: 2px; accent-color: var(--md-primary-fg-color, #00897b); width: 16px; height: 16px; flex-shrink: 0; cursor: pointer; }
.wclbl strong  { display: block; font-size: .93rem; font-weight: 600; }
.wclbl span    { font-size: .82rem; color: var(--md-default-fg-color--light, #757575); }

.wsubq {
  display: none; margin-top: 1rem; margin-left: 1rem;
  padding: 1rem 1.2rem;
  background: rgba(0,137,123,.04); border: 1px solid rgba(0,137,123,.2);
  border-radius: 6px;
}
.wsubq.won { display: block; }
.wsubq .wqt { font-size: .93rem; }
.wsubq .wqh { margin-bottom: .85rem; }
.wsubq .wchoices { gap: .4rem; }
.wsubq .wchoice  { padding: .5rem .85rem; }

.wnav {
  display: flex; justify-content: space-between; align-items: center;
  margin-top: 1.75rem; padding-top: 1.25rem;
  border-top: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
}
.wbtn {
  padding: .5rem 1.3rem; border-radius: 6px; border: none;
  font-size: .88rem; font-weight: 600; cursor: pointer;
  transition: background .15s; font-family: inherit;
}
.wbtn-p { background: var(--md-primary-fg-color, #00897b); color: #fff; }
.wbtn-p:hover { background: var(--md-primary-fg-color--dark, #00695c); }
.wbtn-s { background: var(--md-default-fg-color--lightest, #f5f5f5); color: var(--md-default-fg-color, #212121); border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0); }
.wbtn-g { background: transparent; color: var(--md-default-fg-color--light, #757575); font-size: .82rem; font-weight: 400; padding: .5rem; cursor: pointer; border: none; text-decoration: underline; font-family: inherit; }

.wval { color: #c62828; font-size: .83rem; margin-top: .65rem; display: none; }
.wval.won { display: block; }

.wrsec { border-radius: 6px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.wrsec.intro { background: #e8f5e9; border-left: 4px solid #2e7d32; }
.wrsec.intro h3 { color: #2e7d32; font-size: 1rem; margin: 0 0 .4rem; }
.wrsec.steps { background: var(--md-default-fg-color--lightest, #f5f5f5); border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0); }
.wrsec.steps h4 { font-size: .93rem; font-weight: 700; margin: 0 0 .65rem; }
.wrsec.steps ol { margin-left: 1.2rem; line-height: 1.85; font-size: .91rem; }
.wrsec.links { background: #e3f2fd; border-left: 4px solid #1565c0; }
.wrsec.links h4 { color: #1565c0; font-size: .93rem; font-weight: 700; margin: 0 0 .5rem; }
.wrsec.links ul { list-style: none; padding: 0; margin: 0; }
.wrsec.links li { font-size: .87rem; margin-bottom: .22rem; }
.wrsec.links a  { color: #1565c0; text-decoration: none; font-weight: 500; }
.wrsec.links a:hover { text-decoration: underline; }

.wbdg { display: inline-block; padding: .13rem .48rem; border-radius: 20px; font-size: .71rem; font-weight: 700; margin-right: .28rem; vertical-align: middle; }
.wbg  { background: #c8e6c9; color: #2e7d32; }
.wbb  { background: #bbdefb; color: #1565c0; }
.wba  { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
.wtags { margin-bottom: .65rem; }
</style>

<div class="wcard">

  <div class="wprog">
    <div class="wbub won" id="wb0">1</div>
    <div class="wcon" id="wc0"></div>
    <div class="wbub" id="wb1">2</div>
    <div class="wcon" id="wc1"></div>
    <div class="wbub" id="wb2">3</div>
    <div class="wcon" id="wc2"></div>
    <div class="wbub" id="wb3">✓</div>
  </div>

  <!-- Page 0: NWB -->
  <div class="wpage2 won" id="wp0">
    <div class="wqt">Are you working with NWB files?</div>
    <div class="wqh">NWB (<code>.nwb</code>) files are self-contained — EthoGraph loads them directly without any extra setup.</div>
    <div class="wchoices">
      <label class="wchoice">
        <input type="radio" name="wnwb" value="yes" onchange="wOnRad(this,'nwb')">
        <div class="wclbl"><strong>Yes — I have .nwb files</strong><span>From DANDI, NeuroConv, or another NWB pipeline</span></div>
      </label>
      <label class="wchoice">
        <input type="radio" name="wnwb" value="no" onchange="wOnRad(this,'nwb')">
        <div class="wclbl"><strong>No — I have raw files (video, audio, pose, ephys, …)</strong></div>
      </label>
    </div>
    <div class="wval" id="wv0">Please select an option to continue.</div>
  </div>

  <!-- Page 1: Data types + sub-questions -->
  <div class="wpage2" id="wp1">
    <div class="wqt">What do you want to visualise in EthoGraph?</div>
    <div class="wqh">Select everything that applies. More questions may appear below based on your selection.</div>
    <div class="wchoices">
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="video" onchange="wOnChk(this,'video')">
        <div class="wclbl"><strong>Videos</strong><span>.mp4 camera recordings</span></div>
      </label>
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="pose" onchange="wOnChk(this,'pose')">
        <div class="wclbl"><strong>Pose estimation</strong><span>DeepLabCut, SLEAP, LightningPose — .h5/.csv/... files</span></div>
      </label>
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="audio" onchange="wOnChk(this,'audio')">
        <div class="wclbl"><strong>Audio / Spectrogram</strong><span>.wav, .mp3, or .mp4 files with sound</span></div>
      </label>
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="ephys" onchange="wOnChk(this,'ephys')">
        <div class="wclbl"><strong>Electrophysiology</strong><span>Raw ephys (.rhd, .abf, .oebin, …) or spike-sorted units (Kilosort)</span></div>
      </label>
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="numpy" onchange="wOnChk(this,'numpy')">
        <div class="wclbl"><strong>Custom feature array</strong><span>Pre-computed signals stored as .npy</span></div>
      </label>
      <label class="wchoice">
        <input type="checkbox" name="wdtype" value="other" onchange="wOnChk(this,'other')">
        <div class="wclbl"><strong>Other / custom format</strong><span>Format not listed above — convert to a supported format</span></div>
      </label>
    </div>

    <!-- Sub: cameras (video or pose) -->
    <div class="wsubq" id="wsubCam">
      <div class="wqt">Did you record from multiple cameras?</div>
      <div class="wqh">i.e. do you have multiple video files or multiple pose files.</div>
      <div class="wchoices">
        <label class="wchoice">
          <input type="radio" name="wcam" value="single" onchange="wOnRad(this,'cameras')">
          <div class="wclbl"><strong>No — single camera</strong><span>One video file and/or one pose file</span></div>
        </label>
        <label class="wchoice">
          <input type="radio" name="wcam" value="multi" onchange="wOnRad(this,'cameras')">
          <div class="wclbl"><strong>Yes — multiple cameras</strong><span>Multiple video or pose files</span></div>
        </label>
      </div>
      <div class="wval" id="wvcam">Please answer this question.</div>
    </div>

    <!-- Sub: audio setup -->
    <div class="wsubq" id="wsubAud">
      <div class="wqt">How are your audio files organised?</div>
      <div class="wqh">This determines whether the Create dialog is sufficient or if a script is needed.</div>
      <div class="wchoices">
        <label class="wchoice">
          <input type="radio" name="waudsetup" value="single" onchange="wOnRad(this,'audio_setup')">
          <div class="wclbl"><strong>Single microphone</strong><span>One .wav (or .mp3) file</span></div>
        </label>
        <label class="wchoice">
          <input type="radio" name="waudsetup" value="multichannel" onchange="wOnRad(this,'audio_setup')">
          <div class="wclbl"><strong>Multiple microphones in one multichannel file</strong><span>E.g. all mics stored in a single .wav file with multiple channels</span></div>
        </label>
        <label class="wchoice">
          <input type="radio" name="waudsetup" value="multi_files" onchange="wOnRad(this,'audio_setup')">
          <div class="wclbl"><strong>Multiple sound files</strong><span>One separate file per microphone</span></div>
        </label>
      </div>
      <div class="wval" id="wvaud">Please answer this question.</div>
    </div>

    <div class="wval" id="wv1">Please select at least one data type.</div>
  </div>

  <!-- Page 2: Trials -->
  <div class="wpage2" id="wp2">
    <div class="wqt">Does your data have a trial structure?</div>
    <div class="wqh">Choose <strong>Yes</strong> if either: (a) you have multiple separate files per stream (e.g. 20 video clips, one per trial), or (b) you have one continuous recording but want to define discrete trial windows within it (e.g. "trial 1 starts at t=45 s, trial 2 at t=2:30 min").</div>
    <div class="wchoices">
      <label class="wchoice">
        <input type="radio" name="wtrials" value="no" onchange="wOnRad(this,'trials')">
        <div class="wclbl"><strong>No — treat the whole recording as one session</strong><span>One continuous video, audio, and/or ephys file with no defined trial boundaries</span></div>
      </label>
      <label class="wchoice">
        <input type="radio" name="wtrials" value="yes" onchange="wOnRad(this,'trials')">
        <div class="wclbl"><strong>Yes — I have trials or epochs</strong><span>Multiple files per stream, or one continuous file with defined trial start/stop times</span></div>
      </label>
    </div>
    <div class="wval" id="wv2">Please select an option to continue.</div>
  </div>

  <!-- Result -->
  <div class="wpage2" id="wpRes"><div id="wResHtml"></div></div>

  <div class="wnav">
    <button class="wbtn wbtn-s" id="wBP" onclick="wPrev()" style="display:none">← Back</button>
    <div style="display:flex;gap:1rem;align-items:center;">
      <button class="wbtn wbtn-g" id="wBR" onclick="wRestart()" style="display:none">Start over</button>
      <button class="wbtn wbtn-p" id="wBN" onclick="wNext()">Next →</button>
    </div>
  </div>

</div>

<script>
(function(){
var ws={nwb:null,dtypes:new Set(),cameras:null,audio_setup:null,trials:null};
var cur=0;

function wOnRad(inp,key){
  document.querySelectorAll('input[name="'+inp.name+'"]').forEach(function(o){o.closest('.wchoice').classList.remove('wsel');});
  inp.closest('.wchoice').classList.add('wsel');
  ws[key]=inp.value;
  wCv();
}
window.wOnRad=wOnRad;

function wOnChk(inp,value){
  inp.closest('.wchoice').classList.toggle('wsel',inp.checked);
  if(inp.checked)ws.dtypes.add(value);else ws.dtypes.delete(value);
  wUpdateSub();wCv();
}
window.wOnChk=wOnChk;

function wUpdateSub(){
  var needCam=ws.dtypes.has('video')||ws.dtypes.has('pose');
  var needAud=ws.dtypes.has('audio');
  document.getElementById('wsubCam').classList.toggle('won',needCam);
  document.getElementById('wsubAud').classList.toggle('won',needAud);
  if(!needCam){ws.cameras=null;wClearGrp('wcam');}
  if(!needAud){ws.audio_setup=null;wClearGrp('waudsetup');}
}

function wClearGrp(name){
  document.querySelectorAll('input[name="'+name+'"]').forEach(function(i){i.checked=false;i.closest('.wchoice').classList.remove('wsel');});
}

function wCv(){document.querySelectorAll('.wval').forEach(function(e){e.classList.remove('won');});}

function wProg(idx){
  for(var i=0;i<=3;i++){var b=document.getElementById('wb'+i);if(!b)continue;b.classList.remove('won','wdone');if(i<idx)b.classList.add('wdone');else if(i===idx)b.classList.add('won');}
  for(var j=0;j<3;j++){var c=document.getElementById('wc'+j);if(c)c.classList.toggle('wdone',j<idx);}
}

function wShow(id){document.querySelectorAll('.wpage2').forEach(function(p){p.classList.remove('won');});document.getElementById(id).classList.add('won');}

function wValP1(){
  var ok=true;
  if(ws.dtypes.size===0){document.getElementById('wv1').classList.add('won');ok=false;}
  if(ws.dtypes.has('video')&&ws.dtypes.has('pose')&&!ws.cameras){document.getElementById('wvcam').classList.add('won');ok=false;}
  if(ws.dtypes.has('audio')&&!ws.audio_setup){document.getElementById('wvaud').classList.add('won');ok=false;}
  return ok;
}

function wNext(){
  wCv();
  if(cur===0){if(!ws.nwb){document.getElementById('wv0').classList.add('won');return;}if(ws.nwb==='yes'){wRes();return;}cur=1;wShow('wp1');wProg(1);document.getElementById('wBP').style.display='';document.getElementById('wBN').textContent='Next →';return;}
  if(cur===1){if(!wValP1())return;if(ws.dtypes.size===1&&ws.dtypes.has('other')){wRes();return;}cur=2;wShow('wp2');wProg(2);document.getElementById('wBN').textContent='See my setup →';return;}
  if(cur===2){if(!ws.trials){document.getElementById('wv2').classList.add('won');return;}wRes();}
}
window.wNext=wNext;

function wPrev(){
  wCv();
  if(cur===1){cur=0;wShow('wp0');wProg(0);document.getElementById('wBP').style.display='none';document.getElementById('wBN').textContent='Next →';}
  else if(cur===2){cur=1;wShow('wp1');wProg(1);document.getElementById('wBN').textContent='Next →';}
  else if(cur==='r'){var back=ws.nwb==='yes'?0:2;cur=back;wShow('wp'+back);wProg(back);document.getElementById('wBN').style.display='';document.getElementById('wBN').textContent=back===2?'See my setup →':'Next →';document.getElementById('wBP').style.display=back>0?'':'none';document.getElementById('wBR').style.display='none';}
}
window.wPrev=wPrev;

function wRestart(){
  ws.nwb=null;ws.dtypes.clear();ws.cameras=null;ws.audio_setup=null;ws.trials=null;
  document.querySelectorAll('input[type=radio],input[type=checkbox]').forEach(function(i){i.checked=false;});
  document.querySelectorAll('.wchoice').forEach(function(e){e.classList.remove('wsel');});
  document.querySelectorAll('.wsubq').forEach(function(e){e.classList.remove('won');});
  cur=0;wShow('wp0');wProg(0);
  document.getElementById('wBP').style.display='none';
  document.getElementById('wBN').style.display='';document.getElementById('wBN').textContent='Next →';
  document.getElementById('wBR').style.display='none';wCv();
}
window.wRestart=wRestart;

function wRes(){
  cur='r';wShow('wpRes');wProg(3);
  document.getElementById('wBP').style.display='';document.getElementById('wBN').style.display='none';document.getElementById('wBR').style.display='';
  document.getElementById('wResHtml').innerHTML=wBuild();
}

function wBuild(){
  if(ws.nwb==='yes')return rNWB();
  var d=ws.dtypes;
  if(d.has('other'))return rOther();
  var needsMultiSetup=ws.trials==='yes'||ws.cameras==='multi'||ws.audio_setup==='multi_files';
  if(d.has('ephys')&&needsMultiSetup)return rEphysMulti();
  if(d.has('ephys'))return rEphysSingle();
  if(needsMultiSetup)return rScript();
  if(d.size===1&&d.has('audio'))return rAudioOnly();
  if(d.size===1&&d.has('numpy'))return rNumpy();
  return rSimple();
}

function tags(){
  var d=ws.dtypes,t='';
  if(d.has('video'))t+='<span class="wbdg wbg">Video</span>';
  if(d.has('pose')) t+='<span class="wbdg wbb">Pose</span>';
  if(d.has('audio'))t+='<span class="wbdg wba">Audio</span>';
  if(d.has('ephys'))t+='<span class="wbdg wbb">Ephys</span>';
  if(d.has('numpy'))t+='<span class="wbdg wbb">Numpy</span>';
  if(d.has('other'))t+='<span class="wbdg wbb">Other</span>';
  return '<div class="wtags">'+t+'</div>';
}

function rOther(){
  return s('intro','<h3>Data format not listed above</h3><div class="wtags"><span class="wbdg wbb">Other</span></div><p>EthoGraph doesn\'t have a built-in loader for your format. Convert or wrap your data using one of these options:</p>')
  +s('steps','<h4>Options</h4><ul style="margin-left:1.2rem;line-height:1.85;font-size:.91rem;">'
  +'<li><strong>A — Convert to <code>.npy</code></strong>: Save as a numpy array and use the <em>4) Generate from npy file</em> dialog. For high-SR data that loads slowly, enable the <strong>Downsample</strong> checkbox in the I/O widget.</li>'
  +'<li><strong>B — Convert to <code>.wav</code></strong>: For high-SR periodic data you want to visualise quickly (e.g. LFP, EMG, pressure). Use <code>audioio.write_audio()</code> to convert, then load as audio. Min/max downsampling makes waveform and spectrogram rendering fast — no manual downsample needed.</li>'
  +'<li><strong>C — xarray script</strong>: For multi-dimensional arrays (&gt;2-D). Wrap your data in an <code>xr.Dataset</code> with <code>eto.dataset_to_basic_trialtree()</code>. For high-SR data, use the Downsample checkbox or <code>eto.downsample_trialtree(dt, factor)</code> before saving.</li>'
  +'</ul>')
  +s('links','<h4>Full guide</h4><ul><li><a target="_blank" rel="noopener" href="../loading-other/">Other / custom formats</a></li></ul>');
}
function s(c,h){return '<div class="wrsec '+c+'">'+h+'</div>';}

var WLAUNCH='<li><a target="_blank" rel="noopener" href="../installation/">Install EthoGraph</a> if you haven\'t already</li>'
+'<li>Launch EthoGraph — double-click the desktop shortcut, or run: <code>conda activate ethograph &amp;&amp; ethograph launch</code></li>';
function rNWB(){
  return s('intro','<h3>NWB — no extra setup needed</h3><p>EthoGraph reads <code>.nwb</code> files directly.</p>')
  +s('steps','<h4>Steps</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click the file browser next to <em>Session data</em></li><li>Select your <code>.nwb</code> file</li><li><em>Optional:</em> select a local video folder if videos are not embedded</li><li>Click <strong>Load</strong></li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../data-loading/">Loading data</a></li></ul>');
}

function rSimple(){
  var d=ws.dtypes;
  var hasPose=d.has('pose'),hasAud=d.has('audio');
  var wizLabel=hasPose?'1) Generate from pose file (DeepLabCut, SLEAP, ...)':hasAud?'3) Generate from audio file':'4) Generate from npy file';
  var audNote='';
  if(hasAud&&ws.audio_setup==='multichannel')audNote='<li>For <strong>Audio file</strong>, select your multichannel <code>.wav</code></li>';
  else if(hasAud&&ws.audio_setup==='single'&&(d.has('video')||hasPose))audNote='<li>Also set <strong>Audio file</strong></li>';
  return s('intro','<h3>Single recording — use the Create dialog</h3>'+tags()+'<p>The built-in <em>➕Create with own data</em> dialog handles this without any scripting.</p>')
  +s('steps','<h4>Steps</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Under <strong>Single trial</strong>, select: <strong>'+wizLabel+'</strong></li><li>Click <strong>Next →</strong> — the dialog opens</li><li>Fill in the file paths</li>'+audNote+'<li>Click <strong>Generate .nc file</strong> — fields auto-populate, then click <strong>Load</strong></li></ol>')
  +s('links','<h4>Relevant docs</h4><ul>'+(hasPose?'<li><a target="_blank" rel="noopener" href="../loading-pose/">From a pose file</a></li>':(!hasPose&&hasAud?'<li><a target="_blank" rel="noopener" href="../loading-audio/">From an audio file</a></li>':'<li><a target="_blank" rel="noopener" href="../loading-numpy/">From a numpy file</a></li>'))+'<li><a target="_blank" rel="noopener" href="../data-loading/#try-the-gui-with-template-datasets">Try template datasets first</a></li></ul>');
}

function rAudioOnly(){
  var mc=ws.audio_setup==='multichannel';
  return s('intro','<h3>Audio-only mode</h3><div class="wtags"><span class="wbdg wba">Audio</span></div><p>EthoGraph supports datasets with no video. A time slider replaces the video player.'+(mc?' Your multichannel file is loaded as-is.':'')+'</p>')
  +s('steps','<h4>Steps</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Under <strong>Single trial</strong>, select: <strong>3) Generate from audio file</strong></li><li>Click <strong>Next →</strong> — the dialog opens</li><li>Set your <code>'+(mc?'multichannel .wav':'.wav / .mp3')+'</code> file</li><li>Click <strong>Generate .nc file</strong> — fields auto-populate, then click <strong>Load</strong></li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../loading-audio/">From an audio file</a></li><li><a target="_blank" rel="noopener" href="../data-requirements/#audio-only-no-video">Data requirements — audio-only</a></li></ul>');
}

function rNumpy(){
  return s('intro','<h3>Custom numpy feature</h3><div class="wtags"><span class="wbdg wbb">Numpy</span></div><p>For pre-computed signals stored as <code>.npy</code>. Shape: <code>(n_samples, n_variables)</code> or transpose.</p>')
  +s('steps','<h4>Steps</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Under <strong>Single trial</strong>, select: <strong>4) Generate from npy file</strong></li><li>Click <strong>Next →</strong> — the dialog opens</li><li>Set the <code>.npy</code> file and the <strong>Data sampling rate</strong></li><li>Click <strong>Generate .nc file</strong> — fields auto-populate, then click <strong>Load</strong></li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../loading-numpy/">From a numpy file</a></li></ul>');
}

function rEphysSingle(){
  return s('intro','<h3>Electrophysiology — single session</h3><div class="wtags"><span class="wbdg wbb">Ephys</span></div><p>Use the <em>➕Create with own data</em> dialog. Video and audio are optional add-ons.</p>')
  +s('steps','<h4>Steps</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Under <strong>Single trial</strong>, select: <strong>5) Generate from ephys file and/or kilosort folder</strong></li><li>Click <strong>Next →</strong> — the dialog opens</li><li>Set <strong>Ephys file</strong> and/or <strong>Kilosort folder</strong></li><li><em>Optional:</em> add <strong>Video file</strong> or <strong>Audio file</strong></li><li>Click <strong>Generate .nc file</strong> — fields auto-populate, then click <strong>Load</strong></li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../loading-ephys/">Ephys data — formats &amp; Kilosort</a></li></ul>');
}

function rScript(){
  var reason=ws.cameras==='multi'?'You recorded from <strong>multiple cameras</strong>, each with its own video and pose file.':ws.audio_setup==='multi_files'?'You have <strong>multiple separate audio files</strong> (one per microphone).':'Your data spans <strong>multiple trials</strong>.';
  var camNote=ws.cameras==='multi'?'<li>Set video per camera: <code>dt.set_media("video", [[cam1.mp4, cam2.mp4]])</code></li><li>Set pose per camera: <code>dt.set_media("pose", [[cam1.h5, cam2.h5]])</code></li>':'';
  var audNote=ws.audio_setup==='multi_files'?'<li>Set mic files: <code>dt.set_media("audio", [[mic1.wav, mic2.wav]])</code></li>':ws.audio_setup==='multichannel'?'<li>Set audio: <code>dt.set_media("audio", [[recording.wav]])</code></li>':'';
  return s('intro','<h3>Multi-trial dataset</h3>'+tags()+'<p>'+reason+' You have two options: use the <strong>built-in GUI wizard</strong> (no coding) or write a short Python script.</p>')
  +s('steps','<h4>Option A — GUI wizard (no coding)</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Select <strong>Multiple trials</strong></li><li>Check the modalities you want (Video, Pose, Audio, …) and choose <em>Files aligned to trial period</em> or <em>Continuous recording across session</em></li><li>Configure file patterns, device labels, and offsets per modality</li><li>Set trial boundaries and output path, then click <strong>Generate .nc file</strong></li></ol>')
  +s('steps','<h4>Option B — Python script</h4><ol><li><a target="_blank" rel="noopener" href="../installation/">Install EthoGraph</a> if you haven\'t already</li><li>Follow the setup guide: <a target="_blank" rel="noopener" href="../loading-script/">Multi-trial setup</a></li><li>Create one <code>xr.Dataset</code> per trial with features tagged <code>type="features"</code></li><li>Call <code>eto.from_datasets(datasets)</code></li>'+camNote+audNote+'<li>Save: <code>dt.save("trials.nc")</code></li><li>Launch EthoGraph and load the <code>.nc</code> file and your media folders</li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../loading-script/">Multi-trial setup guide (scripting)</a></li><li><a target="_blank" rel="noopener" href="../trialtree/">TrialTree reference</a></li><li><a target="_blank" rel="noopener" href="../tutorials/">Tutorial notebooks</a></li></ul>');
}

function rEphysMulti(){
  return s('intro','<h3>Ephys + multi-trial dataset</h3><div class="wtags"><span class="wbdg wbb">Ephys</span><span class="wbdg wbg">Multi-trial</span></div><p>Ephys is a session-wide stream. Create a <code>trials.nc</code> for the behavioural structure and point EthoGraph to the ephys file separately in the GUI. You have two options:</p>')
  +s('steps','<h4>Option A — GUI wizard (no coding)</h4><ol>'+WLAUNCH+'<li>In the <strong>I/O widget</strong>, click <strong>➕Create with own data</strong></li><li>Select <strong>Multiple trials</strong></li><li>Check <strong>Ephys</strong> (session-wide) plus any other modalities</li><li>Configure file patterns and offsets, then click <strong>Generate .nc file</strong></li><li>After loading, separately select the ephys file / Kilosort folder in the I/O widget</li></ol>')
  +s('steps','<h4>Option B — Python script</h4><ol><li><a target="_blank" rel="noopener" href="../installation/">Install EthoGraph</a> if you haven\'t already</li><li>Build your <code>trials.nc</code> with <code>eto.from_datasets()</code></li><li>If clocks differ: <code>dt.set_stream_offset("ephys", offset_s)</code></li><li>Save: <code>dt.save("trials.nc")</code></li><li>Launch EthoGraph, load the <code>.nc</code>, then separately select the ephys file / Kilosort folder</li></ol>')
  +s('links','<h4>Relevant docs</h4><ul><li><a target="_blank" rel="noopener" href="../loading-ephys/">Ephys data</a></li><li><a target="_blank" rel="noopener" href="../loading-script/">Multi-trial setup guide (scripting)</a></li><li><a target="_blank" rel="noopener" href="../trialtree/#stream-offsets">TrialTree — stream offsets</a></li></ul>');
}
})();
</script>
