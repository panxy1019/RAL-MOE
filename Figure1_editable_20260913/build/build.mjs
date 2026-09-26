import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const root='C:/Users/panxy1019/Documents/CHANNEL/Figure1_editable_20260913';
const skill='C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const python='C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const assets='C:/Users/panxy1019/Documents/CHANNEL/RAL_MOE_ROM_ICLR2027/figures/original';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Arial'});
const P=Presentation.create({slideSize:{width:1600,height:1000}});
const s=P.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#222222',line:'#363B40',blue:['#EDF4FA','#638CAC'],orange:['#FFF3E9','#D88C53'],purple:['#F3EFF8','#9475AD'],green:['#EDF6EF','#71A07B'],gray:['#F4F5F6','#969DA3']};
let count=0;
function shape(x,y,w,h,fill='none',stroke='none',bw=0,name=''){
 return s.shapes.add({name:name||`shape-${++count}`,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:bw}});
}
function txt(x,y,w,h,t,size=28,bold=false,align='center',family=font,color=C.ink){
 const z=shape(x,y,w,h,'none','none',0,`text-${t.replace(/\n/g,' ')}`);z.text=t;
 z.text.style={typeface:family,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return z;
}
function box(x,y,w,h,title,sub='',role='gray',size=27){
 const z=shape(x,y,w,h,...C[role],1.45,title);
 z.text=sub?[[{run:title,textStyle:{bold:true}}],[{run:sub,textStyle:{fontSize:`${size-1}px`}}]]:title;
 z.text.style={typeface:font,fontSize:size,bold:!sub,color:C.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:6,right:6,top:4,bottom:4}};return z;
}
function anchor(x,y){return shape(x-.1,y-.1,.2,.2);}
function con(a,b,from='right',to='left',kind='straight',arrow=true,dash=false,width=2){
 const z=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:C.line,width,style:dash?'dashed':'solid'},tail:{type:arrow?'triangle':'none',width:'sm',length:'sm'}});z.bringToFront();return z;
}
function line(x1,y1,x2,y2,arrow=false,dash=false,width=2){return con(anchor(x1,y1),anchor(x2,y2),'right','left','straight',arrow,dash,width);}
function route(points,arrow=true,dash=false,width=2){for(let i=1;i<points.length;i++)line(...points[i-1],...points[i],arrow&&i===points.length-1,dash,width);}
function circle(x,y,d,text){const q=s.shapes.add({geometry:'ellipse',name:`sum-${text}`,position:{left:x,top:y,width:d,height:d},fill:'#FFFFFF',line:{fill:C.line,width:1.8}});q.text=text;q.text.style={typeface:font,fontSize:text.length>1?22:32,alignment:'center',verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};return q;}

// Panel a: one reading spine, with the library and history outside it.
txt(28,22,520,42,'a   Two-level hierarchy',32,true,'left');
const mu=txt(26,141,65,60,'μ',42,false,'center','Cambria Math');
const e2=box(131,121,167,100,'E2 router','parameter-only','purple',24);
const sel=box(341,121,244,100,'Applicability +\nselection','Top-1 / adj. Top-2','purple',25);
const roll=box(635,96,237,150,'Independent rollout','native coordinates','orange',28);
// Short separate editable labels retain the selected-pair meaning.
roll.text=[[{run:'Selected r / (r, s)',textStyle:{bold:true,fontSize:'26px'}}],[{run:'Independent',textStyle:{fontSize:'27px'}}],[{run:'rollout',textStyle:{fontSize:'27px'}}],[{run:'native coordinates',textStyle:{fontSize:'24px'}}]];
const dec=box(920,121,163,100,'Decode','physical space','blue',24);
const fusion=box(1126,115,211,112,'T2-C fusion','α(μ̃, π̄, dₙ)','green',29);
const out=box(1380,121,190,100,'Predicted fields','û(t), p̂°(t)','green',25);
con(mu,e2);con(e2,sel,'right','left','straight',true,false,2.6);con(sel,roll,'right','left','straight',true,false,2.6);con(roll,dec);con(dec,fusion);con(fusion,out);
txt(295,80,85,35,'π(μ)',27,false,'center','Cambria Math');
txt(1356,239,225,66,'Top-1: identity\nTop-2: fusion',24);

txt(75,274,430,35,'Regime-local specialists',27,true,'left');
const names=['Steady','Hopf','Periodic'];
const codes=['S','H','P'];const accents=['#5F8CAB','#D98B52','#70A277'];
for(let i=0;i<3;i++){
 const x=75+i*185;
 shape(x,321,169,122,'#FFFFFF','#B8BEC4',1.1,`${names[i]} specialist card`);
 shape(x,321,169,4,accents[i]);
 txt(x+2,329,165,32,`${names[i]}  ${codes[i]}`,26,true);
 const bytes=await fs.readFile(path.join(assets,`${names[i].toLowerCase()}_physical_fields.png`));
 // Native, reversible PPT crop of the first CFD-reference velocity panel.
 // Retain the whole cylinder and wake, remove panel labels and blank margins.
 const im=s.images.add({blob:new Uint8Array(bytes),contentType:'image/png',alt:`${names[i]} circular-cylinder CFD velocity-magnitude reference, top-left panel of ${names[i].toLowerCase()}_physical_fields.png`,fit:'contain',position:{left:x+8,top:366,width:153,height:67.45}});
 im.lockAspectRatio=false;
 im.crop={left:.1225,top:.075,right:.585,bottom:.764};
}
line(159,314,604,314,false,false,1.3);
for(const x of [159,344,529])line(x,314,x,321,false,false,1.3);
route([[604,314],[604,265],[463,265],[463,221]],true,false,1.6);
txt(80,451,530,31,'Admissible pairs: S–H or H–P',25,false,'left');
const history=box(698,371,239,70,'Physical history','Hₙ','blue',27);
const desc=box(1101,371,252,70,'History descriptor','dₙ (chart-independent)','blue',24);
route([[817,371],[817,296],[753,296],[753,246]],true,false,1.8);
txt(704,309,219,34,'initialization',25);
con(history,desc,'right','left','straight',true,false,1.8);
route([[1227,371],[1227,227]],true,false,1.8);
// A single zoom relationship, not an algorithmic feedback path.
line(652,257,623,496,false,true,1.1);

// Panel b: velocity dynamics plus algebraic pressure, with local MoE detail.
txt(28,514,610,42,'b   Regime-local specialist',32,true,'left');
const h=box(28,638,99,88,'History','Hₙ','blue',27);
const pod=box(168,630,157,104,'POD encode','chart-local','blue',25);
const local=box(370,630,168,104,'Local history','(aₙ, bₙ)','blue',28);
const gal=box(602,576,264,77,'Galerkin backbone','Fᵣᴳ','blue',28);
const moe=box(602,718,264,77,'Sparse MoE','velocity correction ρᵘ','orange',25);
const sum=circle(920,661,42,'+');
const step=box(1010,637,159,88,'Stepᵣ','chart-specific','gray',24);
const pressure=box(1280,624,216,116,'Pressure map','algebraic','blue',28);
pressure.text=[[{run:'Pressure map',textStyle:{bold:true}}],[{run:'algebraic',textStyle:{fontSize:'26px'}}],[{run:'γ ⊙ Qᴾᴾ + ρᵖ',textStyle:{typeface:'Cambria Math',fontSize:'29px'}}]];
const bnext=txt(1510,655,75,54,'bₙ₊₁',32,false,'center','Cambria Math');
con(h,pod);con(pod,local);
route([[538,682],[568,682],[568,614],[602,614]]);
route([[568,682],[568,756],[602,756]]);
route([[866,614],[941,614],[941,661]]);
route([[866,756],[941,756],[941,703]]);
con(sum,step);
con(step,pressure);
con(pressure,bnext);
txt(955,589,88,40,'ȧ',34,false,'center','Cambria Math');
txt(1172,639,106,45,'aₙ₊₁',31,false,'center','Cambria Math');
txt(972,754,239,35,'Velocity update',26);
txt(1250,754,280,35,'Pressure reconstruction',26);

// Inset: shared expert bypasses sparse selection; routed experts are parallel.
shape(363,829,690,144,'#FFFAF6','#DDA271',1.0,'Sparse-MoE detail inset');
line(615,795,480,829,false,true,1.05);
line(852,795,1020,829,false,true,1.05);
const features=txt(375,898,56,46,'ξₙ',30,false,'center','Cambria Math');
const router=box(452,887,189,68,'Group Top-1','channel Top-2','purple',24);
const shared=box(689,841,209,45,'Shared expert  E₀','','orange',24);
const routed=box(689,912,209,45,'Top-2  Eₑ₁, Eₑ₂','','orange',24);
const mix=circle(978,884,40,'Σω');
route([[431,921],[438,921],[438,863],[689,863]],true,false,1.65);
con(features,router,'right','left','straight',true,false,1.65);
con(router,routed,'right','left','elbow',true,false,1.65);
route([[898,864],[946,864],[946,904],[978,904]],true,false,1.65);
route([[898,935],[946,935],[946,904]],false,false,1.65);
txt(1079,850,435,44,'Structured expert responses',26,true,'left');
txt(1079,897,435,62,'nonlinear + linear\n+ low-rank quadratic',25,false,'left');

s.speakerNotes.textFrame.setText(`RAL-MoE-ROM method schematic. Native editable text, shapes and connectors. Three circular-cylinder reference images are native crops of existing CFD-reference velocity panels, not newly computed results or T2-C predictions. Sources: ${assets}/steady_physical_fields.png; ${assets}/hopf_physical_fields.png; ${assets}/periodic_physical_fields.png. The Hopf reference is near onset and weakly perturbed. Color bars are omitted for qualitative regime identification; these thumbnails are not a quantitative cross-regime magnitude comparison.\nMethod source: C:/Users/panxy1019/Documents/CHANNEL/PMD_Galerkin_Pan_ICLR (1)/sections/method.tex. Gamma pressure gating retained from current manuscript. Group Top-1 and channel Top-2 refer to hierarchical sparse routing; the shared expert bypasses sparse selection. The two selected specialists retain independent native states. Physical-space assembly is identity for Top-1 and T2-C for an adjacent Top-2 pair. No fused-state feedback.\nDesign brief: C:/Users/panxy1019/Downloads/RAL_MoE_ROM_Figure1_Nature_style_redraw_plan.md. User requested three regime thumbnails, overriding the brief's thumbnail-removal recommendation.\nVisual references supplied by user: https://www.nature.com/articles/s41467-021-26434-1 ; https://www.nature.com/articles/s42256-023-00685-7 ; https://www.nature.com/articles/s42256-024-00938-z ; https://www.nature.com/articles/s43588-025-00904-8 ; https://www.nature.com/articles/s41467-026-70245-1 . Article pages for PINN-SR and S3GM were accessible; several dedicated figure pages returned access redirects. No published artwork is reproduced.`);

await (await PresentationFile.exportPptx(P)).save(path.join(root,'build/candidate.pptx'));
const img=await P.export({slide:s,format:'png',scale:1.5});
await fs.writeFile(path.join(root,'build/preview.png'),new Uint8Array(await img.arrayBuffer()));
await fs.writeFile(path.join(root,'build/slide.layout.json'),await (await s.export({format:'layout'})).text());
await fs.writeFile(path.join(root,'build/presentation.json'),JSON.stringify(P.toProto()));
console.log('Draft and preview exported');
if(process.argv.includes('--final')){
 const result=await finalizePresentation({workspaceDir:root,candidatePath:path.join(root,'build/candidate.pptx'),finalPath:path.join(root,'output/Figure1_Nature_editable_final.pptx'),pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','15240000,9525000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Arial','Cambria Math']},verifyArtifactToolImport:true,receiptPath:path.join(root,'build/final-validation.json')});
 console.log(JSON.stringify(result));
}
