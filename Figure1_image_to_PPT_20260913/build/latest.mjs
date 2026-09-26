import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='C:/Users/panxy1019/Documents/CHANNEL/Figure1_image_to_PPT_20260913';
const skill='C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const runtime='C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const source='C:/Users/PANXY1~1/AppData/Local/Temp/codex-clipboard-ff556869-1376-4d06-aa37-ef328edd6874.png';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const p=Presentation.create({slideSize:{width:1672,height:941}}),s=p.slides.add();s.background.fill='#FFFFFF';
const blue='#0086FF',orange='#FF9029',purple='#9548FF',gray='#8496A7',green='#298C32',red='#FF4949';
let n=0;
function grad(a,b){return {type:'gradient',gradientKind:'linear',angleDeg:25,stops:[{offset:0,color:a},{offset:100000,color:b}]};}
const fills={blue:grad('#EFF8FF','#CEE5F8'),orange:grad('#FFFAF2','#FFEAD5'),purple:grad('#F3EAFF','#E7D9FA'),gray:grad('#F1F3F5','#DDE3E8'),green:grad('#F4FCF2','#D9F0D7'),red:grad('#FFF5F1','#FFDFDD')};
function rect(x,y,w,h,fill='none',stroke='none',radius=5,dash=false){return s.shapes.add({name:`module-${++n}`,geometry:'roundRect',position:{left:x,top:y,width:w,height:h},borderRadius:radius,fill,line:{fill:stroke,width:stroke==='none'?0:1.2,style:dash?'dashed':'solid'}});}
function text(x,y,w,h,t,size=23,bold=false,align='center',font='Times New Roman',italic=false){const q=rect(x,y,w,h);q.text=t;q.text.style={typeface:font,fontSize:size,bold,italic,color:'#080808',alignment:align,verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};return q;}
function point(x,y){return rect(x-.05,y-.05,.1,.1);}
function rho(x,y,kind){text(x,y,23,32,'ρ',29,false,'center','Cambria Math',true);text(x+19,y-4,17,18,kind,18,false,'left','Cambria Math',true);text(x+19,y+19,29,18,'θᵣ',17,false,'left','Cambria Math',true);}
function line(x1,y1,x2,y2,arrow=false,color='#111111',width=2,dash=false){const q=s.shapes.connect(point(x1,y1),point(x2,y2),{kind:'straight',fromSide:'right',toSide:'left',line:{fill:color,width,style:dash?'dashed':'solid'},tail:{type:arrow?'triangle':'none',width:'med',length:'med'}});q.bringToFront();return q;}
function route(points,color='#111111',width=2){for(let i=1;i<points.length;i++)line(...points[i-1],...points[i],i===points.length-1,color,width);}
function taper(x,y,w,h,color=blue,fill=fills.blue){return s.shapes.add({name:'Editable tapered module',geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:color,width:1.4},customPaths:[{width:w,height:h,commands:[{moveTo:{x:0,y:0}},{lineTo:{x:w,y:h*.20}},{lineTo:{x:w,y:h*.80}},{lineTo:{x:0,y:h}},{close:{}}]}]});}
const bytes=new Uint8Array(await fs.readFile(source));
function crop(x,y,w,h){const im=s.images.add({blob:bytes,contentType:'image/png',alt:'Original user-supplied flow illustration',position:{left:x,top:y,width:w,height:h},fit:'contain'});im.lockAspectRatio=false;im.crop={left:x/1672,top:y/941,right:(1672-x-w)/1672,bottom:(941-y-h)/941};return im;}
rect(6,3,1660,506,'#FFFFFF','#33B6FA',27,true);
rect(6,534,1660,392,'#FFF7ED',orange,27,true);
text(25,13,1080,46,'(a) Two-level RAL-MoE-ROM hierarchy',36,true,'left');
text(25,541,1030,43,'(b) One regime-local specialist (e.g., Hopf specialist)',33,true,'left');

rect(24,119,137,119,fills.purple,purple,7);
text(59,141,62,39,'μ',39,true,'center','Cambria Math',true);
text(31,178,122,46,'(physical\nparameter)',22);
taper(202,106,160,125,'#88A3BC',grad('#E8EDF1','#C9D7E0'));
text(219,125,126,29,'E2',30,true);
text(211,153,143,48,'parameter-only\nrouter',23);
text(222,220,113,40,'π(μ)',31,false,'center','Cambria Math',true);
rect(431,112,273,86,fills.gray,gray,7);
text(441,119,252,28,'Applicability + selection',23,true);
text(439,145,255,25,'Top-1 or adjacent Top-2',22);
text(452,169,231,25,'(S–H or H–P)',22,false,'center','Times New Roman',true);
rect(337,234,405,178,'#FFFFFF','#8F98A4',17,true);
text(350,239,380,30,'Candidate regime-local specialists',23,true);
for(const c of [{x:349,w:120,fill:fills.blue,color:blue,t:'Steady',sym:'S',crop:[358,348,103,44]},{x:482,w:121,fill:fills.orange,color:orange,t:'Hopf',sym:'H',crop:[491,348,104,44]},{x:615,w:119,fill:fills.green,color:green,t:'Periodic',sym:'P',crop:[623,348,103,44]}]){
 rect(c.x,272,c.w,127,c.fill,c.color,7);text(c.x+5,278,c.w-10,45,c.t+'\nspecialist',21,true);text(c.x+5,323,c.w-10,25,c.sym,25,true,'center','Times New Roman',true);crop(...c.crop);
}
rect(753,67,241,325,fills.orange,orange,8);
text(765,73,215,29,'Selected specialist(s)',25,true);
rect(769,108,211,109,fills.orange,orange,7);text(780,112,190,29,'Specialist r',23,true);crop(777,145,195,59);
text(851,217,45,31,'⋮',29,true);
rect(769,249,211,131,fills.orange,orange,7);text(780,254,190,28,'Specialist s',23,true);text(785,280,180,27,'(if selected)',22);crop(777,309,195,60);
taper(1023,133,127,182);text(1029,177,115,49,'Independent\nlocal rollouts',22,true);text(1031,226,111,48,'(in native\ncoordinates)',22,false,'center','Times New Roman',true);
taper(1178,153,119,141);text(1184,188,107,29,'Reconstruct',22,true);text(1186,214,102,49,'to physical\nspace',22);
rect(1324,141,152,149,fills.green,green,7);text(1334,155,132,31,'T2-C',28,true);text(1330,188,140,49,'physical-space\nfusion',23,true);text(1329,240,142,34,'α(μ̃, π̄, dₙ)',26,false,'center','Cambria Math',true);
rect(1505,135,148,164,fills.red,red,7);text(1513,142,132,49,'Predicted\nphysical fields',22,true);text(1511,199,136,31,'û(t), p̂°(t)',26,false,'center','Cambria Math',true);crop(1513,235,132,56);
rect(24,422,362,71,fills.blue,blue,7);text(38,430,334,30,'Physical history ℋₙ',25,true);text(33,461,344,27,'(initialization for selected specialist(s))',21,false,'center','Times New Roman',true);
rect(1294,417,206,77,fills.blue,blue,7);text(1301,428,191,28,'Chart-independent',23,true);text(1301,455,191,29,'descriptor dₙ',24,true);
line(161,172,202,172,true);line(362,156,431,156,true);line(704,156,753,156,true);
line(459,233,470,198,true,'#929AA1',1.5);line(550,233,557,198,true,'#929AA1',1.5);line(664,233,659,198,true,'#929AA1',1.5);
line(994,221,1023,221,true);line(1150,221,1178,221,true);line(1297,221,1324,221,true);line(1476,221,1505,221,true);
line(386,456,1294,456,true);line(541,456,541,412,true);line(1397,417,1397,290,true);

// Panel b, preserving the supplied diagram, without scientific reinterpretation.
rect(22,691,118,90,fills.blue,blue,6);text(28,709,106,53,'Physical\nhistory ℋₙ',22,true);
rect(167,683,150,112,fills.blue,blue,6);text(175,698,134,51,'Chart-local\nPOD encoding',22,true);text(174,749,136,34,'(Φᵤ⁽ʳ⁾, Φₚ⁽ʳ⁾)',23,false,'center','Cambria Math',true);
rect(344,676,146,119,fills.blue,blue,6);text(350,681,134,48,'Local reduced\nhistory',22,true);text(352,730,130,31,'(aₙ, bₙ)',27,false,'center','Cambria Math',true);text(353,762,128,29,'(+ context)',22);
rect(678,581,310,56,fills.blue,blue,7);text(687,586,292,27,'Projected Galerkin backbone Fᵣᴳ',20,true);text(690,610,287,25,'(from governing equations)',21,false,'center','Times New Roman',true);
rect(569,652,467,230,fills.orange,orange,10);
text(610,658,338,33,'Velocity sparse-MoE correction',23,true);rho(949,659,'u');
rect(579,707,143,117,fills.purple,purple,7);
text(585,714,131,47,'Hierarchical\nsparse routing',21,true);
text(587,762,131,26,'• group Top-1',21,false,'left');text(587,790,131,26,'• channel Top-2',20,false,'left');
rect(751,695,177,59,fills.red,red,7);text(758,700,163,28,'Shared expert',22,true);text(768,728,143,25,'E₀',26,false,'center','Cambria Math',true);
rect(751,766,177,62,fills.red,red,7);text(757,771,165,27,'Top-2 routed experts',18,true);text(763,798,153,28,'Eₑ₁, Eₑ₂',26,false,'center','Cambria Math',true);
rect(963,729,62,61,fills.red,red,6);text(968,742,52,38,'Σ ω',29,false,'center','Cambria Math');
text(591,845,428,29,'nonlinear + linear + low-rank quadratic responses',20,false,'center','Times New Roman',true);
const sum=s.shapes.add({geometry:'ellipse',position:{left:1047,top:651,width:38,height:38},fill:'#FFFFFF',line:{fill:'#111111',width:2}});sum.text='+';sum.text.style={typeface:'Times New Roman',fontSize:30,wrap:'none',alignment:'center',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};
rect(1118,674,156,121,fills.gray,gray,6);text(1123,682,146,47,'Chart-specific\nadvancement',21,true);text(1143,729,106,32,'Stepᵣ',29,false,'center','Cambria Math',true);text(1133,763,126,28,'(e.g., RK)',21);
rect(1363,654,210,159,fills.blue,blue,8);text(1373,665,190,49,'Algebraic pressure\nreconstruction',22,true);text(1374,716,188,47,'(Pressure–Poisson +\nlearned correction)',21);text(1405,769,66,34,'Qᵣᴾᴾ',28,false,'center','Cambria Math',true);text(1470,769,27,34,'+',28,false,'center','Cambria Math');rho(1499,769,'p');
text(1093,825,201,34,'Velocity-state update',23,true);text(1350,825,239,34,'Pressure reconstruction',23,true);
text(1280,700,51,34,'aₙ₊₁',26,false,'center','Cambria Math',true);text(1588,700,66,34,'bₙ₊₁',26,false,'center','Cambria Math',true);
line(1326,620,1326,870,false,'#97A3A6',1.2,true);
line(140,737,167,737,true);line(317,737,344,737,true);
route([[490,724],[547,724],[547,609],[678,609]]);
route([[490,724],[520,724],[520,761],[579,761]]);
route([[988,609],[1066,609],[1066,651]]);
route([[722,741],[730,741],[730,724],[751,724]]);
route([[722,780],[730,780],[730,798],[751,798]]);
route([[928,724],[945,724],[945,749],[963,749]]);
route([[928,798],[945,798],[945,773],[963,773]]);
route([[1025,760],[1066,760],[1066,689]]);
route([[1085,670],[1092,670],[1092,737],[1118,737]]);
line(1274,739,1363,739,true);line(1573,739,1653,739,true);
s.speakerNotes.textFrame.setText('Source: '+source+'. Faithful editable reconstruction of the user-provided figure. Text, panels, module outlines and arrow segments are native PowerPoint objects. Six flow thumbnails remain raster images with native, reversible cropping. Mathematical labels are editable text using Times New Roman and Cambria Math; exact typesetting may differ slightly from the source raster. All wording and connectivity follow the supplied image, not a new method revision.');
const candidate=root+'/build/latest-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
let img=await p.export({slide:s,format:'png',scale:1.25});await fs.writeFile(root+'/build/latest-preview.png',new Uint8Array(await img.arrayBuffer()));
if(process.argv.includes('--final')){
 const finalPath=root+'/output/Figure1_latest_image_editable.pptx';
 const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${1672*9525},${941*9525}`,'--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Times New Roman','Cambria Math']},verifyArtifactToolImport:true,receiptPath:root+'/build/latest-validation.json'});
 console.log(JSON.stringify({path:result.finalPath,integrity:result.packageIntegrity.status,layout:result.presentationLayout}));
 const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));img=await q.export({slide:q.slides.items[0],format:'png',scale:2});await fs.writeFile(root+'/output/Figure1_latest_preview.png',new Uint8Array(await img.arrayBuffer()));
}
