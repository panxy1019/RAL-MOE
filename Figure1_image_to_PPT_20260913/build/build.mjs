import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='C:/Users/panxy1019/Documents/CHANNEL/Figure1_image_to_PPT_20260913';
const skill='C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const runtime='C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const source='C:/Users/PANXY1~1/AppData/Local/Temp/codex-clipboard-ad037dec-bfb2-40a9-b176-ef78d4632647.png';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const p=Presentation.create({slideSize:{width:1672,height:941}}),s=p.slides.add();s.background.fill='#FFFFFF';
const blue='#0086FF',orange='#FF9029',purple='#9548FF',gray='#7890A5',green='#218C27',red='#FF4949';
let n=0;
function grad(a,b){return {type:'gradient',gradientKind:'linear',angleDeg:25,stops:[{offset:0,color:a},{offset:100000,color:b}]};}
const fills={blue:grad('#EDF8FF','#CAE4F8'),orange:grad('#FFF9EE','#FFE9CB'),purple:grad('#F5EEFF','#E2D4F8'),gray:grad('#EDF1F4','#D9DFE4'),green:grad('#F1FCF0','#D5F1D2'),red:grad('#FFF2EF','#FFDCDC')};
function rect(x,y,w,h,fill='none',stroke='none',radius=5,dash=false){return s.shapes.add({name:`module-${++n}`,geometry:'roundRect',position:{left:x,top:y,width:w,height:h},borderRadius:radius,fill,line:{fill:stroke,width:stroke==='none'?0:1.3,style:dash?'dashed':'solid'}});}
function text(x,y,w,h,t,size=23,bold=false,align='center',font='Arial',italic=false){const q=rect(x,y,w,h);q.text=t;q.text.style={typeface:font,fontSize:size,bold,italic,color:'#080808',alignment:align,verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};q.name=t;return q;}
function point(x,y){return rect(x-.05,y-.05,.1,.1);}
function line(x1,y1,x2,y2,arrow=false,color='#111111',width=2,dash=false){const q=s.shapes.connect(point(x1,y1),point(x2,y2),{kind:'straight',fromSide:'right',toSide:'left',line:{fill:color,width,style:dash?'dashed':'solid'},tail:{type:arrow?'triangle':'none',width:'med',length:'med'}});q.bringToFront();return q;}
function route(points,color='#111111',width=2){for(let i=1;i<points.length;i++)line(...points[i-1],...points[i],i===points.length-1,color,width);}
function taper(x,y,w,h){return s.shapes.add({name:'editable tapered module',geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:fills.blue,line:{fill:blue,width:1.3},customPaths:[{width:w,height:h,commands:[{moveTo:{x:0,y:0}},{lineTo:{x:w,y:h*.17}},{lineTo:{x:w,y:h*.83}},{lineTo:{x:0,y:h}},{close:{}}]}]});}
const bytes=new Uint8Array(await fs.readFile(source));
function crop(x,y,w,h,dx=x,dy=y,dw=w,dh=h){const im=s.images.add({blob:bytes,contentType:'image/png',alt:'Flow illustration retained from user-provided figure',position:{left:dx,top:dy,width:dw,height:dh},fit:'contain'});im.lockAspectRatio=false;im.crop={left:x/1672,top:y/941,right:(1672-x-w)/1672,bottom:(941-y-h)/941};return im;}

// Panel containers, then module surfaces. All native and independently editable.
rect(7,5,1658,506,'#FFFFFF',blue,24);
rect(7,536,1658,389,'#FFFFFF',gray,24);
text(25,13,1070,44,'(a) Two-level RAL-MoE-ROM hierarchy',36,true,'left');
text(25,544,510,44,'(b) One regime-local specialist',35,true,'left');
rect(26,124,126,114,fills.purple,purple);
text(45,142,88,45,'μ',45,true,'center','Cambria Math',true);
text(32,189,114,31,'(parameter)',21);
taper(198,110,172,133);
text(207,150,155,33,'E2 router',24,true);
text(209,181,152,30,'(parameter only)',20);
rect(425,113,276,85,fills.gray,gray,8);
text(437,124,253,31,'Select specialists',24,true);
text(434,155,260,29,'Top-1 or adjacent Top-2',21);
rect(347,237,388,176,'#FFFFFF',gray,16,true);
text(364,240,353,30,'Candidate specialists',24,true);
const cards=[{x:360,name:'Steady',sym:'S',roi:[368,345,101,49]},{x:484,name:'Hopf',sym:'H',roi:[493,346,101,48]},{x:608,name:'Periodic',sym:'P',roi:[617,346,101,48]}];
for(const c of cards){rect(c.x,275,116,126,fills.orange,orange,8);text(c.x+5,281,106,28,c.name,22,true);text(c.x+9,309,98,33,c.sym,29,true,'center','Cambria Math',true);crop(...c.roi);}
rect(753,64,240,338,fills.orange,orange,12);
text(764,72,218,31,'Selected specialist(s)',24,true);
rect(765,108,215,111,fills.orange,orange,7);
text(775,112,194,30,'Specialist r',23,true);
crop(773,146,199,64);
text(848,220,43,32,'⋮',29,true);
rect(765,253,215,136,fills.orange,orange,7);
text(780,259,183,27,'Specialist s',23,true);
text(780,287,182,29,'(if selected)',21);
crop(773,319,199,61);
taper(1017,130,136,217);
text(1026,148,118,48,'Independent\nrollouts',21,true);
text(1025,196,121,43,'(in native\ncoordinates)',20,false,'center','Arial',true);
crop(1026,246,124,91);
taper(1174,146,124,188);
text(1180,165,111,28,'Reconstruct',20,true);
text(1183,192,108,48,'to physical\nspace',21);
crop(1183,247,111,78);
rect(1323,143,149,149,fills.green,green,8);
text(1335,159,126,62,'T2-C\nfusion',27,true);
text(1329,238,138,35,'α(μ̃, π̄, dₙ)',27,false,'center','Cambria Math',true);
rect(1499,139,151,168,fills.red,red,8);
text(1508,146,133,52,'Predicted\nfields',23,true);
text(1507,206,133,31,'û(t), p̂(t)',27,false,'center','Cambria Math',true);
crop(1511,244,132,53);
rect(28,428,267,68,fills.blue,blue,5);
text(38,445,246,36,'Physical history ℋₙ',25,true);
rect(1287,420,229,76,fills.blue,blue,6);
text(1297,433,208,51,'Chart-independent\ndescriptor dₙ',23,true);
// Exact source connectivity is retained, including initialization of the library.
line(152,177,198,177,true);line(370,164,425,164,true);line(702,162,753,162,true);
for(const [x,y]of [[475,237],[558,237],[653,237]])line(x,y,x+(x===475?12:x===653?-6:0),198,true,'#516B80',1.5);
line(994,223,1017,223,true);line(1153,223,1174,223,true);line(1298,223,1323,223,true);line(1473,223,1499,223,true);
line(295,465,1287,465,true);line(542,465,542,413,true);text(419,427,111,31,'initialize',21);line(1398,420,1398,292,true);

// Lower panel surfaces.
rect(29,701,135,87,fills.blue,blue,6);
text(36,716,121,57,'Physical\nhistory ℋₙ',24,true);
rect(192,699,137,90,fills.blue,blue,6);
text(200,714,121,58,'Local POD\nencoding',24,true);
rect(356,693,138,103,fills.blue,blue,6);
text(365,701,120,51,'Reduced\nstate',23,true);
text(363,755,124,35,'(aₙ, bₙ)',28,false,'center','Cambria Math',true);
rect(545,569,579,331,fills.orange,orange,15);
text(609,573,454,33,'Specialist model (e.g., Hopf)',25,true);
rect(659,608,374,49,fills.blue,blue,6);
text(672,619,348,30,'Projected Galerkin backbone Fᵣᴳ',22,true);
rect(586,670,448,207,fills.purple,purple,6);
text(615,676,390,31,'Sparse MoE correction',24,true);
rect(599,725,98,76,fills.purple,purple,7);
text(606,738,84,49,'Group\nrouter',22,true);
rect(734,711,188,58,fills.red,red,7);
text(748,715,158,27,'Shared expert',21,true);
text(752,740,151,27,'E₀',25,false,'center','Cambria Math',true);
rect(734,777,188,60,fills.red,red,7);
text(743,781,170,27,'Top-2 experts',21,true);
text(747,807,161,27,'Eₑ₁, Eₑ₂',25,false,'center','Cambria Math',true);
rect(955,733,64,61,fills.red,red,6);
text(960,746,54,38,'Σ ω',31,false,'center','Cambria Math');
text(618,843,393,28,'nonlinear + linear + low-rank quadratic',20,false,'center','Arial',true);
const plus=s.shapes.add({geometry:'ellipse',name:'sum',position:{left:1060,top:677,width:40,height:40},fill:'#FFFFFF',line:{fill:'#111111',width:2}});plus.text='+';plus.text.style={typeface:'Arial',fontSize:29,alignment:'center',verticalAlignment:'middle',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};
rect(1150,689,128,104,fills.gray,'#687078',6);
text(1164,713,100,31,'RK4',27,true);
text(1156,744,116,27,'(time advance)',18);
rect(1381,688,195,100,fills.blue,blue,6);
text(1390,712,177,53,'Algebraic pressure\nreconstruction',23,true);
text(1163,810,150,53,'advance\nvelocity only',23);
text(1387,810,185,53,'reconstruct\npressure afterwards',22);
text(1287,697,63,34,'aₙ₊₁',28,false,'center','Cambria Math',true);
text(1582,698,68,34,'bₙ₊₁',28,false,'center','Cambria Math',true);
line(1354,623,1354,884,false,'#929292',1.2,true);
line(164,745,192,745,true);line(329,745,356,745,true);
route([[494,744],[562,744],[562,634],[659,634]]);
route([[562,744],[562,772],[586,772]]);
route([[1033,634],[1080,634],[1080,677]]);
line(697,741,734,741,true);route([[697,786],[716,786],[716,795],[734,795]]);
line(922,746,955,746,true);line(922,785,955,785,true);
route([[1019,768],[1080,768],[1080,717]]);
route([[1100,697],[1108,697],[1108,741],[1150,741]]);
line(1278,741,1381,741,true);line(1576,741,1651,741,true);
s.speakerNotes.textFrame.setText('Source: user-provided image '+source+'. The source diagram is reproduced as editable shapes, text and connectors. Flow illustrations are preserved as native, reversible crops of the supplied raster image; their contents are not editable vectors. Diagram wording and connectivity, including RK4 and Group router, are retained from the supplied image. No scientific correction or new simulation result is implied.');
const candidate=root+'/build/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
let img=await p.export({slide:s,format:'png',scale:1.2});await fs.writeFile(root+'/build/preview.png',new Uint8Array(await img.arrayBuffer()));
if(process.argv.includes('--final')){
 const finalPath=root+'/output/Figure1_from_image_editable.pptx';
 const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${1672*9525},${941*9525}`,'--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Arial','Cambria Math']},verifyArtifactToolImport:true,receiptPath:root+'/build/validation.json'});
 console.log(JSON.stringify({path:result.finalPath,integrity:result.packageIntegrity.status,layout:result.presentationLayout}));
 const reread=await PresentationFile.importPptx(await FileBlob.load(finalPath));
 img=await reread.export({slide:reread.slides.items[0],format:'png',scale:2});await fs.writeFile(root+'/output/Figure1_preview.png',new Uint8Array(await img.arrayBuffer()));
}
