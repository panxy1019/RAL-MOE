import fs from 'node:fs/promises';
import JSZip from 'jszip';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation} from 'file:///C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/Users/panxy1019/Documents/CHANNEL/output';
const skill='C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const z=await JSZip.loadAsync(await fs.readFile(root+'/flowchart_connections_fixed/flowchart_v4_connections_fixed.pptx'));
const before=await z.file('ppt/slides/slide1.xml').async('string');
const em=x=>Math.round(x*9525);
function pathShape(sp,pts){const minx=Math.min(...pts.map(p=>p[0])),miny=Math.min(...pts.map(p=>p[1]));const w=Math.max(1,Math.max(...pts.map(p=>p[0]))-minx),h=Math.max(1,Math.max(...pts.map(p=>p[1]))-miny);sp=sp.replace(/<a:xfrm[^>]*>[\s\S]*?<\/a:xfrm>/,`<a:xfrm><a:off x="${em(minx)}" y="${em(miny)}"/><a:ext cx="${em(w)}" cy="${em(h)}"/></a:xfrm>`);sp=sp.replace(/<a:pathLst>[\s\S]*?<\/a:pathLst>/,`<a:pathLst><a:path w="${em(w)}" h="${em(h)}">${pts.map((p,i)=>`<a:${i?'lnTo':'moveTo'}><a:pt x="${em(p[0]-minx)}" y="${em(p[1]-miny)}"/></a:${i?'lnTo':'moveTo'}>`).join('')}</a:path></a:pathLst>`);return sp;}
let after=before.replace(/<p:sp>[\s\S]*?<\/p:sp>/g,sp=>{const id=sp.match(/<p:cNvPr[^>]*\bid="(\d+)"/)?.[1];
if(id==='14')return pathShape(sp,[[497,243],[661,243]]);
if(id==='15')return sp.replace(/<a:off x="\d+" y="\d+"\s*\/>/,`<a:off x="${em(649)}" y="${em(238)}"/>`);
if(id==='18')return pathShape(sp,[[807,242],[860,242],[860,166],[1035,166]]);
if(id==='20')return pathShape(sp,[[807,242],[860,242],[860,316],[1035,316]]);
if(id==='21')return sp.replace(/<a:off x="\d+" y="\d+"\s*\/>/,`<a:off x="${em(1023)}" y="${em(311)}"/>`);
// Enlarge ordinary text without touching Office Math or wording.
const textIds=['62','65','74','81','82','83','84','86','90','93','96','103','107','116','122','124','129','134','136','138','142','146','149','152','153','2002','2003'];
if(textIds.includes(id)){
 sp=sp.replace(/sz="1725"/g,'sz="1875"');
 const widths={'65':196,'107':192,'116':128,'129':92,'136':157,'146':176,'149':200,'152':180,'2003':226,'142':470};
 if(widths[id]){sp=sp.replace(/<a:xfrm[^>]*>[\s\S]*?<\/a:xfrm>/,part=>{
 let x=Number(part.match(/<a:off x="(\d+)"/)[1]);let w=Number(part.match(/<a:ext cx="(\d+)"/)[1]);let nw=em(widths[id]);
 return part.replace(/<a:off x="\d+"/, '<a:off x="'+Math.round(x+(w-nw)/2)+'"').replace(/<a:ext cx="\d+"/,'<a:ext cx="'+nw+'"');});}
}
if(id==='126')sp=sp.replace(/sz="1800"/g,'sz="1950"');
if(['146','149'].includes(id))sp=sp.replace(/sz="1875"/g,'sz="1800"');
return sp;});
const text=x=>x.match(/<(?:a:t|m:t)[^>]*>[\s\S]*?<\/(?:a:t|m:t)>/g);
if(JSON.stringify(text(before))!==JSON.stringify(text(after)))throw Error('Text changed');
z.file('ppt/slides/slide1.xml',after);
const out=root+'/flowchart_aligned';await fs.mkdir(out,{recursive:true});
const candidate=root+'/.beautify_build/aligned_candidate.pptx';await fs.writeFile(candidate,await z.generateAsync({type:'nodebuffer'}));
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:out+'/flowchart_v4_aligned_larger_text.pptx',pythonExecutable:'C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','17345025,8915400'],explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.beautify_build/aligned_validation.json'});
const p=await PresentationFile.importPptx(await FileBlob.load(out+'/flowchart_v4_aligned_larger_text.pptx'));const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});await fs.writeFile(out+'/preview.png',new Uint8Array(await b.arrayBuffer()));console.log('Saved and rendered');
