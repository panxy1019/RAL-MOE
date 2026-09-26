import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import JSZip from 'jszip';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {finalizePresentation} from 'file:///C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/Users/panxy1019/Documents/CHANNEL/output';
const src=root+'/flowchart_refined_v4_final.pptx';
const skill='C:/Users/panxy1019/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const input=await fs.readFile(src),zip=await JSZip.loadAsync(input);
const before=await zip.file('ppt/slides/slide1.xml').async('string');
const map={
'080E36':'23344A','080A16':'1C2B3D','101638':'23344A','00131E':'23344A','071F22':'23344A',
'176CB1':'4A6B8C','638399':'8A9BAD','75B0FF':'C2CDDA',
'E0EEFC':'EDF3F8','D0E4FC':'EDF3F8','078AFF':'7395B8',
'F0F5F9':'F3F6F9','E7EFF5':'F3F6F9','7D9FC0':'93A8BC',
'EEE8FA':'F0EDF6','E5DEFA':'F0EDF6','9360FF':'9784B4','9964FF':'9784B4',
'C9F2F0':'E3F2EF','A9E6E5':'E3F2EF','00A1A8':'57968E','00A3AA':'57968E',
'FFF9E2':'FAF5E7','FFF0BE':'FAF5E7',
'FFF3E7':'FCF0E4','FFE6D0':'FCF0E4','FF7F29':'CC9867','FF8128':'CC9867','FF791D':'CC9867','F78523':'C48A56',
'FFEAF0':'F5ECF0','FFDFE8':'F5ECF0','FF2981':'BA879E',
'BAF1CA':'E3F1E9','9561FF':'9784B4','9C5CFC':'9784B4',
'FFEAE6':'FAEFEB','FFDCD6':'FAEFEB','FFE9E3':'FAEFEB','FF716B':'CA9B88',
'E1E8EE':'EDF0F3','739FC5':'98A8B7','8DB5D3':'98A8B7',
'F25A08':'A56F44','FF8025':'C99B74','FFF5E9':'FFFCF7','FFEEDC':'FFFCF7','FFFEFC':'FFFCF7'};
// Preserve the original OpenXML objects, equation objects, paths and relationships.
// Only style attributes change, avoiding lossy conversion of Office Math.
let after=before.replace(/<a:srgbClr val="([A-Fa-f0-9]+)"/g,(m,c)=>`<a:srgbClr val="${map[c.toUpperCase()]??c}"`);
after=after.replace(/<a:gradFill\b[^>]*>[\s\S]*?<\/a:gradFill>/g,m=>{const c=m.match(/<a:srgbClr val="([^"]+)"/);return c?`<a:solidFill><a:srgbClr val="${c[1]}"/></a:solidFill>`:m;});
after=after.replace(/<p:sp>[^]*?<\/p:sp>/g,sp=>{
 const id=sp.match(/<p:cNvPr[^>]*\bid="(\d+)"/)?.[1];
 if(['154','2'].includes(id))sp=sp.replace(/<a:prstDash val="[^"]+"\s*\/>/g,'<a:prstDash val="solid"/>');
 // Existing text size, font, equation formatting and all coordinates are retained.
 return sp;
});
function tokens(x){return x.match(/<(?:a:t|m:t)[^>]*>[^]*?<\/(?:a:t|m:t)>/g)??[];}
function geometry(x){return x.match(/<a:(?:xfrm|custGeom|prstGeom)\b[^]*?<\/a:(?:xfrm|custGeom|prstGeom)>/g)??[];}
if(JSON.stringify(tokens(before))!==JSON.stringify(tokens(after)))throw Error('Text changed');
if(JSON.stringify(geometry(before))!==JSON.stringify(geometry(after)))throw Error('Geometry changed');
zip.file('ppt/slides/slide1.xml',after);
const candidate=root+'/.beautify_build/candidate.pptx';
await fs.writeFile(candidate,await zip.generateAsync({type:'nodebuffer'}));
const p=await PresentationFile.importPptx(await FileBlob.load(candidate));
await fs.writeFile(root+'/.beautify_build/after-inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,image',maxChars:100000})).ndjson);
await fs.writeFile(root+'/.beautify_build/preservation.json',JSON.stringify({sourceSha256:crypto.createHash('sha256').update(input).digest('hex'),textIdentical:true,geometryIdentical:true,changedParts:['ppt/slides/slide1.xml'],scope:'color palette, flat fills, panel border styles'},null,2));
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:root+'/flowchart_beautified/flowchart_v4_style_only.pptx',pythonExecutable:'C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','17345025,8915400'],explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.beautify_build/validation.json'});
console.log('Validated; rendering');
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile(root+'/flowchart_beautified/flowchart_v4_style_only.png',new Uint8Array(await b.arrayBuffer()));
console.log('Rendered');
