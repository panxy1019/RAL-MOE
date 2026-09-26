import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/panxy1019/Documents/CHANNEL/output/flowchart_refined_v4_final.pptx'));
await fs.writeFile('C:/Users/panxy1019/Documents/CHANNEL/output/.beautify_build/inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,image',maxChars:100000})).ndjson);
console.log(p.slides.items[0].shapes.items.map(s=>({id:s.id,name:s.name,geometry:s.geometry,pos:s.position,fill:s.fill,line:s.line,text:String(s.text)})));
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile('C:/Users/panxy1019/Documents/CHANNEL/output/.beautify_build/before.png',new Uint8Array(await b.arrayBuffer()));
