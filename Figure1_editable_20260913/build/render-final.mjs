import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/panxy1019/Documents/CHANNEL/Figure1_editable_20260913';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/output/Figure1_Nature_editable_final.pptx'));
const png=await p.export({slide:p.slides.items[0],format:'png',scale:2});
await fs.writeFile(root+'/output/Figure1_preview.png',new Uint8Array(await png.arrayBuffer()));
console.log('Final PPTX reimported and rendered');
