import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/panxy1019/Documents/CHANNEL/output/flowchart_corrected/flowchart_refined_v5_corrected.pptx'));
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile('C:/Users/panxy1019/Documents/CHANNEL/output/.flowchart_v5_build/preview-final.png',new Uint8Array(await b.arrayBuffer()));
