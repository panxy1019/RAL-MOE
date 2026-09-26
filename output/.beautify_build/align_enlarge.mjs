import fs from 'node:fs/promises';
let code=await fs.readFile(new URL('./fix_connections.mjs',import.meta.url),'utf8');
code=code.replace("root+'/flowchart_refined_v4_final.pptx'","root+'/flowchart_connections_fixed/flowchart_v4_connections_fixed.pptx'");
code=code.replace("[[497,246],[620,246],[620,243],[661,243]]","[[497,243],[661,243]]");
code=code.replace('return sp;});',`// Enlarge ordinary text without touching Office Math or wording.
const textIds=['62','65','74','81','82','83','84','86','90','93','96','103','107','116','122','124','129','134','136','138','142','146','149','152','153','2002','2003'];
if(textIds.includes(id)){
 sp=sp.replace(/sz="1725"/g,'sz="1875"');
 const widths={'65':196,'107':192,'116':128,'129':92,'136':157,'146':176,'149':200,'152':180,'2003':226,'142':470};
 if(widths[id]){sp=sp.replace(/<a:xfrm[^>]*>[\\s\\S]*?<\\/a:xfrm>/,part=>{
 let x=Number(part.match(/<a:off x="(\\d+)"/)[1]);let w=Number(part.match(/<a:ext cx="(\\d+)"/)[1]);let nw=em(widths[id]);
 return part.replace(/<a:off x="\\d+"/, '<a:off x="'+Math.round(x+(w-nw)/2)+'"').replace(/<a:ext cx="\\d+"/,'<a:ext cx="'+nw+'"');});}
}
if(id==='126')sp=sp.replace(/sz="1800"/g,'sz="1950"');
if(['146','149'].includes(id))sp=sp.replace(/sz="1875"/g,'sz="1800"');
return sp;});`);
code=code.replace("const out=root+'/flowchart_connections_fixed'","const out=root+'/flowchart_aligned'");
code=code.replaceAll('flowchart_v4_connections_fixed.pptx','flowchart_v4_aligned_larger_text.pptx');
// Restore source reference after renaming outputs.
code=code.replace("root+'/flowchart_connections_fixed/flowchart_v4_aligned_larger_text.pptx'","root+'/flowchart_connections_fixed/flowchart_v4_connections_fixed.pptx'");
code=code.replace('connections_candidate.pptx','aligned_candidate.pptx').replace('connections_validation.json','aligned_validation.json');
await fs.writeFile(new URL('./align_generated.mjs',import.meta.url),code);
await import('./align_generated.mjs');
