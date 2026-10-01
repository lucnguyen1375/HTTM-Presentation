import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const deckPath = "D:/PTIT/Tai_Lieu_Hoc_Tap/Project/HTTM/presentation/output/rnn_preprocessing_gold_price_slides_v2.pptx";
const outDir = "D:/PTIT/Tai_Lieu_Hoc_Tap/Project/HTTM/presentation/.slide-build/previews_v2";
await fs.mkdir(outDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(deckPath));
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(outDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
for (let i = 0; i < presentation.slides.items.length; i += 1) {
  const slide = presentation.slides.items[i];
  const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(outDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(outDir);
