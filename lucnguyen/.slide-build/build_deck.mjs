import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations";
const workspaceDir = "D:/PTIT/Tai_Lieu_Hoc_Tap/Project/HTTM";
const buildDir = path.join(workspaceDir, "presentation", ".slide-build");
const outputDir = path.join(workspaceDir, "presentation", "output");
const finalPath = path.join(outputDir, "rnn_preprocessing_gold_price_slides_v2.pptx");
const notebookPath = path.join(workspaceDir, "presentation", "rnn_sequence_preprocessing_gold_price.ipynb");

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const family = resolvePresentationFont();
const W = 1280;
const H = 720;
const C = {
  bg: "#FBF8F1",
  navy: "#102A43",
  ink: "#17324D",
  muted: "#5C6B73",
  gold: "#C58B2A",
  goldLight: "#F2E5C4",
  blue: "#2D6A8F",
  blueLight: "#DCEAF2",
  green: "#3F7D5A",
  greenLight: "#E4F0E8",
  red: "#B65345",
  redLight: "#F5E2DE",
  line: "#D9DEE5",
  white: "#FFFFFF",
  dark: "#0D1B2A",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function shape(slide, geometry, position, fill = "none", lineFill = "none", lineWidth = 0, name) {
  return slide.shapes.add({
    geometry,
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function text(slide, value, position, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 22,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrink",
  };
  return box;
}

function card(slide, position, fill = C.white, lineFill = C.line, radius = "rounded-2xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: 1 },
    borderRadius: radius,
  });
}

function title(slide, number, heading, subheading = "") {
  shape(slide, "rect", { left: 0, top: 0, width: W, height: 10 }, C.gold);
  text(slide, String(number).padStart(2, "0"), { left: 72, top: 35, width: 52, height: 30 }, {
    fontSize: 16, color: C.gold, bold: true,
  });
  text(slide, heading, { left: 72, top: 65, width: 1050, height: 58 }, {
    fontSize: 34, color: C.navy, bold: true,
  });
  if (subheading) {
    text(slide, subheading, { left: 74, top: 116, width: 1080, height: 30 }, {
      fontSize: 17, color: C.muted,
    });
  }
  text(slide, "Tiền xử lý chuỗi cho RNN · PyTorch", { left: 72, top: 684, width: 500, height: 18 }, {
    fontSize: 11, color: C.muted,
  });
}

function note(slide, body) {
  slide.speakerNotes.textFrame.setText(body);
}

function bullet(slide, label, body, x, y, width, accent = C.gold) {
  shape(slide, "ellipse", { left: x, top: y + 5, width: 10, height: 10 }, accent);
  text(slide, label, { left: x + 22, top: y, width: 145, height: 24 }, { fontSize: 17, color: C.navy, bold: true });
  text(slide, body, { left: x + 170, top: y, width: width - 170, height: 42 }, { fontSize: 16, color: C.ink });
}

function stepBox(slide, number, heading, body, x, y, w, h, fill = C.white) {
  card(slide, { left: x, top: y, width: w, height: h }, fill, C.line, "rounded-xl");
  shape(slide, "ellipse", { left: x + 14, top: y + 14, width: 28, height: 28 }, C.gold);
  text(slide, String(number), { left: x + 14, top: y + 19, width: 28, height: 18 }, { fontSize: 14, color: C.white, bold: true, alignment: "center" });
  text(slide, heading, { left: x + 54, top: y + 13, width: w - 66, height: 22 }, { fontSize: 16, color: C.navy, bold: true });
  text(slide, body, { left: x + 16, top: y + 50, width: w - 32, height: h - 58 }, { fontSize: 14, color: C.muted });
}

function codeBox(slide, code, x, y, w, h, fontSize = 15) {
  card(slide, { left: x, top: y, width: w, height: h }, C.dark, C.dark, "rounded-xl");
  text(slide, code, { left: x + 18, top: y + 16, width: w - 36, height: h - 28 }, {
    fontSize, color: "#F7F4EC", autoFit: "shrink",
  });
}

function addNotebookImage(slide, bytes, position, alt) {
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
    geometry: "roundRect",
    borderRadius: "rounded-xl",
  });
}

// Extract exact rendered figures from the executed notebook.
const notebook = JSON.parse(await fs.readFile(notebookPath, "utf8"));
function findNotebookImage(marker) {
  for (const cell of notebook.cells) {
    const source = (cell.source ?? []).join("");
    if (!source.includes(marker)) continue;
    for (const output of cell.outputs ?? []) {
      const base64 = output.data?.["image/png"];
      if (base64) return Buffer.from(base64, "base64");
    }
  }
  throw new Error(`Notebook image not found for marker: ${marker}`);
}
const lossImage = findNotebookImage("# Vẽ đường loss");
const predictionImage = findNotebookImage("test_dates = df.iloc");
const residualImage = findNotebookImage("# Phân tích phần dư của RNN");

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  shape(slide, "rect", { left: 0, top: 0, width: 18, height: H }, C.gold);
  text(slide, "PHẦN 2", { left: 88, top: 76, width: 150, height: 28 }, { fontSize: 18, color: C.gold, bold: true });
  text(slide, "Tiền xử lý dữ liệu chuỗi\ncho RNN", { left: 88, top: 125, width: 650, height: 150 }, { fontSize: 48, color: C.navy, bold: true });
  text(slide, "Từ token văn bản đến tensor giá vàng", { left: 92, top: 300, width: 560, height: 34 }, { fontSize: 23, color: C.muted });
  card(slide, { left: 88, top: 410, width: 510, height: 108 }, C.goldLight, C.gold);
  text(slide, "Thông điệp chính", { left: 112, top: 430, width: 180, height: 24 }, { fontSize: 16, color: C.gold, bold: true });
  text(slide, "Văn bản cần tokenizer và embedding.\nGiá vàng đã là dữ liệu số liên tục.", { left: 112, top: 459, width: 430, height: 46 }, { fontSize: 19, color: C.navy, bold: true });
  text(slide, "Notebook chạy được: rnn_sequence_preprocessing_gold_price.ipynb", { left: 88, top: 660, width: 560, height: 24 }, { fontSize: 13, color: C.muted });

  card(slide, { left: 770, top: 100, width: 390, height: 470 }, C.white, C.line);
  text(slide, "Dữ liệu thô trở thành tensor như thế nào?", { left: 800, top: 135, width: 320, height: 64 }, { fontSize: 24, color: C.navy, bold: true, alignment: "center" });
  const series = ["priceₜ₋₂", "priceₜ₋₁", "priceₜ", "yₜ₊₁"];
  series.forEach((label, i) => {
    const x = 805 + i * 78;
    card(slide, { left: x, top: 280, width: 68, height: 68 }, i === 3 ? C.redLight : C.blueLight, i === 3 ? C.red : C.blue, "rounded-xl");
    text(slide, label, { left: x + 3, top: 305, width: 62, height: 24 }, { fontSize: 13, color: i === 3 ? C.red : C.blue, bold: true, alignment: "center" });
    if (i < series.length - 1) shape(slide, "rect", { left: x + 69, top: 312, width: 9, height: 2 }, C.gold);
  });
  text(slide, "chuỗi đầu vào", { left: 835, top: 378, width: 230, height: 22 }, { fontSize: 16, color: C.muted, alignment: "center" });
  text(slide, "RNN", { left: 924, top: 425, width: 80, height: 38 }, { fontSize: 30, color: C.navy, bold: true, alignment: "center" });
  text(slide, "dự đoán bước kế tiếp", { left: 842, top: 480, width: 245, height: 28 }, { fontSize: 17, color: C.muted, alignment: "center" });
  note(slide, "Mở đầu: Không phải mọi dữ liệu chuỗi đều được xử lý giống nhau. Phần này phân biệt pipeline NLP với pipeline giá vàng. Nguồn: notebook rnn_sequence_preprocessing_gold_price.ipynb; dataset https://www.kaggle.com/datasets/lbronchal/gold-and-silver-prices-dataset");
}

// Slide 2
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 2, "Hai loại dữ liệu chuỗi, hai pipeline", "Tokenizer phù hợp với token rời rạc; giá vàng cần scaling và windowing.");
  card(slide, { left: 72, top: 165, width: 522, height: 430 }, C.white, C.line);
  card(slide, { left: 686, top: 165, width: 522, height: 430 }, C.white, C.line);
  text(slide, "Chuỗi văn bản", { left: 102, top: 192, width: 300, height: 36 }, { fontSize: 26, color: C.blue, bold: true });
  text(slide, "Chuỗi giá vàng", { left: 716, top: 192, width: 300, height: 36 }, { fontSize: 26, color: C.gold, bold: true });
  const left = ["Tokenizer", "Vocabulary", "Integer encoding", "Padding", "Embedding"];
  const right = ["Làm sạch date/price", "Chia theo thời gian", "log1p + scaling", "Cửa sổ 30 bước", "Tensor (N, L, F)"];
  left.forEach((item, i) => {
    const y = 260 + i * 57;
    bullet(slide, item, i === 0 ? "tách câu thành token" : i === 1 ? "token → ID" : i === 2 ? "ID nguyên cho mô hình" : i === 3 ? "cùng độ dài trong batch" : "ID → vector thực", 104, y, 450, C.blue);
  });
  right.forEach((item, i) => {
    const y = 260 + i * 57;
    bullet(slide, item, i === 0 ? "date là chỉ mục, price là số" : i === 1 ? "giữ đúng thứ tự quá khứ → tương lai" : i === 2 ? "đưa giá về miền dễ học" : i === 3 ? "30 quan sát → dự đoán quan sát kế" : "(samples, timesteps, features)", 718, y, 450, C.gold);
  });
  card(slide, { left: 182, top: 615, width: 916, height: 42 }, C.redLight, C.red, "rounded-xl");
  text(slide, "date không phải token · price không cần vocabulary · fixed window nên thường không cần padding", { left: 205, top: 625, width: 870, height: 20 }, { fontSize: 16, color: C.red, bold: true, alignment: "center" });
  note(slide, "Nhấn mạnh ranh giới khái niệm. Chuỗi văn bản bắt đầu từ string và cần ánh xạ token sang số. Dataset giá vàng đã có giá trị số, nên không được áp dụng embedding theo cách của NLP.");
}

// Slide 3
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 3, "Tokenizer, vocabulary, padding và embedding", "Một câu văn đi qua bốn bước trước khi đến RNN.");
  const steps = [
    ["01", "Tokenizer", "\"giá vàng tăng\"\n→ [giá, vàng, tăng]", C.blueLight],
    ["02", "Vocabulary", "giá → 2\nvàng → 3\ntăng → 4", C.goldLight],
    ["03", "Padding", "[2, 3, 4]\n→ [2, 3, 4, 0, 0, 0]", C.greenLight],
    ["04", "Embedding", "ID 3\n→ vector 8 chiều", C.redLight],
  ];
  steps.forEach((item, i) => {
    const x = 72 + i * 286;
    card(slide, { left: x, top: 190, width: 250, height: 255 }, item[3], C.line, "rounded-xl");
    text(slide, item[0], { left: x + 20, top: 214, width: 48, height: 24 }, { fontSize: 15, color: C.gold, bold: true });
    text(slide, item[1], { left: x + 20, top: 254, width: 200, height: 32 }, { fontSize: 23, color: C.navy, bold: true });
    text(slide, item[2], { left: x + 20, top: 325, width: 205, height: 80 }, { fontSize: 18, color: C.ink, bold: true });
    if (i < steps.length - 1) shape(slide, "rect", { left: x + 252, top: 315, width: 32, height: 3 }, C.gold);
  });
  card(slide, { left: 192, top: 500, width: 896, height: 105 }, C.navy, C.navy, "rounded-xl");
  text(slide, "Embedding matrix", { left: 225, top: 523, width: 250, height: 25 }, { fontSize: 18, color: C.goldLight, bold: true });
  text(slide, "E ∈ R^(V × d)     |     token_id i → E[i]     |     output: (batch, length, d)", { left: 225, top: 560, width: 820, height: 27 }, { fontSize: 21, color: C.white, bold: true });
  text(slide, "Notebook output: token_ids (4, 6)  ·  embedding (4, 6, 8)", { left: 72, top: 635, width: 700, height: 24 }, { fontSize: 16, color: C.muted });
  note(slide, "Giải thích padding bằng số 0 và liên hệ với mask. Embedding không biến giá vàng thành vector; nó dành cho các ID nguyên của token. Nguồn tham khảo: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html");
}

// Slide 4
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 4, "Pipeline tiền xử lý giá vàng", "Dataset gold_price.csv: từ CSV đến cửa sổ 30 bước.");
  const metrics = [
    ["13,461", "dòng thô", C.blueLight, C.blue],
    ["13,320", "dòng sau làm sạch", C.greenLight, C.green],
    ["30", "bước nhìn lại", C.goldLight, C.gold],
  ];
  metrics.forEach((item, i) => {
    const x = 72 + i * 250;
    card(slide, { left: x, top: 155, width: 218, height: 72 }, item[2], item[3], "rounded-xl");
    text(slide, item[0], { left: x + 18, top: 168, width: 170, height: 28 }, { fontSize: 21, color: item[3], bold: true });
    text(slide, item[1], { left: x + 18, top: 198, width: 170, height: 20 }, { fontSize: 14, color: C.ink });
  });
  const pipeline = [
    [1, "Đọc CSV", "date, price"], [2, "Chuyển kiểu", "datetime, float"], [3, "Làm sạch", "dropna, deduplicate"], [4, "Sắp xếp", "chronological"],
    [5, "Chia tập", "70 / 15 / 15"], [6, "Biến đổi", "log1p(price)"], [7, "Scaling", "fit trên train"], [8, "Windowing", "30 → 1 bước"],
  ];
  pipeline.forEach((item, i) => {
    const row = Math.floor(i / 4);
    const col = i % 4;
    stepBox(slide, item[0], item[1], item[2], 72 + col * 286, 270 + row * 125, 250, 100, row === 0 ? C.white : C.blueLight);
  });
  card(slide, { left: 72, top: 540, width: 1136, height: 92 }, C.redLight, C.red, "rounded-xl");
  text(slide, "Min-Max scaling", { left: 102, top: 558, width: 190, height: 24 }, { fontSize: 17, color: C.red, bold: true });
  text(slide, "x_scaled = (x − min_train) / (max_train − min_train)", { left: 300, top: 556, width: 590, height: 28 }, { fontSize: 22, color: C.navy, bold: true });
  text(slide, "Scaler chỉ học từ train; fit trên toàn bộ dữ liệu sẽ gây data leakage.", { left: 300, top: 593, width: 790, height: 22 }, { fontSize: 16, color: C.red });
  note(slide, "Đi chậm ở bước chia tập và scaling. Đây là phần dễ bị hỏi nhất: vì sao không fit scaler trên toàn bộ dataset? Vì max/min của tương lai sẽ làm mô hình biết trước phân phối test. Dataset source: https://www.kaggle.com/datasets/lbronchal/gold-and-silver-prices-dataset");
}

// Slide 5
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 5, "Code minh họa: tokenizer và embedding", "Snippet ngắn trên slide; code đầy đủ nằm trong notebook.");
  const code = "tokens = text.lower().split()\n\nvocab = {'<PAD>': 0, '<UNK>': 1}\n\nencoded = [\n    vocab.get(token, vocab['<UNK>'])\n    for token in tokens\n]\n\npadded = encoded[:max_len] + [0] * (\n    max_len - len(encoded)\n)\n\nembedding = nn.Embedding(\n    len(vocab), embedding_dim=8,\n    padding_idx=0\n)";
  codeBox(slide, code, 72, 175, 650, 420, 17);
  card(slide, { left: 770, top: 175, width: 438, height: 420 }, C.white, C.line);
  text(slide, "Kết quả shape", { left: 810, top: 210, width: 340, height: 32 }, { fontSize: 25, color: C.navy, bold: true });
  text(slide, "token_ids", { left: 810, top: 290, width: 160, height: 28 }, { fontSize: 19, color: C.blue, bold: true });
  text(slide, "(4, 6)", { left: 1000, top: 287, width: 150, height: 34 }, { fontSize: 28, color: C.navy, bold: true, alignment: "right" });
  shape(slide, "rect", { left: 810, top: 340, width: 330, height: 1 }, C.line);
  text(slide, "embedding", { left: 810, top: 375, width: 160, height: 28 }, { fontSize: 19, color: C.gold, bold: true });
  text(slide, "(4, 6, 8)", { left: 980, top: 372, width: 170, height: 34 }, { fontSize: 28, color: C.navy, bold: true, alignment: "right" });
  card(slide, { left: 810, top: 470, width: 330, height: 82 }, C.goldLight, C.gold, "rounded-xl");
  text(slide, "Trong NLP, input là ID rời rạc nên cần embedding.", { left: 830, top: 492, width: 290, height: 44 }, { fontSize: 16, color: C.navy, bold: true, alignment: "center" });
  note(slide, "Đây là slide code duy nhất dành riêng cho NLP. Không giải thích từng dòng quá lâu; chỉ chỉ ra luồng string → ID → padding → vector. Full code đã chạy trong notebook.");
}

// Slide 6
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 6, "Code minh họa: chuỗi giá vàng", "Giá vàng không qua tokenizer; giá được làm sạch, biến đổi và cắt cửa sổ.");
  const code = "df['date'] = pd.to_datetime(df['date'])\ndf['price'] = pd.to_numeric(df['price'])\n\ndf = (\n    df.dropna(subset=['date', 'price'])\n      .drop_duplicates('date')\n      .sort_values('date')\n)\n\nraw_prices = df['price'].to_numpy().reshape(-1, 1)\nvalues = np.log1p(raw_prices)\nscaler.fit(values[:train_end])\nscaled_values = scaler.transform(values)";
  codeBox(slide, code, 72, 172, 690, 430, 16);
  card(slide, { left: 810, top: 172, width: 398, height: 430 }, C.white, C.line);
  text(slide, "Tensor đưa vào RNN", { left: 842, top: 210, width: 330, height: 32 }, { fontSize: 24, color: C.navy, bold: true });
  text(slide, "X_train", { left: 842, top: 285, width: 150, height: 30 }, { fontSize: 22, color: C.blue, bold: true });
  text(slide, "(9294, 30, 1)", { left: 842, top: 326, width: 320, height: 42 }, { fontSize: 32, color: C.navy, bold: true });
  text(slide, "samples · timesteps · features", { left: 842, top: 375, width: 320, height: 22 }, { fontSize: 15, color: C.muted });
  shape(slide, "rect", { left: 842, top: 425, width: 316, height: 2 }, C.gold);
  text(slide, "y_train", { left: 842, top: 455, width: 150, height: 28 }, { fontSize: 22, color: C.gold, bold: true });
  text(slide, "(9294, 1)", { left: 842, top: 495, width: 320, height: 38 }, { fontSize: 30, color: C.navy, bold: true });
  text(slide, "30 quan sát quá khứ → giá bước kế tiếp", { left: 842, top: 555, width: 320, height: 25 }, { fontSize: 15, color: C.muted });
  note(slide, "Giải thích shape bằng một mẫu cụ thể: X là 30 giá liên tiếp, y là giá tiếp theo. Nhấn mạnh split theo thời gian trước khi scaler fit. Full implementation nằm trong cell windowing của notebook.");
}

// Slide 7
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 7, "Simple RNN: input, training và loss", "Chỉ giữ phần RNN cần để hiểu tensor đi vào đâu và kết quả được tạo ra thế nào.");
  const code = "class SimpleRNNRegressor(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.rnn = nn.RNN(\n            input_size=1,\n            hidden_size=32,\n            batch_first=True\n        )\n        self.output_layer = nn.Linear(32, 1)\n\n    def forward(self, x):\n        output, _ = self.rnn(x)\n        return self.output_layer(output[:, -1, :])";
  codeBox(slide, code, 72, 168, 575, 300, 15);
  text(slide, "Luồng dữ liệu", { left: 710, top: 172, width: 300, height: 28 }, { fontSize: 22, color: C.navy, bold: true });
  const flow = [
    ["(B, 30, 1)", C.blueLight, C.blue],
    ["RNN\nhidden=32", C.goldLight, C.gold],
    ["last hidden", C.greenLight, C.green],
    ["Linear\n32 → 1", C.redLight, C.red],
  ];
  flow.forEach((item, i) => {
    const x = 710 + i * 126;
    card(slide, { left: x, top: 235, width: 108, height: 82 }, item[1], item[2], "rounded-xl");
    text(slide, item[0], { left: x + 5, top: 257, width: 98, height: 40 }, { fontSize: 14, color: item[2], bold: true, alignment: "center" });
    if (i < flow.length - 1) shape(slide, "rect", { left: x + 109, top: 275, width: 17, height: 2 }, C.gold);
  });
  card(slide, { left: 710, top: 355, width: 470, height: 78 }, C.navy, C.navy, "rounded-xl");
  text(slide, "prediction → loss → backward() → optimizer.step()", { left: 735, top: 382, width: 420, height: 28 }, { fontSize: 19, color: C.white, bold: true, alignment: "center" });
  addNotebookImage(slide, lossImage, { left: 710, top: 470, width: 470, height: 160 }, "Training and validation loss from the executed notebook");
  note(slide, "Không đi sâu vào W, U, V hoặc vanishing gradient vì đó là phần kiến trúc/training chung. Ở đây chỉ cần nối shape input với lớp RNN, output layer và quy trình loss.backward(). Nguồn API: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html");
}

// Slide 8
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, 8, "Kết quả và phân tích", "RNN chưa vượt baseline: nguyên nhân nằm ở phân phối dữ liệu, không phải lỗi chạy code.");
  const table = slide.tables.add({
    rows: 3,
    columns: 5,
    left: 72,
    top: 160,
    width: 590,
    height: 175,
    values: [
      ["Mô hình", "MAE", "RMSE", "MAPE", "Direction"],
      ["Persistence", "9.2163", "13.4675", "0.6714%", "77.3273%"],
      ["Simple RNN", "236.1134", "252.1612", "16.9713%", "49.1992%"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 5; c += 1) {
      const cell = table.getCell(r, c);
      cell.text.style = { typeface: family, fontSize: r === 0 ? 14 : 15, color: r === 0 ? C.white : C.ink, bold: r === 0 || c === 0, alignment: c === 0 ? "left" : "center" };
      if (r === 0) cell.fill = C.navy;
      if (r === 2 && c > 0) cell.fill = C.redLight;
    }
  }
  text(slide, "Bảng metrics trên tập test", { left: 72, top: 135, width: 330, height: 22 }, { fontSize: 17, color: C.navy, bold: true });
  card(slide, { left: 704, top: 145, width: 504, height: 205 }, C.redLight, C.red, "rounded-xl");
  text(slide, "Distribution shift", { left: 735, top: 175, width: 300, height: 30 }, { fontSize: 24, color: C.red, bold: true });
  text(slide, "Train max: 850.00\nTest min: 1,049.40\n100% giá test > train max", { left: 735, top: 220, width: 410, height: 88 }, { fontSize: 21, color: C.navy, bold: true });
  addNotebookImage(slide, predictionImage, { left: 72, top: 365, width: 590, height: 220 }, "Actual versus prediction plot from the executed notebook");
  addNotebookImage(slide, residualImage, { left: 704, top: 365, width: 504, height: 220 }, "Residual analysis plot from the executed notebook");
  card(slide, { left: 72, top: 615, width: 1136, height: 45 }, C.goldLight, C.gold, "rounded-xl");
  text(slide, "Kết luận: preprocessing phải đi cùng phân tích phân phối; hướng cải thiện là log-return, cửa sổ gần đây hoặc GRU/LSTM.", { left: 95, top: 627, width: 1090, height: 22 }, { fontSize: 16, color: C.navy, bold: true, alignment: "center" });
  note(slide, "Đọc bảng trước, sau đó chỉ vào biểu đồ actual-prediction và residual. Kết quả được lấy trực tiếp từ notebook đã chạy. Không nói RNN tốt hơn baseline. Đây là ví dụ cho thấy distribution shift làm Simple RNN khó ngoại suy mức giá mới. Notebook: rnn_sequence_preprocessing_gold_price.ipynb");
}

const candidatePath = path.join(buildDir, "candidate_rnn_preprocessing_gold_price_v2.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [8],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [family] },
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "8",
  ],
  requiredNativeTableOwnerSlides: [8],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "rnn_preprocessing_gold_price_slides_v2.validation.json"),
});

console.log(JSON.stringify({ finalPath, candidatePath, validation: result }, null, 2));
