const input = document.querySelector("#file-input");
const image = document.querySelector("#result-image");
const empty = document.querySelector("#empty-state");
const busy = document.querySelector("#busy");
const list = document.querySelector("#detections");
const summary = document.querySelector("#summary");
const count = document.querySelector("#count");
const timing = document.querySelector("#timing");
const error = document.querySelector("#error");
let lastResult;
let cameraStream;
const cameraPreview = document.querySelector("#camera-preview");

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    document.querySelector("#health").textContent = data.model_loaded ? "模型已就绪" : "等待模型";
  } catch {
    document.querySelector("#health").textContent = "服务未连接";
  }
}

function clearResult() {
  input.value = "";
  image.hidden = true;
  empty.hidden = false;
  list.replaceChildren();
  count.textContent = "0";
  summary.textContent = "等待图片";
  timing.hidden = true;
  error.hidden = true;
  lastResult = undefined;
  for (const selector of ["#download-image", "#download-json", "#download-csv"]) {
    document.querySelector(selector).hidden = true;
  }
}

input.addEventListener("change", async () => {
  const file = input.files?.[0];
  if (!file) return;
  stopCamera();
  await detectFile(file);
});

async function detectFile(file) {
  clearResult();
  busy.hidden = false;
  summary.textContent = file.name;
  try {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "检测失败");
    const resultImageUrl = `/api/results/${result.result_id}.jpg`;
    image.src = resultImageUrl;
    image.hidden = false;
    empty.hidden = true;
    lastResult = result;
    const imageDownload = document.querySelector("#download-image");
    imageDownload.href = resultImageUrl;
    imageDownload.hidden = false;
    document.querySelector("#download-json").hidden = false;
    document.querySelector("#download-csv").hidden = false;
    count.textContent = String(result.detections.length);
    summary.textContent = result.detections.length ? "检测完成" : "未检出目标";
    for (const detection of result.detections) {
      const item = document.createElement("li");
      item.className = "detection";
      const label = document.createElement("div");
      const name = document.createElement("div");
      name.className = "detection-name";
      name.textContent = detection.class_name;
      const position = document.createElement("div");
      position.className = "detection-meta";
      position.textContent = `框 ${detection.xyxy.map(Math.round).join(", ")}`;
      label.append(name, position);
      const confidence = document.createElement("span");
      confidence.className = "confidence";
      confidence.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
      item.append(label, confidence);
      list.append(item);
    }
    timing.querySelector("strong").textContent = `${result.inference_ms.toFixed(1)} ms`;
    timing.hidden = false;
  } catch (exception) {
    error.textContent = exception.message;
    error.hidden = false;
    summary.textContent = "本次检测未完成";
  } finally {
    busy.hidden = true;
  }
}

function stopCamera() {
  cameraStream?.getTracks().forEach(track => track.stop());
  cameraStream = undefined;
  cameraPreview.srcObject = null;
  cameraPreview.hidden = true;
  document.querySelector("#camera-start").hidden = false;
  document.querySelector("#camera-capture").hidden = true;
  document.querySelector("#camera-stop").hidden = true;
  if (image.hidden) empty.hidden = false;
}

document.querySelector("#camera-start").addEventListener("click", async () => {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器无法访问相机，请在本机 localhost 打开页面");
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}, audio: false});
    cameraPreview.srcObject = cameraStream;
    cameraPreview.hidden = false;
    image.hidden = true;
    empty.hidden = true;
    error.hidden = true;
    document.querySelector("#camera-start").hidden = true;
    document.querySelector("#camera-capture").hidden = false;
    document.querySelector("#camera-stop").hidden = false;
  } catch (exception) {
    error.textContent = exception.message;
    error.hidden = false;
  }
});

document.querySelector("#camera-capture").addEventListener("click", () => {
  if (!cameraPreview.videoWidth || !cameraPreview.videoHeight) return;
  const canvas = document.createElement("canvas");
  canvas.width = cameraPreview.videoWidth;
  canvas.height = cameraPreview.videoHeight;
  canvas.getContext("2d").drawImage(cameraPreview, 0, 0);
  canvas.toBlob(async blob => {
    if (!blob) return;
    stopCamera();
    await detectFile(new File([blob], "camera.jpg", {type: "image/jpeg"}));
  }, "image/jpeg", 0.92);
});

document.querySelector("#camera-stop").addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);

document.querySelector("#clear-button").addEventListener("click", () => { stopCamera(); clearResult(); });
document.querySelector("#download-json").addEventListener("click", () => {
  if (lastResult) downloadBlob(JSON.stringify(lastResult, null, 2), "cotton-detections.json", "application/json");
});
document.querySelector("#download-csv").addEventListener("click", () => {
  if (!lastResult) return;
  const lines = ["class_id,class_name,confidence,x1,y1,x2,y2"];
  for (const item of lastResult.detections) {
    const name = `"${String(item.class_name).replaceAll('"', '""')}"`;
    lines.push([item.class_id, name, item.confidence, ...item.xyxy].join(","));
  }
  downloadBlob(lines.join("\n") + "\n", "cotton-detections.csv", "text/csv;charset=utf-8");
});

function downloadBlob(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

checkHealth();
