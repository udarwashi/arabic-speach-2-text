'use strict';

/* Single-page client. No framework, no build step.
   The server is the source of truth for status labels and model metadata, so
   nothing here duplicates that text. */

const el = (id) => document.getElementById(id);

const ui = {
  deviceBadge: el('device-badge'),
  logoutBtn: el('logout-btn'),
  dropzone: el('dropzone'),
  fileInput: el('file-input'),
  dzHint: el('dz-hint'),
  fileChip: el('file-chip'),
  fileName: el('file-name'),
  fileSize: el('file-size'),
  fileClear: el('file-clear'),
  modelOptions: el('model-options'),
  languageSelect: el('language-select'),
  taskSelect: el('task-select'),
  startBtn: el('start-btn'),
  cancelBtn: el('cancel-btn'),
  statusCard: el('status-card'),
  statusLabel: el('status-label'),
  statusMeta: el('status-meta'),
  progress: el('progress'),
  progressFill: el('progress-fill'),
  alert: el('alert'),
  alertTitle: el('alert-title'),
  alertBody: el('alert-body'),
  alertDetails: el('alert-details'),
  alertDetailText: el('alert-detail-text'),
  resultCard: el('result-card'),
  tabFlow: el('tab-flow'),
  tabTimed: el('tab-timed'),
  copyBtn: el('copy-btn'),
  dlTxt: el('dl-txt'),
  dlSrt: el('dl-srt'),
  dlVtt: el('dl-vtt'),
  flow: el('transcript-flow'),
  timed: el('transcript-timed'),
  resultMeta: el('result-meta'),
};

const state = {
  file: null,
  jobId: null,
  source: null,        // EventSource
  models: [],
  model: null,         // key of the chosen model
  maxUploadMb: null,
  extensions: [],
  seen: new Set(),     // segment indices already rendered
  texts: [],
  startedAt: 0,
  timer: null,
  finished: false,
};

/* ---------- helpers ---------- */

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

function showAlert(title, body, detail) {
  ui.alertTitle.textContent = title;
  ui.alertBody.textContent = body || '';
  if (detail) {
    ui.alertDetailText.textContent = detail;
    ui.alertDetails.hidden = false;
  } else {
    ui.alertDetails.hidden = true;
  }
  ui.alert.hidden = false;
}

function hideAlert() {
  ui.alert.hidden = true;
}

async function readError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === 'string') return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail.map((d) => d.msg).join('، ');
  } catch (_) { /* not JSON */ }
  return `تعذّر إكمال الطلب (رمز ${response.status}).`;
}

/* ---------- bootstrap ---------- */

async function loadMetadata() {
  const response = await fetch('/api/models');
  if (!response.ok) throw new Error('models endpoint failed');
  const data = await response.json();

  state.models = data.models;
  state.maxUploadMb = data.max_upload_mb;
  state.extensions = data.extensions || [];

  fill(ui.languageSelect, data.languages.map((l) => ({ value: l.key, label: l.label })));
  fill(ui.taskSelect, data.tasks.map((t) => ({ value: t.key, label: t.label })));
  renderModels(data.default_model);

  const pretty = state.extensions
    .map((ext) => ext.replace('.', '').toUpperCase())
    .join('، ');
  ui.dzHint.textContent = `${pretty} — حتى ${state.maxUploadMb} ميجابايت`;
}

function fill(select, options) {
  select.replaceChildren(
    ...options.map(({ value, label }) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      return option;
    })
  );
}

/* The models are radio cards rather than a <select>: the trade-off between the
   three (speed, accuracy, weight) only reads if the notes are visible at once. */
function renderModels(defaultKey) {
  state.model = state.models.some((m) => m.key === defaultKey)
    ? defaultKey
    : (state.models[0] || {}).key || null;

  ui.modelOptions.replaceChildren(
    ...state.models.map((spec) => {
      const card = document.createElement('label');
      card.className = 'model-card';

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'model';
      radio.value = spec.key;
      radio.checked = spec.key === state.model;
      radio.addEventListener('change', () => { state.model = spec.key; });

      const name = document.createElement('span');
      name.className = 'model-name';
      name.textContent = spec.label;

      const note = document.createElement('span');
      note.className = 'model-note';
      note.textContent = spec.note;

      const size = document.createElement('span');
      size.className = 'model-size';
      size.dir = 'ltr';
      size.textContent = spec.download;

      card.append(radio, name, note, size);
      return card;
    })
  );
}

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) return;
    const data = await response.json();
    const runtime = data.runtime || {};
    const onGpu = runtime.device === 'cuda';
    ui.deviceBadge.textContent = onGpu
      ? `يعمل على كرت الرسوميات (GPU)`
      : `يعمل على المعالج (CPU)`;
    ui.deviceBadge.classList.toggle('gpu', onGpu);
    if (runtime.warnings && runtime.warnings.length) {
      ui.deviceBadge.title = runtime.warnings.join('\n');
    }
    // Only worth offering when there is a session to end.
    ui.logoutBtn.hidden = !data.auth;
  } catch (_) {
    ui.deviceBadge.textContent = 'الخادم غير متاح';
  }
}

ui.logoutBtn.addEventListener('click', async () => {
  ui.logoutBtn.disabled = true;
  try {
    await fetch('/api/logout', { method: 'POST' });
  } finally {
    window.location.replace('/login');
  }
});

/* ---------- file selection ---------- */

function acceptFile(file) {
  if (!file) return;
  const dot = file.name.lastIndexOf('.');
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (state.extensions.length && !state.extensions.includes(ext)) {
    showAlert('صيغة غير مدعومة', `الصيغة ${ext || '(بدون امتداد)'} غير مدعومة.`);
    return;
  }
  if (state.maxUploadMb && file.size > state.maxUploadMb * 1024 * 1024) {
    showAlert(
      'الملف كبير جداً',
      `حجم الملف ${formatBytes(file.size)} ويتجاوز الحد المسموح (${state.maxUploadMb} ميجابايت).`
    );
    return;
  }

  hideAlert();
  state.file = file;
  ui.fileName.textContent = file.name;
  ui.fileSize.textContent = formatBytes(file.size);
  ui.fileChip.hidden = false;
  ui.startBtn.disabled = false;
}

function clearFile() {
  state.file = null;
  ui.fileInput.value = '';
  ui.fileChip.hidden = true;
  ui.startBtn.disabled = true;
}

ui.dropzone.addEventListener('click', () => ui.fileInput.click());
ui.dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    ui.fileInput.click();
  }
});
ui.fileInput.addEventListener('change', () => acceptFile(ui.fileInput.files[0]));
ui.fileClear.addEventListener('click', (event) => {
  event.stopPropagation();
  clearFile();
});

['dragenter', 'dragover'].forEach((name) =>
  ui.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach((name) =>
  ui.dropzone.addEventListener(name, () => ui.dropzone.classList.remove('dragover'))
);
ui.dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  acceptFile(event.dataTransfer.files[0]);
});

/* ---------- transcription ---------- */

ui.startBtn.addEventListener('click', start);
ui.cancelBtn.addEventListener('click', cancel);

async function start() {
  if (!state.file) return;
  resetRun();

  const body = new FormData();
  body.append('file', state.file);
  body.append('model', state.model);
  body.append('language', ui.languageSelect.value);
  body.append('task', ui.taskSelect.value);

  ui.startBtn.disabled = true;
  ui.statusCard.hidden = false;
  setStatus('جارٍ رفع الملف…', true);

  let response;
  try {
    response = await fetch('/api/transcribe', { method: 'POST', body });
  } catch (error) {
    finishRun();
    showAlert('تعذّر الاتصال بالخادم', String(error));
    return;
  }

  if (!response.ok) {
    finishRun();
    showAlert('تعذّر بدء التحويل', await readError(response));
    return;
  }

  const job = await response.json();
  state.jobId = job.job_id;
  state.startedAt = Date.now();
  state.timer = setInterval(tickElapsed, 1000);
  ui.cancelBtn.hidden = false;
  setStatus(job.status_label, true);
  openStream(job.job_id);
}

function resetRun() {
  closeStream();
  hideAlert();
  state.seen.clear();
  state.texts = [];
  state.finished = false;
  ui.flow.textContent = '';
  ui.timed.replaceChildren();
  ui.resultCard.hidden = true;
  ui.resultMeta.textContent = '';
  setProgress(0);
  [ui.dlTxt, ui.dlSrt, ui.dlVtt].forEach((anchor) => {
    anchor.removeAttribute('href');
    anchor.setAttribute('aria-disabled', 'true');
  });
}

function openStream(jobId) {
  const source = new EventSource(`/api/jobs/${jobId}/stream`);
  state.source = source;

  source.addEventListener('snapshot', (event) => {
    const data = JSON.parse(event.data);
    applyStatus(data);
    (data.segments || []).forEach(addSegment);
  });

  source.addEventListener('status', (event) => applyStatus(JSON.parse(event.data)));

  source.addEventListener('segment', (event) => {
    const data = JSON.parse(event.data);
    addSegment(data.segment);
    setProgress(data.progress);
  });

  source.addEventListener('done', (event) => {
    const data = JSON.parse(event.data);
    (data.segments || []).forEach(addSegment);
    applyStatus(data);
    setProgress(1);
    enableDownloads(jobId);
    showResultMeta(data);
    finishRun();
  });

  source.addEventListener('error', (event) => {
    // A named "error" event carries a server-side failure payload; a bare one
    // is a transport problem, so only the former has data.
    if (event.data) {
      const data = JSON.parse(event.data);
      applyStatus(data);
      showAlert('فشل التحويل', data.error || 'حدث خطأ.', data.error_detail);
      finishRun();
      return;
    }
    if (!state.finished && source.readyState === EventSource.CLOSED) {
      reconcile(jobId);
    }
  });

  source.addEventListener('cancelled', () => {
    setStatus('تم الإلغاء', false);
    finishRun();
  });
}

/* If the SSE transport dies, ask once for the authoritative job state rather
   than leaving the UI stuck mid-progress. */
async function reconcile(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    (data.segments || []).forEach(addSegment);
    applyStatus(data);
    if (data.status === 'done') {
      setProgress(1);
      enableDownloads(jobId);
      showResultMeta(data);
    } else if (data.status === 'error') {
      showAlert('فشل التحويل', data.error || '', data.error_detail);
    } else {
      showAlert('انقطع الاتصال بالخادم', 'المهمة ما زالت قيد التنفيذ. أعد تحميل الصفحة للمتابعة.');
    }
  } catch (error) {
    showAlert('انقطع الاتصال بالخادم', String(error));
  } finally {
    finishRun();
  }
}

function applyStatus(data) {
  const busy = data.status === 'queued' || data.status === 'converting' || data.status === 'loading';
  setStatus(data.status_label || '', busy);
  if (typeof data.progress === 'number' && !busy) setProgress(data.progress);
}

function addSegment(segment) {
  if (!segment || state.seen.has(segment.index)) return;
  state.seen.add(segment.index);
  state.texts.push(segment.text);

  ui.flow.textContent = state.texts.join(' ');

  const item = document.createElement('li');
  item.className = 'fresh';

  const stamp = document.createElement('span');
  stamp.className = 'ts';
  stamp.dir = 'ltr';
  stamp.textContent = `${formatClock(segment.start)} – ${formatClock(segment.end)}`;

  const text = document.createElement('span');
  text.className = 'seg-text';
  text.textContent = segment.text;

  item.append(stamp, text);
  ui.timed.append(item);

  ui.resultCard.hidden = false;
  autoScroll();
}

function autoScroll() {
  const pane = ui.timed.hidden ? ui.flow : ui.timed;
  const nearBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 120;
  if (nearBottom) pane.scrollTop = pane.scrollHeight;
}

function setStatus(label, indeterminate) {
  ui.statusLabel.textContent = label;
  ui.progress.classList.toggle('indeterminate', Boolean(indeterminate));
}

function setProgress(fraction) {
  const percent = Math.round(Math.min(1, Math.max(0, fraction || 0)) * 100);
  ui.progressFill.style.width = `${percent}%`;
  ui.progress.setAttribute('aria-valuenow', String(percent));
}

function tickElapsed() {
  const elapsed = (Date.now() - state.startedAt) / 1000;
  ui.statusMeta.textContent = formatClock(elapsed);
}

function enableDownloads(jobId) {
  const targets = [
    [ui.dlTxt, 'txt'],
    [ui.dlSrt, 'srt'],
    [ui.dlVtt, 'vtt'],
  ];
  targets.forEach(([anchor, fmt]) => {
    anchor.href = `/api/jobs/${jobId}/download?fmt=${fmt}`;
    anchor.removeAttribute('aria-disabled');
  });
}

function modelLabel(key) {
  const spec = state.models.find((m) => m.key === key);
  return spec ? spec.label : '';
}

function showResultMeta(data) {
  const info = data.info || {};
  const elapsed = (Date.now() - state.startedAt) / 1000;
  const speed = info.duration && elapsed > 0 ? (info.duration / elapsed).toFixed(1) : null;
  const parts = [
    `عدد المقاطع: ${data.segment_count}`,
    info.duration ? `مدة الصوت: ${formatClock(info.duration)}` : null,
    `زمن المعالجة: ${formatClock(elapsed)}`,
    speed ? `أسرع من الزمن الحقيقي بـ ${speed}×` : null,
    // The checkpoint name means nothing to the reader; show the chosen label.
    modelLabel(info.model) ? `النموذج: ${modelLabel(info.model)}` : null,
    info.device ? `المعالجة: ${info.device === 'cuda' ? 'GPU' : 'CPU'}` : null,
    info.language ? `اللغة المكتشفة: ${info.language}` : null,
  ].filter(Boolean);
  ui.resultMeta.textContent = parts.join(' • ');
}

function finishRun() {
  state.finished = true;
  closeStream();
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  ui.cancelBtn.hidden = true;
  ui.startBtn.disabled = !state.file;
  ui.progress.classList.remove('indeterminate');
}

function closeStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

async function cancel() {
  if (!state.jobId) return;
  ui.cancelBtn.disabled = true;
  try {
    await fetch(`/api/jobs/${state.jobId}`, { method: 'DELETE' });
    setStatus('تم الإلغاء', false);
  } catch (error) {
    showAlert('تعذّر إلغاء المهمة', String(error));
  } finally {
    ui.cancelBtn.disabled = false;
    finishRun();
  }
}

/* ---------- result view ---------- */

ui.tabFlow.addEventListener('click', () => selectTab(true));
ui.tabTimed.addEventListener('click', () => selectTab(false));

function selectTab(flow) {
  ui.flow.hidden = !flow;
  ui.timed.hidden = flow;
  ui.tabFlow.classList.toggle('active', flow);
  ui.tabTimed.classList.toggle('active', !flow);
  ui.tabFlow.setAttribute('aria-selected', String(flow));
  ui.tabTimed.setAttribute('aria-selected', String(!flow));
}

ui.copyBtn.addEventListener('click', async () => {
  const text = state.texts.join(' ');
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    ui.copyBtn.textContent = 'تم النسخ ✓';
  } catch (_) {
    ui.copyBtn.textContent = 'تعذّر النسخ';
  }
  setTimeout(() => { ui.copyBtn.textContent = 'نسخ النص'; }, 1800);
});

/* ---------- go ---------- */

loadMetadata().catch((error) =>
  showAlert('تعذّر تحميل إعدادات الخادم', String(error))
);
loadHealth();
