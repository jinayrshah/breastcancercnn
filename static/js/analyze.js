// Analyze page: choose image or sample -> POST /api/predict -> render result.
const $ = id => document.getElementById(id);
let chosenFile = null, chosenSample = null;

function ready() { $('go').disabled = !(chosenFile || chosenSample); }

function showPreview(src) { $('preview').src = src; $('preview').hidden = false; $('dropText').textContent = 'Change image'; }

$('file').addEventListener('change', e => {
  chosenFile = e.target.files[0] || null; chosenSample = null;
  document.querySelectorAll('.sample').forEach(b => b.classList.remove('on'));
  $('truth').textContent = '';
  if (chosenFile) showPreview(URL.createObjectURL(chosenFile));
  ready();
});

const drop = $('drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) { const dt = new DataTransfer(); dt.items.add(f); $('file').files = dt.files; $('file').dispatchEvent(new Event('change')); }
});

document.querySelectorAll('.sample').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.sample').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); chosenSample = b.dataset.file; chosenFile = null; $('file').value = '';
  showPreview(b.querySelector('img').src);
  $('truth').textContent = 'Ground truth for this sample: ' + b.title.split('ground truth ')[1];
  ready();
}));

$('form').addEventListener('submit', async e => {
  e.preventDefault();
  $('err').hidden = true; $('empty').hidden = true; $('out').hidden = true; $('loading').hidden = false; $('go').disabled = true;
  const fd = new FormData();
  if (chosenFile) fd.append('image', chosenFile); else fd.append('sample', chosenSample);
  fd.append('backend', $('backend').value); fd.append('patient', $('patient').value);
  try {
    const r = await fetch('/api/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Prediction failed');
    render(d);
  } catch (err) {
    $('loading').hidden = true; $('empty').hidden = false;
    $('err').textContent = err.message; $('err').hidden = false;
  } finally { ready(); }
});

function render(d) {
  const mal = d.label === 'Malignant';
  $('verdict').className = 'verdict ' + (mal ? 'malig' : 'benign');
  $('label').textContent = d.label;
  $('sub').textContent = `${(d.confidence * 100).toFixed(1)}% confidence · ${d.risk} risk`;
  $('pval').textContent = d.prob.toFixed(3);
  $('pin').style.left = '50%'; requestAnimationFrame(() => setTimeout(() => $('pin').style.left = (d.prob * 100) + '%', 30));
  const rows = [['Model', d.model], ['Mode', d.mode === 'scan' ? 'Region scan (sliding window)' : 'Single patch'],
                ['Inference time', d.ms + ' ms'], ['File', d.filename]];
  if (d.tiles) rows.push(['Tissue tiles', `${d.tiles.tissue} of ${d.tiles.total}`],
                         ['Suspicious tiles', `${d.tiles.suspicious} (${d.tiles.pct}%)`]);
  $('kv').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v).replace(/</g, '&lt;')}</dd>`).join('');
  $('steps').innerHTML = d.steps.map(s => `<figure><img src="${s.img}" alt="${s.title}"><figcaption><b>${s.title}</b><span>${s.note}</span></figcaption></figure>`).join('');
  $('exTitle').textContent = d.explain_title; $('exImg').src = d.explain; $('exNote').textContent = d.explain_note;
  $('reportLink').href = '/report/' + d.id;
  $('loading').hidden = true; $('out').hidden = false;
  if (window.innerWidth < 980) $('result').scrollIntoView({ behavior: 'smooth' });
}
