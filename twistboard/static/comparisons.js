// Model comparisons on shared physics samples
//
// mikael.mieskolainen@cern.ch (2026)

const comparisonCheckpoints = new Map()
let comparisonRecord = null
let comparisonRequest = null
let startingComparison = false
let comparisonSort = {column: 1, ascending: false}
const comparisonHidden = new Set()
const comparisonColors = ['#80a9ff', '#ff8585', '#59d6c8', '#e6a3dc', '#e3ce77', '#a4b2cb', '#a8dc86']

function addComparisonCheckpoint(container, run, checkpoint) {
  /*
  Add a checkpoint to the persistent comparison selection

  Args:
      container: Checkpoint row action cell
      run: Experiment metadata
      checkpoint: Saved checkpoint entry
  Returns:
      None: Selection button is attached
  */
  const path = `${run.command_directory || run.directory}/checkpoints/${checkpoint.file}`
  const button = node('button', comparisonCheckpoints.has(path) ? 'Selected' : 'Compare', 'command-button')
  button.type = 'button'
  button.setAttribute('aria-pressed', comparisonCheckpoints.has(path))
  button.onclick = () => {
    if (comparisonCheckpoints.has(path)) comparisonCheckpoints.delete(path)
    else comparisonCheckpoints.set(path, {label: `${run.id} / ${checkpoint.file}`, checkpoint: `${run.id}/checkpoints/${checkpoint.file}`})
    button.textContent = comparisonCheckpoints.has(path) ? 'Selected' : 'Compare'
    button.setAttribute('aria-pressed', comparisonCheckpoints.has(path))
    renderComparisonCheckpoints()
  }
  container.append(button)
}

function renderComparisonCheckpoints() {
  /*
  Show selected models without changing their checkpoint files

  Returns:
      None: Selection list and launch controls are updated
  */
  $('comparison-checkpoints').replaceChildren()
  for (const [path, {label}] of comparisonCheckpoints) {
    const row = node('p', `${label} `)
    const remove = node('button', 'Remove', 'command-button')
    remove.onclick = () => { comparisonCheckpoints.delete(path); renderComparisonCheckpoints() }
    row.append(remove)
    $('comparison-checkpoints').append(row)
  }
  $('comparison-card').disabled = comparisonCheckpoints.size < 2
  $('comparison-run').disabled = startingComparison || comparisonCheckpoints.size < 2
  $('open-comparisons').textContent = `Compare models${comparisonCheckpoints.size ? ` (${comparisonCheckpoints.size})` : ''}`
}

function comparisonOptions(id, rows, current) {
  /*
  Replace select options and retain a valid selection

  Args:
      id: Select element ID
      rows: Option values and labels
      current: Preferred selected value
  Returns:
      None: Options are updated
  */
  const select = $(id)
  select.replaceChildren()
  for (const [value, label] of rows) {
    const option = node('option', label)
    option.value = value
    select.append(option)
  }
  if (rows.some(([value]) => value === current)) select.value = current
}

async function loadComparisons(preferred) {
  /*
  Fetch saved comparisons from the dashboard output directory

  Args:
      preferred: Optional comparison ID to open
  Returns:
      None: Saved comparisons and the selected result are loaded
  */
  try {
    const rows = await get('/api/comparisons')
    comparisonOptions('comparison-choice', rows.map(row => [row.id, `${row.name} - ${row.status} - ${formatTime(row.started_at)}`]), typeof preferred === 'string' ? preferred : $('comparison-choice').value)
    if (rows.length) await loadComparison()
    else $('comparison-status').textContent = 'No saved comparisons. Run a comparison card to create one.'
  } catch (error) { $('comparison-status').textContent = error.message }
}

async function loadComparison(selection) {
  /*
  Discard stale requests when the selected comparison changes

  Args:
      selection: Optional category selection ID
  Returns:
      None: Comparison controls and results are updated
  */
  comparisonRequest?.abort()
  const controller = new AbortController()
  comparisonRequest = controller
  try {
    const id = $('comparison-choice').value
    if (!id) return
    const endpoint = '/api/comparisons/' + encodeURIComponent(id)
    let record = await get(endpoint, controller.signal)
    if (controller.signal.aborted) return
    if (!record.selections.length) {
      comparisonRecord = record
      $('comparison-status').textContent = `${record.status} - ${record.stage || 'Starting comparison'}${record.error ? ': ' + record.error : ''}`
      $('comparison-chart').replaceChildren()
      $('comparison-table').replaceChildren()
      $('comparison-models').replaceChildren()
      $('comparison-log').textContent = record.log || ''
      return
    }
    const chosen = record.selections.some(row => row.id === selection) ? selection : record.selections[0].id
    record = await get(endpoint + '?selection=' + encodeURIComponent(chosen), controller.signal)
    if (controller.signal.aborted) return
    const same = comparisonRecord?.id === record.id && !!comparisonRecord?.selection
    comparisonRecord = record
    const classification = record.schema.name === 'classification'
    $('comparison-output-label').textContent = classification ? 'Class' : 'Target'
    comparisonOptions('comparison-plot', classification ? [['roc', 'ROC'], ['rejection', 'Background rejection'], ['precision_recall', 'Precision–recall'], ['calibration', 'Calibration']] : [['residual', 'Residuals'], ['rmse', 'RMSE by target'], ['coverage_1sigma', 'Interval coverage']], same ? $('comparison-plot').value : '')
    if (!same) comparisonSort = {column: 1, ascending: !classification}
    $('comparison-log').textContent = record.log || ''
    comparisonOptions('comparison-selection', record.selections.map(row => [row.id, row.label]), chosen)
    comparisonOptions('comparison-reference', record.models.map(row => [row.label, row.label]), same ? $('comparison-reference').value : record.reference_model)
    comparisonOptions('comparison-class', record.schema.outputs.map((label, i) => [String(i), label]), same ? $('comparison-class').value : String(record.schema.signal_class ?? 0))
    if (!same) comparisonHidden.clear()
    $('comparison-models').replaceChildren()
    record.models.forEach((model, index) => {
      const label = node('label')
      const input = node('input')
      input.type = 'checkbox'
      input.checked = !comparisonHidden.has(model.label)
      input.onchange = () => {
        if (input.checked) comparisonHidden.delete(model.label)
        else comparisonHidden.add(model.label)
        renderComparison()
      }
      label.style.color = comparisonColors[index % comparisonColors.length]
      label.append(input, node('span', model.label))
      label.title = `${model.checkpoint}\nSHA256: ${model.sha256}`
      $('comparison-models').append(label)
    })
    $('comparison-downloads').replaceChildren(link(`${record.id}/config.json`, 'Resolved comparison config'), link(`${record.id}/dataset.json`, 'Dataset details'), link(`${record.id}/selections/${chosen}/comparison.json`, 'Metrics and paired intervals'), link(`${record.id}/selections/${chosen}/metrics.csv`, 'Metric table CSV'), link(`${record.id}/selections/${chosen}/paired.csv`, 'Paired intervals CSV'))
    for (const file of record.files || []) $('comparison-downloads').append(link(file.path, file.path.split('/').at(-1)))
    record.models.forEach((model, index) => {
      $('comparison-downloads').append(link(`${record.id}/${model.directory}/predictions.npz`, `${model.label} predictions`))
      if (classification && record.selection.models?.length) $('comparison-downloads').append(link(`${record.id}/selections/${chosen}/model_${String(index).padStart(3, '0')}/curves.npz`, `${model.label} full curves`))
    })
    renderComparison()
  } catch (error) {
    if (error.name !== 'AbortError') $('comparison-status').textContent = error.message
  } finally {
    if (comparisonRequest === controller) comparisonRequest = null
  }
}

function renderComparison() {
  /*
  Render sortable metrics and overlays for the common event selection

  Returns:
      None: Metrics and curves are updated
  */
  if (!comparisonRecord?.selection) return
  const record = comparisonRecord
  const selection = record.selection
  const reference = $('comparison-reference').value
  const className = record.schema.outputs[Number($('comparison-class').value)]
  const uncertainty = Object.values(Object.values(Object.values(selection.paired || {})[0] || {})[0] || {})[0]
  const intervalNote = uncertainty?.requested_resamples ? `${(100 * uncertainty.confidence).toFixed(1)}% paired bootstrap intervals` : 'Bootstrap intervals disabled'
  const interval = entry => !entry || entry.estimate == null ? '—' : `${number(entry.estimate)}${entry.lower == null ? '' : ` [${number(entry.lower)}, ${number(entry.upper)}]`}`
  $('comparison-status').textContent = `${record.status}${record.stage ? ' - ' + record.stage : ''} - ${selection.label} - ${number(selection.support.events)} events${selection.reason ? ' - ' + selection.reason : ''}${record.error ? ' - ' + record.error : ''}. ${intervalNote}. Hover or click curves for values.${record.schema.name === 'classification' ? ' Display curves are sampled. Downloads retain every threshold.' : ''}`
  const table = node('table')
  const head = node('tr')
  const classification = record.schema.name === 'classification'
  const efficiencies = classification ? selection.models?.flatMap(model => model.metrics.classes[className]?.operating_points?.map(point => point.requested_efficiency) || []) || [] : []
  const unique = [...new Set(efficiencies)]
  const titles = classification ? ['Model', 'AUC', `ΔAUC vs ${reference}`, ...unique.flatMap(eff => [`Rejection at ${number(eff * 100)}%`, `Ratio vs ${reference}`]), 'Cross-entropy', 'Brier'] : ['Model', 'RMSE', `ΔRMSE vs ${reference}`, 'MAE', 'Bias', 'NLL', '1σ coverage', '2σ coverage']
  const density = !classification && selection.models?.some(model => 'density_nll' in model.metrics)
  if (density) titles.push('Joint density NLL', `Δ joint NLL vs ${reference}`)
  titles.forEach((title, column) => {
    const cell = node('th')
    const sort = node('button', title, 'comparison-sort')
    sort.onclick = () => {
      comparisonSort = {column, ascending: comparisonSort.column === column ? !comparisonSort.ascending : false}
      renderComparison()
    }
    cell.append(sort)
    head.append(cell)
  })
  const thead = node('thead')
  thead.append(head)
  table.append(thead)
  const rows = (selection.models || []).filter(model => !comparisonHidden.has(model.label)).map(model => {
    const metrics = classification ? model.metrics.classes[className] || {} : model.metrics[className] || {}
    const paired = selection.paired?.[reference]?.[model.label] || {}
    if (!classification) {
      const pairedRmse = paired[`rmse:${className}`]
      const values = [model.label, metrics.rmse, pairedRmse?.estimate, metrics.mae, metrics.bias, metrics.nll, metrics.coverage_1sigma, metrics.coverage_2sigma]
      const texts = [model.label, number(metrics.rmse), interval(pairedRmse), ...values.slice(3).map(number)]
      if (density) {
        values.push(model.metrics.density_nll, paired.density_nll?.estimate)
        texts.push(number(model.metrics.density_nll), interval(paired.density_nll))
      }
      return {values, texts}
    }
    const values = [model.label, metrics.auc, paired[`auc:${className}`]?.estimate]
    const texts = [model.label, number(metrics.auc), interval(paired[`auc:${className}`])]
    for (const eff of unique) {
      const point = metrics.operating_points?.find(p => p.requested_efficiency === eff)
      const ratio = paired[`rejection:${className}:${eff}`]
      values.push(point?.rejection, ratio?.estimate)
      texts.push(point?.zero_background ? 'No background observed' : point?.physical === false ? 'Undefined' : number(point?.rejection), interval(ratio))
    }
    values.push(model.metrics.cross_entropy, model.metrics.brier)
    texts.push(number(model.metrics.cross_entropy), number(model.metrics.brier))
    return {values, texts}
  })
  rows.sort((a, b) => {
    const x = a.values[comparisonSort.column], y = b.values[comparisonSort.column]
    if (x == null) return y == null ? 0 : 1
    if (y == null) return -1
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * (comparisonSort.ascending ? 1 : -1)
  })
  const body = node('tbody')
  for (const row of rows) {
    const tr = node('tr')
    for (const text of row.texts) tr.append(node('td', text))
    body.append(tr)
  }
  table.append(body)
  $('comparison-table').replaceChildren(table)
  renderComparisonChart(selection, Number($('comparison-class').value))
}

function renderComparisonChart(selection, index) {
  /*
  Draw physical XY curves with log controls and hover or pinned values

  Args:
      selection: Common-event metrics and curves
      index: Class or regression target index
  Returns:
      None: Interactive SVG curves are attached
  */
  const container = $('comparison-chart')
  container.replaceChildren()
  const kind = $('comparison-plot').value
  const logX = $('comparison-x').dataset.scale === 'log10', logY = $('comparison-y').dataset.scale === 'log10'
  const valid = (x, y) => Number.isFinite(x) && Number.isFinite(y) && (!logX || x > 0) && (!logY || y > 0)
  const series = (selection.models || []).map((model, n) => {
    const curve = model.curves[String(index)]
    let pairs = []
    if (comparisonRecord.schema.name !== 'classification') pairs = (curve?.[kind]?.x || []).map((x, i) => [x, curve[kind].y[i]])
    else if (kind === 'calibration') pairs = (model.metrics.calibration || []).map(bin => [bin.confidence, bin.accuracy])
    else if (curve) pairs = kind === 'precision_recall' ? curve.recall.map((x, i) => [x, curve.precision[i]]) : curve.fpr.map((x, i) => kind === 'rejection' ? [curve.tpr[i], x > 0 && x <= 1 && curve.tpr[i] >= 0 && curve.tpr[i] <= 1 ? 1 / x : NaN] : [x, curve.tpr[i]])
    return {label: model.label, n, color: comparisonColors[n % comparisonColors.length], pairs: pairs.filter(([x, y]) => valid(x, y))}
  }).filter(model => !comparisonHidden.has(model.label) && model.pairs.length)
  if (!series.length) { container.append(node('p', 'No curves available on these scales.', 'subtle')); return }
  const tx = x => logX ? Math.log10(x) : x, ty = y => logY ? Math.log10(y) : y
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity
  for (const model of series) for (const [x, y] of model.pairs) {
    xmin = Math.min(xmin, tx(x)); xmax = Math.max(xmax, tx(x))
    ymin = Math.min(ymin, ty(y)); ymax = Math.max(ymax, ty(y))
  }
  if (xmin === xmax) xmax = xmin + 1
  if (ymin === ymax) ymax = ymin + 1
  const sx = x => 130 + (tx(x) - xmin) / (xmax - xmin) * 740
  const sy = y => 340 - (ty(y) - ymin) / (ymax - ymin) * 310
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', '0 0 900 400')
  svg.setAttribute('role', 'img')
  svg.setAttribute('aria-label', `${kind} model comparison`)
  svg.setAttribute('tabindex', '0')
  const make = (tag, attrs, text) => {
    const element = document.createElementNS(svg.namespaceURI, tag)
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value)
    if (text !== undefined) element.textContent = text
    svg.append(element)
    return element
  }
  for (let i = 0; i <= 8; i++) {
    const x = 130 + i / 8 * 740, y = 340 - i / 8 * 310
    make('line', {x1: x, x2: x, y1: 30, y2: 340, stroke: '#283244'})
    make('line', {x1: 130, x2: 870, y1: y, y2: y, stroke: '#283244'})
    const xv = xmin + i / 8 * (xmax - xmin), yv = ymin + i / 8 * (ymax - ymin)
    make('text', {x, y: 366, fill: '#8e9bb0', 'text-anchor': 'middle', 'font-size': 14}, number(logX ? 10 ** xv : xv))
    make('text', {x: 117, y: y + 5, fill: '#8e9bb0', 'text-anchor': 'end', 'font-size': 14}, number(logY ? 10 ** yv : yv))
  }
  const labels = {roc: ['Background efficiency εB', 'Signal efficiency εS'], rejection: ['Signal efficiency εS', 'Background rejection 1/εB'], precision_recall: ['Recall', 'Precision'], calibration: ['Confidence', 'Accuracy'], residual: ['Prediction − target', 'Weighted density'], rmse: ['Target', 'RMSE'], coverage_1sigma: ['Target', 'One-sigma interval coverage']}
  if (comparisonRecord.schema.name !== 'classification' && comparisonRecord.schema.units?.[index]) {
    const unit = ` [${comparisonRecord.schema.units[index]}]`
    labels[kind][0] += unit
    if (kind === 'rmse') labels[kind][1] += unit
  }
  make('text', {x: 500, y: 395, fill: '#8e9bb0', 'text-anchor': 'middle'}, labels[kind][0])
  make('text', {x: 20, y: 185, transform: 'rotate(-90 20 185)', fill: '#8e9bb0', 'text-anchor': 'middle'}, labels[kind][1])
  const points = []
  for (const model of series) {
    make('path', {d: model.pairs.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x)},${sy(y)}`).join(' '), stroke: model.color, 'stroke-width': 2.5, 'stroke-dasharray': ['', '8 4', '8 3 2 3', '2 4'][model.n % 4], fill: 'none'})
    for (const [x, y] of model.pairs) points.push({x, y, px: sx(x), py: sy(y), label: model.label, color: model.color})
  }
  if (kind === 'roc' || kind === 'calibration') {
    const low = Math.max(logX ? 10 ** xmin : xmin, logY ? 10 ** ymin : ymin, 0)
    const high = Math.min(logX ? 10 ** xmax : xmax, logY ? 10 ** ymax : ymax, 1)
    if (high > low && valid(low, low)) {
      const points = Array.from({length: 101}, (_, i) => low + (high - low) * i / 100)
      make('path', {d: points.map((x, i) => `${i ? 'L' : 'M'}${sx(x)},${sy(x)}`).join(' '), stroke: '#8e9bb0', 'stroke-width': 1.5, 'stroke-dasharray': '5 5', fill: 'none'})
      make('text', {x: 140, y: 22, fill: '#8e9bb0', 'font-size': 13}, kind === 'roc' ? 'random' : 'ideal calibration')
    }
  }
  const tooltip = node('div', undefined, 'curve-tooltip')
  tooltip.hidden = true
  const marker = make('circle', {r: 5, fill: 'none', 'stroke-width': 2, visibility: 'hidden'})
  let pinned = false
  const hide = () => { tooltip.hidden = true; marker.setAttribute('visibility', 'hidden') }
  svg.onpointermove = event => {
    if (pinned) return
    const matrix = svg.getScreenCTM()
    if (!matrix) return
    let nearest, distance = 20
    for (const point of points) {
      const d = Math.hypot(event.clientX - (matrix.a * point.px + matrix.e), event.clientY - (matrix.d * point.py + matrix.f))
      if (d < distance) { distance = d; nearest = point }
    }
    if (!nearest) return hide()
    tooltip.textContent = `${nearest.label}: ${labels[kind][0]} = ${number(nearest.x)}, ${labels[kind][1]} = ${number(nearest.y)}`
    tooltip.hidden = false
    tooltip.style.left = '135px'
    tooltip.style.top = '8px'
    marker.setAttribute('cx', nearest.px)
    marker.setAttribute('cy', nearest.py)
    marker.setAttribute('stroke', nearest.color)
    marker.setAttribute('visibility', 'visible')
  }
  svg.onclick = event => { if (!pinned) svg.onpointermove(event); pinned = !pinned }
  svg.onpointerleave = () => { if (!pinned) hide() }
  svg.onkeydown = event => { if (event.key === 'Escape') { pinned = false; hide() } }
  container.append(svg, tooltip)
}

$('open-comparisons').onclick = () => {
  if (window.twistExplorer) window.twistExplorer.hide()
  comparisonMode = true
  $('detail').hidden = true
  $('empty').hidden = true
  $('comparison-view').hidden = false
  renderComparisonCheckpoints()
  loadComparisons()
}
$('comparison-card').onclick = () => {
  const models = [...comparisonCheckpoints].map(([checkpoint, {label}]) => ({label, checkpoint}))
  const config = {comparison: {name: 'model-comparison', dataset: {reference_checkpoint: models[0].checkpoint, split: 'test'}, models, reference_model: models[0].label}}
  const url = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2) + '\n'], {type: 'application/yaml'}))
  const download = node('a')
  download.href = url
  download.download = 'comparison.yml'
  download.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  $('comparison-command').textContent = 'Save comparison.yml in the project directory, then run:\npython -m twistformer.cli.compare --config comparison.yml'
  $('comparison-command').hidden = false
}
$('comparison-choice').onchange = () => loadComparison()
$('comparison-selection').onchange = () => loadComparison($('comparison-selection').value)
$('comparison-refresh').onclick = loadComparisons
for (const id of ['comparison-reference', 'comparison-class', 'comparison-plot']) $(id).onchange = renderComparison
for (const id of ['comparison-x', 'comparison-y']) {
  for (const button of $(id).querySelectorAll('button')) button.onclick = () => {
    setAxisScale(id, button.dataset.scale)
    renderComparison()
  }
}
$('comparison-enlarge').onclick = () => {
  const svg = $('comparison-chart').querySelector('svg')
  if (!svg) return
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: 'image/svg+xml'}))
  openFigure(url, `${comparisonRecord.name} - ${comparisonRecord.selection.label}`)
  $('figure-viewer').addEventListener('close', () => URL.revokeObjectURL(url), {once: true})
}

$('comparison-run').onclick = async () => {
  if (startingComparison || comparisonCheckpoints.size < 2) return
  startingComparison = true
  $('comparison-run').disabled = true
  $('comparison-status').textContent = 'Starting comparison on the Twistboard host…'
  try {
    const models = [...comparisonCheckpoints.values()].map(model => ({label: model.label, checkpoint: model.checkpoint}))
    const response = await fetch('/api/comparisons', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Twistboard-Action': 'compare'}, body: JSON.stringify({models})})
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || 'Cannot start comparison')
    await loadComparisons(result.id)
  } catch (error) { $('comparison-status').textContent = error.message }
  finally {
    startingComparison = false
    $('comparison-run').disabled = comparisonCheckpoints.size < 2
  }
}
setInterval(() => {
  if (comparisonMode && !comparisonRequest && ['queued', 'evaluating'].includes(comparisonRecord?.status)) loadComparison($('comparison-selection').value)
}, 3000)
