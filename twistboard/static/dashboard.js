// Experiment dashboard
//
// mikael.mieskolainen@cern.ch (2026)

const $ = id => document.getElementById(id)
let runs = []
let selected = decodeURIComponent(location.hash.slice(1))
let refreshController = null
let deleting = false
let startingEvaluation = false
let actionMessageTimer = null
let comparisonMode = false
let figureAnimation = null
const evaluationSelection = new Map()
const categorySelection = new Map()
let evaluationKey = null
let curveRows = []
let curveKey = null
let runListKey = null
let featureImportanceKey = null
let featureImportanceRequest = null
let distributionKey = null
let distributionRequest = null
const axisControls = ['loss-x-scale', 'loss-y-scale']
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
const number = value => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumSignificantDigits: 5})
const stepUnits = new Intl.NumberFormat(undefined, {maximumFractionDigits: 3})
const formatSteps = value => Number.isFinite(value) ? `${stepUnits.format(value / 1000)}k` : '—'
function scientificNumber(value) {
  /*
  Format a value with two mantissa decimals and a superscript exponent

  Args:
      value:  Numerical value or missing value
  Returns:
      text:  Scientific notation with an exponent set by the value
  */
  if (value == null || !Number.isFinite(Number(value))) return '—'
  if (Number(value) === 0) return '0.00'
  const [mantissa, exponent] = Number(value).toExponential(2).split('e')
  const superscripts = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
  const power = [...String(Number(exponent))].map(character => superscripts[character]).join('')
  return `${mantissa} × 10${power}`
}
const timeFormat = new Intl.DateTimeFormat('en-GB', {
  hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  weekday: 'long', day: 'numeric', month: 'short', year: 'numeric'
})
function formatTime(value) {
  /*
  Format a saved timestamp in the browser time zone

  Args:
      value: Saved ISO timestamp
  Returns:
      text: Time, weekday, day, month and year
  */
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  const parts = Object.fromEntries(timeFormat.formatToParts(date).map(part => [part.type, part.value]))
  return `${parts.hour}:${parts.minute}, ${parts.weekday} ${parts.day} ${parts.month} ${parts.year}`
}
const fileURL = path => '/files/' + path.split('/').map(encodeURIComponent).join('/')
function formatDuration(seconds) {
  /*
  Format active wall time without wrapping hours at one day

  Args:
      seconds:  Duration from the host or missing timing
  Returns:
      text: Hours, minutes and seconds
  */
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.floor(seconds)
  return [Math.floor(total / 3600), Math.floor(total / 60) % 60, total % 60]
    .map(value => String(value).padStart(2, '0')).join(':')
}
const node = (tag, text, className) => {
  const value = document.createElement(tag)
  if (text !== undefined) value.textContent = text
  if (className) value.className = className
  return value
}
const link = (path, title) => {
  const value = node('a', title)
  value.href = fileURL(path)
  value.target = '_blank'
  value.rel = 'noopener'
  return value
}
function readPreference(key) {
  /*
  Read an optional browser preference when storage is available

  Args:
      key:  Dashboard preference name
  Returns:
      value: Saved text or null
  */
  try { return localStorage.getItem(`twistboard.${key}`) } catch { return null }
}
function writePreference(key, value) {
  /*
  Save a browser preference without interrupting dashboard controls

  Args:
      key:  Dashboard preference name
      value:  Selected value
  */
  try { localStorage.setItem(`twistboard.${key}`, String(value)) } catch {}
}
function metricCard(label, value) {
  /*
  Build a shared training or evaluation metric card

  Args:
      label:  Observable name or label element
      value:  Formatted value
  Returns:
      card: Dashboard metric element
  */
  const card = node('div', undefined, 'card')
  card.append(typeof label === 'string' ? node('span', label) : label, node('strong', value))
  return card
}
function validationLosses(history) {
  /*
  Find the most recent and lowest recorded validation loss

  Args:
      history:  Training epoch records in order
  Returns:
      losses:  Last and best finite validation loss
  */
  let last = null
  let best = null
  for (const {val_loss: loss} of history) {
    if (!Number.isFinite(loss)) continue
    last = loss
    if (best === null || loss < best) best = loss
  }
  return {last, best}
}
function evaluationMetric(value, uncertainty) {
  /*
  Format a metric and its stored statistical uncertainty

  Args:
      value:  Scalar diagnostic
      uncertainty:  Bootstrap standard error or stored percentile bounds
  Returns:
      text:  Metric with statistical uncertainty when available
  */
  const error = uncertainty?.std
  if (Number.isFinite(value) && Number.isFinite(error) && error >= 0) {
    if (error === 0) return `${number(value)} ± 0`
    const digits = Math.max(0, 1 - Math.floor(Math.log10(error)))
    if (digits > 12) return `${number(value)} ± ${scientificNumber(error)}`
    const options = {minimumFractionDigits: digits, maximumFractionDigits: digits}
    return `${value.toLocaleString(undefined, options)} ± ${error.toLocaleString(undefined, options)}`
  }
  const lower = uncertainty?.lower_unbounded ? Infinity : uncertainty?.lower ?? uncertainty?.auc_lower
  const upper = uncertainty?.upper_unbounded ? Infinity : uncertainty?.upper ?? uncertainty?.auc_upper
  if (typeof value === 'number' && (Number.isFinite(lower) || lower === Infinity) && (Number.isFinite(upper) || upper === Infinity)) {
    return `${number(value)} [${number(lower)}, ${number(upper)}]`
  }
  return number(value)
}
function openFigure(source, title) {
  /*
  Show an evaluation figure over the current dashboard

  Args:
      source:  Figure image URL
      title:  Figure description
  */
  $('figure-image').src = source
  $('figure-image').alt = title
  $('figure-caption').textContent = title
  $('figure-viewer').showModal()
  document.body.classList.add('figure-open')
  if (figureAnimation) figureAnimation.cancel()
  if (!reducedMotion.matches) {
    figureAnimation = $('figure-viewer').animate([
      {opacity: 0, transform: 'scale(0.94)'},
      {opacity: 1, transform: 'scale(1)'}
    ], {duration: 220, easing: 'ease-out'})
  }
}
function closeFigure() {
  /*
  Animate the plot preview before closing its dialog

  Returns:
      None: Focus returns to the plot after the animation
  */
  const viewer = $('figure-viewer')
  if (!viewer.open) return
  const style = getComputedStyle(viewer)
  const start = {opacity: style.opacity, transform: style.transform}
  if (figureAnimation) figureAnimation.cancel()
  if (reducedMotion.matches) {
    viewer.close()
    return
  }
  figureAnimation = viewer.animate([
    start, {opacity: 0, transform: 'scale(0.94)'}
  ], {duration: 180, easing: 'ease-in', fill: 'forwards'})
  figureAnimation.onfinish = () => viewer.close()
}
function showEvaluationCommand(directory, checkpoint) {
  /*
  Show a shell command for test evaluation of one checkpoint

  Args:
      directory:  Run directory on the training host
      checkpoint:  Checkpoint filename
  Returns:
      None: A selectable command opens in a dialog
  */
  const path = `${directory}/checkpoints/${checkpoint}`
  const quoted = "'" + path.replaceAll("'", "'\"'\"'") + "'"
  $('evaluation-command').value = `python -m twistformer.cli.evaluate --checkpoint ${quoted} --split test`
  $('command-title').textContent = `Evaluate ${checkpoint}`
  $('command-copy-status').textContent = ''
  $('command-viewer').showModal()
  $('evaluation-command').focus()
  $('evaluation-command').select()
}
async function get(path, signal) {
  const response = await fetch(path, {cache: 'no-store', signal})
  if (!response.ok) throw new Error('Unable to read experiment files')
  return response.json()
}
function renderList() {
  const list = $('runs')
  const search = $('search').value.toLowerCase()
  const key = JSON.stringify([selected, deleting, search, runs.map(run => [run.id, run.model, run.task, run.driver, run.status, run.epochs, run.training_started_at, run.created_at])])
  if (key === runListKey) return
  runListKey = key
  const scroll = list.scrollTop
  const focused = list.contains(document.activeElement) ? document.activeElement.dataset.runId : null
  list.replaceChildren()
  for (const run of runs.filter(run => [run.id, run.model, run.task, run.driver].join(' ').toLowerCase().includes(search))) {
    const button = node('button', undefined, 'run' + (selected === run.id ? ' active' : ''))
    button.disabled = deleting
    button.dataset.runId = run.id
    button.dataset.status = run.status
    button.append(node('strong', run.id), node('small', `${run.driver} / ${run.model}`), node('small', `${run.status}: ${run.epochs} epochs`, 'state'))
    const started = run.training_started_at || run.created_at
    const date = node('small', formatTime(started))
    date.title = `Latest training start: ${formatTime(started)}`
    button.append(date)
    button.onclick = () => activateRun(button)
    list.append(button)
    if (run.id === focused) button.focus({preventScroll: true})
  }
  list.scrollTop = scroll
  $('count').textContent = runs.length
}
function activateRun(button) {
  /*
  Activate an experiment through the shared mouse and keyboard action

  Args:
      button:  Experiment button in the filtered list
  Returns:
      None: Focus, scroll position, URL and selected experiment are updated
  */
  if (deleting || button.disabled) return
  button.focus({preventScroll: true})
  button.scrollIntoView({block: 'nearest', inline: 'nearest'})
  location.hash = encodeURIComponent(button.dataset.runId)
  selectRun(button.dataset.runId)
}
function navigateRuns(event) {
  /*
  Select the adjacent visible experiment with the arrow keys

  Args:
      event:  Keyboard event from the experiment list or search field
  Returns:
      None: Selection and keyboard focus move within the filtered list
  */
  if (!['ArrowUp', 'ArrowDown'].includes(event.key) || deleting ||
      event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.isComposing) return
  const buttons = [...$('runs').querySelectorAll('button.run:not(:disabled)')].filter(button => button.offsetParent !== null)
  if (!buttons.length) return
  event.preventDefault()
  let index = buttons.indexOf(document.activeElement)
  if (index < 0) index = buttons.findIndex(button => button.dataset.runId === selected)
  const step = event.key === 'ArrowDown' ? 1 : -1
  const next = index < 0 ? (step > 0 ? 0 : buttons.length - 1) : Math.max(0, Math.min(buttons.length - 1, index + step))
  const button = buttons[next]
  if (button.dataset.runId !== selected || document.activeElement !== button) button.click()
}
function selectRun(id) {
  /*
  Disable deletion until the selected run has loaded

  Args:
      id:  Selected experiment directory name
  */
  comparisonMode = false
  if (window.twistExplorer) window.twistExplorer.hide()
  if ($('comparison-view')) $('comparison-view').hidden = true
  if (id === selected) return refresh()
  selected = id
  evaluationKey = null
  for (const button of document.querySelectorAll('.delete-button')) button.disabled = true
  refreshController?.abort()
  refreshController = null
  return refresh()
}
function renderCurves(rows) {
  /*
  Render training and validation loss by epoch

  Args:
      rows:  Saved training history
  */
  curveRows = rows
  const key = JSON.stringify([selected, rows, ...axisControls.map(id => $(id).dataset.scale)])
  if (key === curveKey) return
  curveKey = key
  const epochs = rows.map(row => ({...row, x: Number.isFinite(row.epoch) ? row.epoch + 1 : null}))
  renderCurve(epochs, 'curve', 'loss', 'Loss', 'Training and validation loss', [['train_loss', '#59d6c8'], ['val_loss', '#80a9ff']])
}
function integerTicks(first, last, minimumValue = 1, capacity = 16) {
  /*
  Place count ticks at a constant positive integer interval

  Args:
      first:  Lower axis bound
      last:  Upper axis bound
      minimumValue:  Smallest allowed tick
      capacity:  Maximum desired number of labels
  Returns:
      ticks: Whole counts within the axis bounds
  */
  const minimum = Math.max(minimumValue, Math.ceil(first))
  const maximum = Math.floor(last)
  if (maximum < minimum) return []
  const required = Math.max(1, (maximum - minimum) / (capacity - 1))
  const power = 10 ** Math.floor(Math.log10(required))
  const step = [1, 2, 5, 10].find(value => value * power >= required) * power
  return Array.from({length: Math.floor((maximum - minimum) / step) + 1}, (_, index) => minimum + index * step)
}
function renderCurve(rows, container, controls, label, description, series, axis = 'epoch') {
  /*
  Plot recorded values with independent linear or logarithmic axes

  Args:
      rows:  Saved observations with horizontal coordinate x
      container:  Chart element ID
      controls:  Axis selector prefix
      label:  Vertical axis label
      description:  Accessible chart description
      series:  Metric keys and line colors
      axis:  Epoch, optimizer step or elapsed seconds
  */
  $(container).replaceChildren()
  const note = $(`${container}-note`)
  note.hidden = true
  if (!rows.length) return
  const logX = $(`${controls}-x-scale`).dataset.scale === 'log10'
  const logY = $(`${controls}-y-scale`).dataset.scale === 'log10'
  const finite = value => typeof value === 'number' && Number.isFinite(value)
  const xLabel = {step: 'Optimizer step', epoch: 'Epoch', elapsed_seconds: 'Total training time (s)'}[axis]
  const xNumber = axis === 'step' ? formatSteps : number
  const validX = row => finite(row.x) && (!logX || row.x > 0)
  const valid = (row, key) => validX(row) && finite(row[key]) && (!logY || row[key] > 0)
  const tx = value => logX ? Math.log10(value) : value
  const ty = value => logY ? Math.log10(value) : value
  const yNumber = label === 'Learning rate' ? scientificNumber : number
  const points = rows.flatMap(row => series.filter(([key]) => valid(row, key)).map(([key]) => [tx(row.x), ty(row[key])]))
  const omitted = rows.length * series.length - points.length
  if (omitted) {
    note.textContent = `${omitted} missing, non-finite or out-of-domain points omitted${logX || logY ? '. Log10 axes require positive values' : ''}`
    note.hidden = false
  }
  if (!points.length) {
    $(container).append(node('p', logX || logY ? 'No values available on these scales. Try linear axes.' : `No finite ${label.toLowerCase()} values available.`, 'subtle'))
    return
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', '0 0 900 400')
  svg.setAttribute('role', 'img')
  svg.setAttribute('tabindex', '0')
  svg.setAttribute('aria-label', `${description} by ${xLabel.toLowerCase()}, ${logX ? 'log10' : 'linear'} x axis, ${logY ? 'log10' : 'linear'} y axis. Hover or click points for values. Press Escape to close.`)
  const make = (tag, attrs, text) => {
    const child = document.createElementNS(svg.namespaceURI, tag)
    for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, value)
    if (text !== undefined) child.textContent = text
    svg.append(child)
    return child
  }
  let low = Infinity, high = -Infinity, first = Infinity, last = -Infinity
  for (const [px, py] of points) {
    low = Math.min(low, py)
    high = Math.max(high, py)
    first = Math.min(first, px)
    last = Math.max(last, px)
  }
  const nonnegative = low >= 0
  const floor = controls === 'loss' ? 0.01 : Math.max(Math.abs(low), Math.abs(high)) * 0.05
  const pad = Math.max((high - low) * 0.1, logY ? 0.1 : floor, Number.EPSILON)
  low -= pad
  high += pad
  if (label === 'Learning rate' && !logY && nonnegative) low = Math.max(0, low)
  if (first === last) {
    first -= logX ? 0.1 : 0.5
    last += logX ? 0.1 : 0.5
  }
  const y = value => 345 - (value - low) / (high - low) * 320
  const ticks = (min, max, log) => {
    if (log && Math.floor(max) - Math.ceil(min) >= 8) {
      const stride = Math.max(1, Math.ceil((Math.floor(max) - Math.ceil(min)) / 8))
      return Array.from({length: Math.floor((Math.floor(max) - Math.ceil(min)) / stride) + 1}, (_, i) => Math.ceil(min) + i * stride)
    }
    return Array.from({length: 9}, (_, i) => min + (max - min) * i / 8)
  }
  const yTicks = ticks(low, high, logY).map(value => [value, yNumber(logY ? 10 ** value : value)])
  const measure = document.createElement('canvas').getContext('2d')
  measure.font = `15px ${getComputedStyle($(container)).fontFamily}`
  const left = Math.max(110, 50 + Math.max(...yTicks.map(([, text]) => measure.measureText(text).width)))
  const x = value => left + (value - first) / (last - first) * (875 - left)
  for (const [value, text] of yTicks) {
    make('line', {x1: left, x2: 875, y1: y(value), y2: y(value), stroke: '#283244'})
    make('text', {x: left - 12, y: y(value) + 4, fill: '#8e9bb0', 'text-anchor': 'end', 'font-size': 15}, text)
  }
  const xTicks = logX || axis === 'elapsed_seconds' ? ticks(first, last, logX) : integerTicks(first, last, controls === 'observable' ? 0 : 1)
  for (const value of xTicks) {
    make('line', {x1: x(value), x2: x(value), y1: 25, y2: 345, stroke: '#283244'})
    make('text', {x: x(value), y: 367, fill: '#8e9bb0', 'text-anchor': 'middle', 'font-size': 15}, xNumber(logX ? 10 ** value : value))
  }
  const hoverPoints = []
  for (const [key, color] of series) {
    let drawing = false
    const path = rows.map(row => {
      if (!valid(row, key)) { drawing = false; return '' }
      const command = drawing ? 'L' : 'M'
      drawing = true
      return `${command}${x(tx(row.x))},${y(ty(row[key]))}`
    }).join(' ')
    make('path', {d: path, fill: 'none', stroke: color, 'stroke-width': 2.5})
    for (const row of rows) {
      if (!valid(row, key)) continue
      const px = x(tx(row.x))
      const py = y(ty(row[key]))
      hoverPoints.push({row, color, x: px, y: py})
      if (rows.length < 30) make('circle', {cx: px, cy: py, r: 3, fill: color})
    }
  }
  make('text', {x: (left + 875) / 2, y: 393, fill: '#8e9bb0', 'text-anchor': 'middle', 'font-size': 16}, logX ? `${xLabel} (log10)` : xLabel)
  make('text', {x: 16, y: 185, transform: 'rotate(-90 16 185)', fill: '#8e9bb0', 'text-anchor': 'middle', 'font-size': 16}, logY ? `${label} (log10)` : label)
  const marker = make('circle', {r: 5, fill: 'none', 'stroke-width': 2, 'pointer-events': 'none', visibility: 'hidden'})
  const tooltip = node('div', undefined, 'curve-tooltip')
  tooltip.setAttribute('role', 'tooltip')
  tooltip.hidden = true
  const names = {train_loss: 'Training loss', val_loss: 'Validation loss'}
  let pinned = false
  let activePoint = null
  const hide = () => {
    activePoint = null
    tooltip.hidden = true
    marker.setAttribute('visibility', 'hidden')
  }
  svg.onpointermove = event => {
    /*
    Show saved epoch values near the nearest plotted point

    Args:
        event:  Pointer position in screen coordinates
    */
    if (pinned) return
    const matrix = svg.getScreenCTM()
    if (!matrix) return hide()
    let nearest = null
    let distance = 16
    for (const point of hoverPoints) {
      const dx = event.clientX - (matrix.a * point.x + matrix.c * point.y + matrix.e)
      const dy = event.clientY - (matrix.b * point.x + matrix.d * point.y + matrix.f)
      const delta = Math.hypot(dx, dy)
      if (delta < distance) {
        nearest = point
        distance = delta
      }
    }
    if (!nearest) return hide()
    activePoint = nearest
    marker.setAttribute('cx', nearest.x)
    marker.setAttribute('cy', nearest.y)
    marker.setAttribute('stroke', nearest.color)
    marker.setAttribute('visibility', 'visible')
    tooltip.replaceChildren(node('strong', `${xLabel} ${xNumber(nearest.row.x)}`))
    for (const [key, color] of series) {
      if (!valid(nearest.row, key)) continue
      const value = node('div', `${controls === 'observable' ? label : names[key] || key}: ${yNumber(nearest.row[key])}`)
      value.style.color = color
      tooltip.append(value)
    }
    tooltip.hidden = false
    const bounds = $(container).getBoundingClientRect()
    const left = event.clientX - bounds.left + 12
    const top = event.clientY - bounds.top - tooltip.offsetHeight - 12
    tooltip.style.left = `${Math.max(0, Math.min(left, bounds.width - tooltip.offsetWidth))}px`
    tooltip.style.top = `${Math.max(0, Math.min(top, bounds.height - tooltip.offsetHeight))}px`
  }
  svg.onclick = event => {
    const previous = pinned ? activePoint : null
    pinned = false
    svg.onpointermove(event)
    if (previous === activePoint) hide()
    else pinned = !tooltip.hidden
    svg.focus({preventScroll: true})
  }
  svg.onkeydown = event => {
    if (event.key === 'Escape') {
      pinned = false
      hide()
    }
  }
  svg.onpointerleave = svg.onpointercancel = () => {
    if (!pinned) hide()
  }
  $(container).append(svg, tooltip)
}
function metricRows(value, path = '', uncertainty) {
  if (value !== null && typeof value === 'object') {
    return Object.entries(value).filter(([key]) => key !== 'uncertainties').flatMap(([key, item]) => {
      const error = value.uncertainties?.[key] ?? uncertainty?.[key] ?? (key === 'auc' ? value.auc_intervals?.rest : undefined)
      const estimate = key === 'rejection' && value.zero_background && error?.valid_resamples >= 2 ? Infinity : item
      return metricRows(estimate, path ? `${path}.${key}` : key, error)
    })
  }
  return [[path, value, uncertainty]]
}
function renderEvaluations(run) {
  const key = JSON.stringify([run.id, run.evaluations, evaluationSelection.get(run.id), [...categorySelection]])
  if (key === evaluationKey) return
  evaluationKey = key
  $('evaluations').replaceChildren()
  $('gallery').replaceChildren()
  $('evaluation-metrics').replaceChildren()
  $('evaluation-actions').replaceChildren()
  if (!run.evaluations.length) {
    $('evaluations').append(node('p', 'No evaluation results yet.', 'subtle'))
    return
  }
  const choices = [...run.evaluations].reverse()
  let current = choices.find(item => item.directory === evaluationSelection.get(run.id))
  current ||= choices.find(item => item.split === 'test' && item.status === 'completed') || choices[0]
  const controls = node('div', undefined, 'evaluation-controls')
  const label = node('label', 'Saved evaluation')
  label.htmlFor = 'evaluation-choice'
  const select = node('select')
  select.id = 'evaluation-choice'
  for (const [index, item] of choices.entries()) {
    const checkpoint = item.checkpoint?.split('/').at(-1) || 'Unknown checkpoint'
    const option = node('option', `#${choices.length - index} - ${item.split || 'custom'} - ${checkpoint} - ${formatTime(item.started_at)}`)
    option.value = item.directory
    option.selected = item.directory === current.directory
    select.append(option)
  }
  select.onchange = () => {
    evaluationSelection.set(run.id, select.value)
    renderEvaluations(run)
  }
  const remove = node('button', 'Delete evaluation', 'delete-button')
  remove.type = 'button'
  remove.disabled = deleting || startingEvaluation || !!current.delete_blocked || ['queued', 'evaluating'].includes(current.status)
  remove.title = current.delete_blocked || (current.status === 'evaluating' ? 'Stop evaluation before deleting this directory' : '')
  remove.onclick = () => {
    if (!remove.disabled) return deleteDirectory('evaluation', current.directory, current.path || current.directory)
  }
  controls.append(label, select)
  $('evaluation-actions').append(remove)
  $('evaluations').append(controls)
  const categories = current.category_views || []
  const categoryId = categorySelection.get(current.directory) || ''
  const category = categories.find(item => item.id === categoryId)
  const view = category || {
    directory: current.directory,
    metrics: current.metrics,
    files: (current.files || []).filter(item => !item.path.startsWith(current.directory + '/categories/'))
  }
  if (categories.length) {
    const categoryControls = node('div', undefined, 'evaluation-controls')
    const categoryLabel = node('label', 'Category')
    categoryLabel.htmlFor = 'category-choice'
    const categorySelect = node('select')
    categorySelect.id = 'category-choice'
    const inclusive = node('option', 'Inclusive - all events')
    inclusive.value = ''
    categorySelect.append(inclusive)
    for (const item of categories) {
      const option = node('option', item.label)
      option.value = item.id
      categorySelect.append(option)
    }
    categorySelect.value = category?.id || ''
    categorySelect.onchange = () => {
      categorySelection.set(current.directory, categorySelect.value)
      renderEvaluations(run)
    }
    categoryControls.append(categoryLabel, categorySelect)
    $('evaluations').append(categoryControls)
  }
  const box = node('div', undefined, 'evaluation')
  const summary = node('dl', undefined, 'evaluation-summary')
  for (const [label, value] of [
    ['Dataset', current.split || 'custom'],
    ['Status', current.status || 'unknown'],
    ['Started', formatTime(current.started_at)],
    ['Elapsed time', formatDuration(current.elapsed_seconds)],
    ['Checkpoint', current.checkpoint?.split('/').at(-1) || 'Unknown']
  ]) {
    const field = node('div')
    field.append(node('dt', label), node('dd', value))
    summary.append(field)
  }
  box.append(summary)
  if (category) {
    box.append(node('strong', category.label))
    if (category.support) box.append(node('p', `${number(category.support.events)} events - Weighted yield ${number(category.support.sum_weights)} - Effective events ${number(category.support.effective_events)}`, 'subtle'))
    box.append(node('p', `Missing: ${number(category.missing_events)} - Outside bins: ${number(category.out_of_range_events)}`, 'subtle'))
    if (category.reason) box.append(node('p', category.reason, 'subtle'))
    if (run.task === 'classification' && category.status !== 'comparison') box.append(node('p', 'ROC thresholds are determined within this bin.', 'subtle'))
  }
  const paths = node('details', undefined, 'evaluation-paths')
  paths.append(node('summary', 'File paths'))
  const pathList = node('dl')
  for (const [label, value] of [['Checkpoint', current.checkpoint || 'Unknown'], ['Results folder', current.path || current.directory]]) {
    const entry = node('dd')
    entry.append(node('code', value))
    pathList.append(node('dt', label), entry)
  }
  paths.append(pathList)
  box.append(paths)
  if (current.error) box.append(node('p', current.error, 'error'))
  if (current.log) {
    const log = node('details', undefined, 'evaluation-log')
    log.append(node('summary', 'Evaluation log'), node('pre', current.log))
    box.append(log)
  }
  const metrics = view.metrics || {}
  const cards = node('div', undefined, 'cards')
  for (const [key, title] of [['accuracy', 'Accuracy'], ['macro_auc', 'Macro AUC'], ['cross_entropy', 'Cross entropy'], ['preprocessed_loss', 'Evaluation loss'], ['density_nll', 'Joint density NLL'], ['bits_per_dimension', 'Image bits per dimension'], ['sample_count', 'Samples per event']]) {
    if (!(key in metrics)) continue
    const uncertainty = metrics.uncertainties?.[key]
    const card = metricCard(title, evaluationMetric(metrics[key], uncertainty))
    if (Number.isFinite(uncertainty?.std)) card.title = 'Statistical uncertainty from event bootstrap replicas'
    cards.append(card)
  }
  if (Array.isArray(metrics.signal_operating_points)) {
    for (const point of metrics.signal_operating_points) {
      if (!Number.isFinite(point?.requested_efficiency)) continue
      const uncertainty = point.uncertainties?.rejection
      let value
      if (point.physical === false) value = 'Undefined'
      else if (point.zero_background) value = uncertainty?.valid_resamples >= 2
        ? evaluationMetric(Infinity, uncertainty) : 'Zero background efficiency'
      else value = Number.isFinite(point.rejection) && point.rejection > 0
        ? evaluationMetric(point.rejection, uncertainty) : 'Undefined'
      const label = node('span', undefined, 'rejection-label')
      label.append('1/ϵ', node('sub', 'B'), ' at ϵ', node('sub', 'S'), ` = ${number(point.requested_efficiency)}`)
      const card = metricCard(label, value)
      card.title = point.physical === false ? 'This operating point has nonphysical signed efficiencies'
        : point.zero_background ? 'Zero background efficiency in this sample. Finite rejection is undefined.'
        : `1 / background efficiency. Achieved signal efficiency: ${number(point.efficiency * 100)}%. Score threshold: ${number(point.threshold)}`
      if (uncertainty?.std == null && uncertainty?.valid_resamples >= 2) {
        card.title += `\n${number(100 * uncertainty.confidence)}% event bootstrap interval.`
      }
      if (uncertainty?.zero_background_resamples) {
        card.title += `\n${number(uncertainty.zero_background_resamples)} of ${number(uncertainty.valid_resamples)} valid bootstrap replicas have zero background efficiency.`
      }
      cards.append(card)
    }
  }
  box.append(cards)
  const metricDetails = node('div', undefined, 'evaluation')
  const rows = metricRows(metrics)
  if (rows.length) {
    const wrap = node('div', undefined, 'table-wrap metric-table')
    const table = node('table')
    const head = node('thead')
    const heading = node('tr')
    heading.append(node('th', 'Metric / class / operating point'), node('th', 'Value'))
    head.append(heading)
    const body = node('tbody')
    for (const [name, value, uncertainty] of rows) {
      const row = node('tr')
      const display = value === null ? 'Undefined' : typeof value === 'number' ? evaluationMetric(value, uncertainty) : String(value)
      const cell = node('td', display)
      if (Number.isFinite(uncertainty?.std)) cell.title = 'Statistical uncertainty from event bootstrap replicas'
      row.append(node('td', name), cell)
      body.append(row)
    }
    table.append(head, body)
    wrap.append(table)
    metricDetails.append(wrap)
  } else {
    metricDetails.append(node('p', category?.status === 'comparison' ? 'ROC comparison across bins. Select one bin for its metrics.' : 'Metrics are not available for this selection. Check its status and log.', 'subtle'))
  }
  const downloads = node('div', undefined, 'file-list')
  for (const file of view.files || []) {
    if (!file.path.endsWith('.png') && !file.path.endsWith('.pdf')) downloads.append(link(file.path, file.path.slice(view.directory.length + 1)))
  }
  metricDetails.append(downloads)
  const details = node('details')
  details.append(node('summary', 'Raw metrics and checkpoint details'), node('pre', JSON.stringify(category || current, null, 2)))
  metricDetails.append(details)
  $('evaluation-metrics').append(metricDetails)
  $('evaluations').append(box)
  const figures = (view.files || []).filter(item => /\.(png|pdf)$/.test(item.path))
  const grouped = new Map()
  for (const file of figures) {
    const stem = file.path.replace(/\.(png|pdf)$/, '')
    if (!grouped.has(stem)) grouped.set(stem, {})
    grouped.get(stem)[file.path.split('.').at(-1)] = file
  }
  for (const [stem, files] of grouped) {
    const title = stem.slice(view.directory.length + 1)
    const figure = node('figure')
    if (files.png) {
      const image = node('img')
      image.src = fileURL(files.png.path) + '?t=' + files.png.modified
      image.alt = title
      image.loading = 'lazy'
      const preview = node('button', undefined, 'figure-preview')
      preview.type = 'button'
      preview.setAttribute('aria-label', `Enlarge ${title}`)
      preview.setAttribute('aria-haspopup', 'dialog')
      preview.onclick = () => openFigure(image.src, title)
      preview.append(image)
      figure.append(preview)
    }
    const caption = node('figcaption', title + ' ')
    for (const [format, file] of Object.entries(files)) caption.append(link(file.path, format.toUpperCase() + ' '))
    figure.append(caption)
    $('gallery').append(figure)
  }
  if (!grouped.size) $('gallery').append(node('p', category?.reason || 'No figures saved for this selection. Run evaluation without --no-plots to generate them.', 'subtle'))
}
function formatParameterCount(counts) {
  /*
  Format the model size for the experiment header

  Args:
      counts: Saved total and trainable parameter counts
  Returns:
      text: Exact parameter count for display
  */
  if (!Number.isSafeInteger(counts?.total)) return 'Parameter count unavailable'
  const total = `${counts.total.toLocaleString()} parameters`
  if (Number.isSafeInteger(counts.trainable) && counts.trainable !== counts.total) {
    return `${total} · ${counts.trainable.toLocaleString()} trainable`
  }
  return total
}
function renderFeatureImportance(run) {
  /*
  Load checkpoint rankings when the run has a report

  Args:
      run:  Run record with downloadable output files
  */
  const files = new Map(run.files.map(file => [file.path, file]))
  const reports = ['best', 'last'].flatMap(name => {
    const ranking = files.get(`${run.id}/checkpoints/${name}_feature_importance.json`)
    const pdf = files.get(`${run.id}/plots/${name}_feature_importance.pdf`)
    return ranking ? [{name, ranking, pdf}] : []
  })
  const key = JSON.stringify(reports)
  if (key === featureImportanceKey) return
  featureImportanceKey = key
  featureImportanceRequest?.abort()
  $('feature-importance').hidden = !reports.length
  if (!reports.length) return

  const select = $('importance-checkpoint')
  const filter = $('importance-filter')
  const previous = $('importance-prev')
  const next = $('importance-next')
  const status = $('importance-status')
  const rows = $('importance-rows')
  select.replaceChildren(...reports.map(({name, ranking}) => {
    const option = node('option', `${name}.pt`)
    option.value = ranking.path
    return option
  }))

  const load = async () => {
    featureImportanceRequest?.abort()
    const controller = new AbortController()
    featureImportanceRequest = controller
    const {ranking, pdf} = reports.find(item => item.ranking.path === select.value)
    const links = $('importance-files')
    links.replaceChildren(link(ranking.path, 'Rankings JSON'))
    if (pdf) links.append(link(pdf.path, 'Importance PDF'))
    filter.value = ''
    filter.disabled = previous.disabled = next.disabled = true
    rows.replaceChildren()
    status.textContent = 'Loading feature rankings…'

    try {
      const result = await get(fileURL(ranking.path), controller.signal)
      if (controller.signal.aborted) return
      const maxGain = result.features[0]?.score || 1
      let features = result.features
      let page = 0
      const pageSize = 32
      const showPage = () => {
        const start = page * pageSize
        const shown = features.slice(start, start + pageSize)
        rows.replaceChildren(...shown.map(({rank, feature, score}) => {
          const row = node('tr')
          const gain = node('td')
          const content = node('span', undefined, 'importance-gain')
          const bar = node('span', undefined, 'importance-gain-bar')
          bar.setAttribute('aria-hidden', 'true')
          bar.style.setProperty('--gain-width', `${score / maxGain * 100}%`)
          content.append(bar, node('span', number(score)))
          gain.append(content)
          row.append(node('td', rank.toLocaleString()), node('td', feature), gain)
          return row
        }))
        $('importance-table').scrollTop = 0
        status.textContent = shown.length
          ? `${(start + 1).toLocaleString()}–${(start + shown.length).toLocaleString()} of ${features.length.toLocaleString()} features`
          : 'No matching features'
        previous.disabled = page === 0
        next.disabled = start + shown.length >= features.length
      }
      filter.oninput = () => {
        const query = filter.value.trim().toLowerCase()
        features = result.features.filter(({feature}) => feature.toLowerCase().includes(query))
        page = 0
        showPage()
      }
      previous.onclick = () => {
        page -= 1
        showPage()
      }
      next.onclick = () => {
        page += 1
        showPage()
      }
      filter.disabled = false
      showPage()
    } catch (error) {
      if (!controller.signal.aborted) status.textContent = 'Feature rankings could not be loaded'
    }
  }
  select.onchange = load
  return load()
}
function drawTrainingDistribution(feature, edges, weights, classes, fraction) {
  /*
  Draw the class histograms for one training variable and category

  Args:
      feature:  Driver feature name
      edges:  Shared variable bin edges
      weights:  Total event weights in each bin for each class
      classes:  Driver class names
      fraction:  Show the fraction of each class in a bin
  */
  const chart = $('distribution-plot')
  const logX = $('distribution-x-scale').dataset.scale === 'log10'
  const logY = $('distribution-y-scale').dataset.scale === 'log10'
  const series = fraction ? weights.map(row => {
    const total = row.reduce((sum, value) => sum + value, 0)
    return row.map(value => total ? 100 * value / total : 0)
  }) : weights
  const first = logX ? edges.findIndex((edge, index) => edge > 0 && index < edges.length - 1) : 0
  const note = $('distribution-axis-note')
  const notes = []
  if (first !== 0) notes.push('Log10 X omits bins with nonpositive edges.')
  if (logY && series.some(row => row.includes(0, Math.max(0, first)))) notes.push('Log10 Y omits zero-height bins.')
  note.textContent = notes.join(' ')
  note.hidden = !notes.length
  if (first < 0) {
    chart.replaceChildren(node('p', 'No bins with positive edges. Try a linear X axis.', 'subtle'))
    return
  }
  const positive = series.flatMap(row => row.slice(first).filter(value => value > 0))
  const maximum = positive.reduce((max, value) => Math.max(max, value), 0)
  if (!maximum) {
    chart.replaceChildren(node('p', logX ? 'No weighted events on this X scale. Try a linear X axis.'
      : 'No weighted events in this category', 'subtle'))
    return
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', '0 0 900 380')
  svg.setAttribute('role', 'img')
  svg.setAttribute('aria-label', `${feature} training distributions for ${classes.join(' and ')}, ${logX ? 'log10' : 'linear'} X axis and ${logY ? 'log10' : 'linear'} Y axis`)
  const make = (tag, attributes, content) => {
    const child = document.createElementNS(svg.namespaceURI, tag)
    for (const [key, value] of Object.entries(attributes)) child.setAttribute(key, value)
    if (content !== undefined) child.textContent = content
    svg.append(child)
  }
  const tx = value => logX ? Math.log10(value) : value
  const ty = value => logY ? Math.log10(value) : value
  const low = tx(edges[first])
  const high = tx(edges.at(-1))
  const lower = logY ? ty(positive.reduce((min, value) => Math.min(min, value), Infinity)) - 0.1 : 0
  const upper = logY ? ty(maximum) + 0.1 : maximum * 1.05
  const x = value => 100 + 770 * (value - low) / (high - low)
  const y = value => 320 - 295 * (value - lower) / (upper - lower)
  const tickLabel = (value, log) => {
    const physical = log ? 10 ** value : value
    return physical && (Math.abs(physical) < 1e-3 || Math.abs(physical) >= 1e6) ? scientificNumber(physical) : number(physical)
  }
  const axisStyle = {fill: 'var(--muted)', 'font-size': 15, 'text-anchor': 'middle'}
  for (let tick = 0; tick <= 4; tick += 1) {
    const height = lower + (upper - lower) * tick / 4
    const position = low + (high - low) * tick / 4
    make('line', {x1: 100, x2: 870, y1: y(height), y2: y(height), stroke: 'var(--line)'})
    make('text', {...axisStyle, x: 89, y: y(height) + 5, 'text-anchor': 'end'}, tickLabel(height, logY))
    make('text', {...axisStyle, x: x(position), y: 343}, tickLabel(position, logX))
  }
  series.forEach((row, index) => {
    let path = logY ? '' : `M${x(low)},${y(0)}`
    let drawing = !logY
    for (let bin = first; bin < row.length; bin += 1) {
      if (logY && row[bin] === 0) {
        drawing = false
        continue
      }
      const height = y(ty(row[bin]))
      path += drawing ? ` V${height}` : ` M${x(tx(edges[bin]))},${height}`
      path += ` H${x(tx(edges[bin + 1]))}`
      drawing = true
    }
    if (!path) return
    if (!logY) path += ` V${y(0)}`
    make('path', {d: path, fill: 'none',
      stroke: index === 0 ? 'var(--cyan)' : 'var(--blue)', 'stroke-width': 2})
  })
  make('text', {...axisStyle, x: 485, y: 371}, logX ? `${feature} (log10)` : feature)
  const vertical = fraction ? 'Fraction per class (%)' : 'Weighted events'
  make('text', {...axisStyle, x: 22, y: 173, transform: 'rotate(-90 22 173)'}, logY ? `${vertical} (log10)` : vertical)
  chart.replaceChildren(svg)
}

async function renderTrainingDistributions(run) {
  /*
  Read the available feature names and fetch histogram data on demand

  Args:
      run:  Experiment record and downloadable files
  */
  const stem = `${run.id}/plots/training_distributions`
  const manifest = run.files.find(file => file.path === `${stem}.json`)
  const archive = run.files.find(file => file.path === `${stem}.npz`)
  const available = manifest && archive
  const key = available ? JSON.stringify([manifest, archive]) : ''
  if (key === distributionKey) return
  distributionKey = key
  distributionRequest?.abort()
  $('training-distributions').hidden = !available
  if (!available) return

  const controller = new AbortController()
  distributionRequest = controller
  const filter = $('distribution-filter')
  const feature = $('distribution-feature')
  const category = $('distribution-category')
  const scale = $('distribution-scale')
  const status = $('distribution-status')
  const legend = $('distribution-legend')
  const chart = $('distribution-plot')
  filter.value = ''
  filter.disabled = feature.disabled = category.disabled = true
  legend.replaceChildren()
  chart.replaceChildren()
  $('distribution-axis-note').hidden = true
  status.textContent = 'Loading training variables…'
  let request = null
  controller.signal.addEventListener('abort', () => request?.controller.abort(), {once: true})
  try {
    const meta = await get(fileURL(manifest.path), controller.signal)
    if (controller.signal.aborted) return
    const option = (label, index) => {
      const item = node('option', label)
      item.value = String(index)
      return item
    }
    category.replaceChildren(...meta.categories.map(option))
    $('distribution-category-label').hidden = meta.categories.length < 2
    filter.disabled = category.disabled = false
    let histogram = null
    const draw = () => {
      if (!histogram) return
      const selected = Number(category.value)
      legend.replaceChildren(...meta.classes.map((name, index) => {
        const item = node('span', `${name} · ${meta.events[selected][index].toLocaleString()} events`)
        item.style.setProperty('--class-color', index === 0 ? 'var(--cyan)' : 'var(--blue)')
        return item
      }))
      drawTrainingDistribution(meta.features[Number(histogram.feature)], histogram.edges, histogram.weights[selected],
        meta.classes, scale.value === 'fraction')
    }
    const load = async () => {
      const value = feature.value
      if (request?.feature === value) return
      request?.controller.abort()
      request = null
      if (value === '') {
        status.textContent = 'No matching variables'
        return
      }
      if (histogram?.feature === value) {
        status.textContent = 'Preprocessed training inputs before augmentation'
        return
      }
      const current = new AbortController()
      request = {feature: value, controller: current}
      status.textContent = 'Loading histogram…'
      try {
        const result = await get(`/api/training-distributions/${encodeURIComponent(run.id)}?feature=${value}`, current.signal)
        if (current.signal.aborted) return
        histogram = {...result, feature: value}
        status.textContent = 'Preprocessed training inputs before augmentation'
        draw()
      } catch (error) {
        if (!current.signal.aborted) status.textContent = 'Training histogram could not be loaded'
      } finally {
        if (request?.controller === current) request = null
      }
    }
    const filterVariables = () => {
      const query = filter.value.trim().toLowerCase()
      const value = feature.value
      const options = []
      for (let index = 0; index < meta.features.length; index += 1) {
        if (meta.features[index].toLowerCase().includes(query)) options.push(option(meta.features[index], index))
      }
      feature.replaceChildren(...options)
      feature.disabled = !options.length
      if (options.some(item => item.value === value)) feature.value = value
      return load()
    }
    filter.oninput = filterVariables
    feature.onchange = load
    category.onchange = scale.onchange = draw
    for (const axis of ['distribution-x-scale', 'distribution-y-scale']) bindAxisScale(axis, draw)
    return filterVariables()
  } catch (error) {
    if (!controller.signal.aborted) status.textContent = 'Training variables could not be loaded'
  }
}

function renderDetail(run) {
  if (comparisonMode) return
  $('empty').hidden = true
  $('detail').hidden = false
  $('name').textContent = run.id
  const remove = $('delete-run')
  const force = Boolean(run.delete_blocked)
  remove.textContent = force ? 'Force delete experiment' : 'Delete experiment'
  remove.disabled = deleting || startingEvaluation
  remove.title = force ? 'This experiment has active training or evaluation recorded' : ''
  remove.onclick = () => {
    if (!remove.disabled) return deleteDirectory('run', run.id, run.directory, force)
  }
  $('status').textContent = run.status
  $('status').dataset.status = run.status
  $('status').classList.toggle('error', run.status.includes('failed'))
  $('kind').textContent = `${run.driver} / ${run.model} / ${run.task} · ${formatParameterCount(run.model_parameters)}`
  $('created').replaceChildren()
  for (const [label, value] of [['Created', run.created_at], ['Last training activity', run.last_training_activity]]) {
    const entry = node('span', `${label}: `)
    const time = node('time', formatTime(value))
    if (time.textContent !== '—') time.dateTime = value
    time.title = time.textContent === '—' ? 'Time unavailable' : time.textContent
    entry.append(time)
    $('created').append(entry)
  }
  const latest = run.history.at(-1) || {}
  if (Number.isFinite(latest.step)) $('created').append(node('span', `Optimizer steps: ${formatSteps(latest.step)}`))
  $('created').append(node('span', `Total training time: ${formatDuration(run.training_elapsed_seconds)}`))
  const {last, best} = validationLosses(run.history)
  $('cards').replaceChildren()
  for (const [label, value] of [['Completed epochs', run.history.length], ['Training loss', latest.train_loss],
    ['Last validation loss', last], ['Best validation loss', best], ['Learning rate', latest.lr]]) {
    $('cards').append(metricCard(label, label === 'Learning rate' ? scientificNumber(value) : number(value)))
  }
  renderCurves(run.history)
  $('checkpoints').replaceChildren()
  $('selection').textContent = `Selection: ${run.checkpoints.monitor || '—'} / ${run.checkpoints.mode || '—'}`
  for (const checkpoint of run.checkpoints.checkpoints || []) {
    const row = node('tr')
    const first = node('td')
    first.append(link(`${run.id}/checkpoints/${checkpoint.file}`, checkpoint.file))
    row.append(first, node('td', number(checkpoint.epoch == null ? null : checkpoint.epoch + 1)), node('td', number(checkpoint.metric)), node('td', `${number(checkpoint.bytes / 1048576)} MB`))
    const action = node('td', undefined, 'checkpoint-action')
    const evaluate = node('button', 'Run evaluation', 'command-button evaluation-launch')
    evaluate.type = 'button'
    evaluate.setAttribute('aria-label', `Run test evaluation for ${checkpoint.file} on the Twistboard host`)
    evaluate.title = 'Evaluate the test sample on the Twistboard host'
    evaluate.disabled = deleting || startingEvaluation || run.evaluations.some(item => ['queued', 'evaluating'].includes(item.status))
    evaluate.onclick = () => {
      if (!evaluate.disabled) return runEvaluation(run.id, checkpoint.file)
    }
    action.append(evaluate)
    const button = node('button', 'Evaluation command', 'command-button')
    button.type = 'button'
    button.setAttribute('aria-label', `Evaluation command for ${checkpoint.file}`)
    button.setAttribute('aria-haspopup', 'dialog')
    button.onclick = () => showEvaluationCommand(run.command_directory || run.directory, checkpoint.file)
    action.append(button)
    if (typeof addComparisonCheckpoint === 'function') addComparisonCheckpoint(action, run, checkpoint)
    row.append(action)
    $('checkpoints').append(row)
  }
  renderFeatureImportance(run)
  renderTrainingDistributions(run)
  $('training-commands').replaceChildren()
  const attempts = run.attempts || []
  if (!attempts.length) $('training-commands').append(node('p', 'Training command was not recorded for this run.', 'subtle'))
  attempts.forEach((attempt, index) => {
    const entry = node('div')
    entry.append(node('p', `Attempt ${index + 1}${attempt.resume ? ' - Resume' : ''} - ${formatTime(attempt.started_at)}`, 'subtle'))
    if (attempt.command) {
      entry.append(node('p', `Working directory: ${attempt.working_directory || 'Unknown'}`, 'subtle'))
      const row = node('div', undefined, 'training-command')
      const command = node('pre', attempt.command)
      command.tabIndex = 0
      const copy = node('button', undefined, 'command-button')
      copy.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M9 5H6a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3"/><rect x="9" y="2" width="6" height="6" rx="1"/></svg>'
      copy.type = 'button'
      copy.title = 'Copy training command'
      copy.setAttribute('aria-label', `Copy training command for attempt ${index + 1}`)
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(attempt.command)
          showActionMessage('Training command copied')
        } catch {
          command.focus({preventScroll: true})
          const range = document.createRange()
          range.selectNodeContents(command)
          const selection = window.getSelection()
          selection.removeAllRanges()
          selection.addRange(range)
          showActionMessage('Command selected. Press Ctrl+C or Cmd+C to copy')
        }
      }
      row.append(command, copy)
      entry.append(row)
    } else {
      entry.append(node('p', 'Training command was not recorded for this attempt.', 'subtle'))
    }
    $('training-commands').append(entry)
  })
  renderEvaluations(run)
  $('files').replaceChildren()
  for (const file of run.files) {
    $('files').append(link(file.path, file.path))
  }
  $('configuration').textContent = JSON.stringify(run.configuration, null, 2)
  $('log').textContent = run.log || run.error || 'No training log'
}
function showActionMessage(text, error = false) {
  /*
  Show host action status for five seconds in a dismissible dropdown

  Args:
      text:  Action progress, result or failure
      error:  Mark a failed action
  */
  clearTimeout(actionMessageTimer)
  $('action-message').textContent = text
  $('action-notification').className = error ? 'error' : ''
  $('action-notification').hidden = false
  actionMessageTimer = setTimeout(dismissActionMessage, 5000)
}
function dismissActionMessage() {
  /*
  Close the action dropdown and retain keyboard access
  */
  clearTimeout(actionMessageTimer)
  actionMessageTimer = null
  $('action-notification').hidden = true
  $('action-message').textContent = ''
  if (document.activeElement === $('action-close')) $('search').focus({preventScroll: true})
}
async function runEvaluation(runId, checkpoint) {
  /*
  Start test evaluation on the dashboard host through the browser connection

  Args:
      runId: Experiment path relative to the dashboard root
      checkpoint: Selected checkpoint file
  Returns:
      None: Host job status and evaluation results are refreshed
  */
  if (startingEvaluation || deleting) return
  startingEvaluation = true
  const controls = [...document.querySelectorAll('.evaluation-launch, .delete-button')].map(control => [control, control.disabled])
  for (const [control] of controls) control.disabled = true
  showActionMessage(`Starting evaluation of ${checkpoint} on the Twistboard host…`)
  try {
    const response = await fetch('/api/evaluations', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Twistboard-Action': 'evaluate'},
      body: JSON.stringify({checkpoint: `${runId}/checkpoints/${checkpoint}`})
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || 'Unable to start evaluation')
    evaluationSelection.set(runId, result.id)
    evaluationKey = null
    showActionMessage(`Evaluation started on the Twistboard host: ${checkpoint}`)
  } catch (error) {
    showActionMessage(error.message, true)
  } finally {
    startingEvaluation = false
    for (const [control, disabled] of controls) control.disabled = disabled
    await refresh()
  }
}

async function deleteDirectory(kind, id, path, force = false) {
  /*
  Remove the selected file directory after folder confirmation

  Args:
      kind:  Run or evaluation
      id:  Directory relative to the dashboard root
      path:  Absolute directory shown in the confirmation
      force:  Allow removal of an experiment with active work recorded
  */
  if (deleting || startingEvaluation) return
  const label = kind === 'run' ? 'experiment' : 'evaluation'
  const action = force ? 'Force delete' : 'Delete'
  const activeWork = force ? '\n\nTwistboard will stop any evaluation it launched for this experiment. Other jobs may keep running and write new files.' : ''
  if (!window.confirm(`${action} ${label} "${id}"?\n\nFolder: ${path}\n\nThis permanently removes the folder and its contents.${activeWork}`)) return
  deleting = true
  refreshController?.abort()
  refreshController = null
  const controls = [...document.querySelectorAll('.delete-button, button.run, #evaluation-choice')].map(control => [control, control.disabled])
  for (const [control] of controls) control.disabled = true
  showActionMessage(`Deleting ${label}…`)
  try {
    const endpoint = kind === 'run' ? 'runs' : 'evaluations'
    const response = await fetch(`/api/${endpoint}/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      headers: {'X-Twistboard-Action': force ? 'force-delete' : 'delete'}
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || `Unable to delete ${label}`)
    if (kind === 'run') {
      evaluationSelection.delete(id)
      if (selected === id) {
        selected = ''
        history.replaceState(null, '', location.pathname + location.search)
        $('detail').hidden = true
      }
    } else {
      for (const [run, directory] of evaluationSelection) {
        if (directory === id) evaluationSelection.delete(run)
      }
    }
    evaluationKey = null
    showActionMessage(`Deleted ${label}: ${id}`)
  } catch (error) {
    showActionMessage(error.message, true)
  } finally {
    deleting = false
    for (const [control, disabled] of controls) control.disabled = disabled
    await refresh()
  }
}
async function refresh() {
  if (window.twistExplorer) return window.twistExplorer.refresh()
  if (refreshController || deleting) return
  const controller = new AbortController()
  refreshController = controller
  try {
    const nextRuns = await get('/api/runs', controller.signal)
    if (controller.signal.aborted) return
    runs = nextRuns
    if (!runs.some(run => run.id === selected)) selected = runs[0]?.id || ''
    renderList()
    if (selected) {
      const id = selected
      const run = await get('/api/runs/' + encodeURIComponent(id), controller.signal)
      if (controller.signal.aborted || selected !== id) return
      renderDetail(run)
    }
    else if (!comparisonMode) { $('empty').hidden = false; $('detail').hidden = true }
    $('connection').textContent = '● Live'
    $('connection').classList.remove('error')
  } catch (error) {
    if (controller.signal.aborted || error.name === 'AbortError') return
    $('connection').textContent = error.message
    $('connection').classList.add('error')
  } finally {
    if (refreshController === controller) refreshController = null
  }
}
$('search').oninput = renderList
$('action-close').onclick = dismissActionMessage
$('search').addEventListener('keydown', navigateRuns)
$('runs').addEventListener('keydown', navigateRuns)
$('command-close').onclick = () => $('command-viewer').close()
$('command-copy').onclick = async () => {
  try {
    await navigator.clipboard.writeText($('evaluation-command').value)
    $('command-copy-status').textContent = 'Copied'
  } catch {
    $('evaluation-command').focus()
    $('evaluation-command').select()
    $('command-copy-status').textContent = 'Press Ctrl+C or Cmd+C to copy'
  }
}
$('figure-close').onclick = closeFigure
$('figure-viewer').addEventListener('cancel', event => {
  event.preventDefault()
  closeFigure()
})
$('figure-viewer').addEventListener('close', () => {
  if ($('figure-viewer').open) return
  if (figureAnimation) figureAnimation.cancel()
  figureAnimation = null
  document.body.classList.remove('figure-open')
  $('figure-image').removeAttribute('src')
})
function setAxisScale(id, scale) {
  /*
  Select one axis scale and update its toggle buttons

  Args:
      id:  Axis control group ID
      scale:  Linear or log10 scale
  Returns:
      None: The selected scale and accessible button states are updated
  */
  const group = $(id)
  group.dataset.scale = scale === 'log10' ? 'log10' : 'linear'
  for (const button of group.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', button.dataset.scale === group.dataset.scale)
  }
}
function bindAxisScale(id, redraw) {
  /*
  Restore an axis preference and redraw its plot when the scale changes

  Args:
      id: Axis control group ID
      redraw: Plot update function
  */
  setAxisScale(id, readPreference(id))
  for (const button of $(id).querySelectorAll('button')) {
    button.onclick = () => {
      setAxisScale(id, button.dataset.scale)
      writePreference(id, $(id).dataset.scale)
      redraw()
    }
  }
}
for (const id of axisControls) bindAxisScale(id, () => renderCurves(curveRows))
function initializeBlocks() {
  /*
  Add persistent collapse controls to dashboard sections

  Returns:
      None: Each block keeps its heading visible while minimized
  */
  for (const section of document.querySelectorAll('[data-block]')) {
    const heading = section.querySelector('h2')
    const title = heading.textContent
    const header = section.querySelector(':scope > .section-head') || node('div', undefined, 'section-head')
    if (!header.parentElement) {
      header.append(heading)
      section.prepend(header)
    }
    const body = node('div', undefined, 'block-content')
    body.id = `block-${section.dataset.block}`
    const titleButton = node('button', title, 'block-title')
    titleButton.type = 'button'
    titleButton.setAttribute('aria-controls', body.id)
    heading.replaceChildren(titleButton)
    for (const child of [...section.childNodes]) {
      if (child !== header) body.append(child)
    }
    const button = node('button', undefined, 'command-button block-toggle')
    button.type = 'button'
    button.setAttribute('aria-controls', body.id)
    const key = `block.${section.dataset.block}.collapsed`
    let collapsed = false
    let animation = null
    const update = (next, animate = false) => {
      const height = body.getBoundingClientRect().height
      const opacity = body.hidden ? 0 : Number(getComputedStyle(body).opacity)
      if (animation) animation.cancel()
      collapsed = next
      section.classList.toggle('block-collapsed', collapsed)
      button.textContent = collapsed ? '+' : '−'
      button.title = `${collapsed ? 'Expand' : 'Minimize'} ${title}`
      button.setAttribute('aria-expanded', String(!collapsed))
      button.setAttribute('aria-label', button.title)
      titleButton.setAttribute('aria-expanded', String(!collapsed))
      titleButton.title = button.title
      body.inert = collapsed
      if (!animate || reducedMotion.matches) {
        body.hidden = collapsed
        body.style.overflow = ''
        animation = null
        return
      }
      body.hidden = false
      body.style.overflow = 'hidden'
      animation = body.animate([
        {height: `${height}px`, opacity},
        {height: `${collapsed ? 0 : body.scrollHeight}px`, opacity: collapsed ? 0 : 1}
      ], {duration: 240, easing: 'ease-in-out'})
      animation.onfinish = () => {
        body.hidden = collapsed
        body.style.overflow = ''
        animation = null
      }
    }
    collapsed = readPreference(key) === 'true'
    update(collapsed)
    button.onclick = () => {
      update(!collapsed, true)
      writePreference(key, collapsed)
    }
    titleButton.onclick = button.onclick
    header.append(button)
    section.append(body)
  }
}
initializeBlocks()
function setSidebarHidden(hidden) {
  /*
  Slide the experiment list horizontally and update keyboard access

  Args:
      hidden:  Hide the sidebar when true
  Returns:
      None: The main view expands into the released space
  */
  const sidebar = $('experiment-sidebar')
  const button = $('sidebar-toggle')
  if (hidden && (sidebar.contains(document.activeElement) || document.activeElement === $('sidebar-resize'))) button.focus()
  sidebar.inert = hidden
  sidebar.setAttribute('aria-hidden', String(hidden))
  document.body.classList.toggle('sidebar-hidden', hidden)
  button.setAttribute('aria-expanded', String(!hidden))
  button.title = hidden ? 'Show experiments' : 'Hide experiments'
  button.setAttribute('aria-label', button.title)
}
setSidebarHidden(readPreference('sidebar.hidden') === 'true')
$('sidebar-toggle').onclick = () => {
  const hidden = !document.body.classList.contains('sidebar-hidden')
  setSidebarHidden(hidden)
  writePreference('sidebar.hidden', hidden)
}
const sidebarLayout = document.querySelector('.layout')
const sidebarResize = $('sidebar-resize')
let sidebarDrag = null
function updateSidebarWidth(width) {
  /*
  Apply a sidebar width within the available viewport

  Args:
      width: Preferred width in pixels, or undefined to retain the current width
  */
  const maximum = Math.min(600, window.innerWidth * 0.55)
  if (Number.isFinite(width)) sidebarLayout.style.setProperty('--sidebar-preferred-width', `${Math.max(200, Math.min(maximum, width))}px`)
  sidebarResize.setAttribute('aria-valuemin', '200')
  sidebarResize.setAttribute('aria-valuemax', Math.floor(maximum))
  sidebarResize.setAttribute('aria-valuenow', Math.round($('experiment-sidebar').getBoundingClientRect().width))
}
function finishSidebarResize() {
  /*
  Finish a pointer resize and retain the preferred width
  */
  if (!sidebarDrag) return
  const pointerId = sidebarDrag.pointerId
  sidebarDrag = null
  if (sidebarResize.hasPointerCapture(pointerId)) sidebarResize.releasePointerCapture(pointerId)
  document.body.classList.remove('sidebar-resizing')
  const width = parseFloat(sidebarLayout.style.getPropertyValue('--sidebar-preferred-width'))
  if (Number.isFinite(width)) writePreference('sidebar.width', width)
}
const savedSidebarWidth = parseFloat(readPreference('sidebar.width'))
if (Number.isFinite(savedSidebarWidth)) sidebarLayout.style.setProperty('--sidebar-preferred-width', `${Math.max(200, Math.min(600, savedSidebarWidth))}px`)
updateSidebarWidth()
sidebarResize.onpointerdown = event => {
  if (event.button !== 0) return
  event.preventDefault()
  sidebarResize.focus({preventScroll: true})
  sidebarDrag = {pointerId: event.pointerId, x: event.clientX, width: $('experiment-sidebar').getBoundingClientRect().width}
  document.body.classList.add('sidebar-resizing')
  sidebarResize.setPointerCapture(event.pointerId)
}
sidebarResize.onpointermove = event => {
  if (sidebarDrag?.pointerId === event.pointerId) updateSidebarWidth(sidebarDrag.width + event.clientX - sidebarDrag.x)
}
sidebarResize.onpointerup = finishSidebarResize
sidebarResize.onpointercancel = finishSidebarResize
sidebarResize.onlostpointercapture = finishSidebarResize
sidebarResize.ondblclick = () => {
  sidebarLayout.style.removeProperty('--sidebar-preferred-width')
  writePreference('sidebar.width', '')
  updateSidebarWidth()
}
sidebarResize.onkeydown = event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const current = $('experiment-sidebar').getBoundingClientRect().width
  const step = event.shiftKey ? 40 : 10
  updateSidebarWidth(event.key === 'Home' ? 200 : event.key === 'End' ? 600 : current + (event.key === 'ArrowLeft' ? -step : step))
  writePreference('sidebar.width', parseFloat(sidebarLayout.style.getPropertyValue('--sidebar-preferred-width')))
}
window.addEventListener('resize', () => updateSidebarWidth())
window.addEventListener('hashchange', () => selectRun(decodeURIComponent(location.hash.slice(1))))
window.addEventListener('DOMContentLoaded', refresh)
setInterval(refresh, 5000)
