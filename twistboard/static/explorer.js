// Parameter scans and arbitrary training observables
//
// mikael.mieskolainen@cern.ch (2026)

(() => {
  const palette = ['#59d6c8', '#80a9ff', '#ff9b9b', '#e6a3dc', '#e3ce77', '#a8dc86']
  const state = {open: false, offset: 0, filters: {}, selected: new Set(), data: null, request: null, treeKey: '', tree: [], observable: null, hits: [], axes: []}
  let searchTimer = null
  let plotFrame = null
  let curveRequest = null
  let observableRequest = null
  let observableKey = null
  const colorDomains = new WeakMap()
  const logAxes = new Set()

  function scanControls() {
    /*
    Show the controls used by the current scan projection
    */
    const parallel = $('scan-plot').value === 'parallel'
    $('scan-x-label').hidden = parallel
    $('scan-y-label').hidden = parallel
    $('scan-axis-label').hidden = !parallel
  }

  function options(id, values, preferred) {
    /*
    Retain a valid control selection while updating available values

    Args:
        id: Select element identifier
        values: Value and label pairs
        preferred: Preferred selected value
    */
    const control = $(id)
    const current = preferred ?? control.value
    const key = JSON.stringify(values)
    if (control.dataset.options !== key) {
      control.replaceChildren(...values.map(([value, label]) => {
        const option = node('option', label)
        option.value = value
        return option
      }))
      control.dataset.options = key
    }
    if (values.some(([value]) => value === current)) control.value = current
  }

  function query(extra = {}) {
    /*
    Build a query from the shared driver and scan controls

    Args:
        extra: Additional query fields
    Returns:
        query: Encoded query string
    */
    return new URLSearchParams({q: $('search').value, driver: $('scan-driver').value, group_id: $('scan-group').value,
      status: $('scan-status').value, metric: $('scan-metric').value, mode: $('scan-mode').value, reduce: $('scan-reduce').value,
      budget: Math.round(Number($('scan-budget').value) * 1000), filters: JSON.stringify(state.filters), offset: state.offset, limit: 50, sort: 'metric', ...extra}).toString()
  }

  async function loadBranch(container, branch, offset = 0) {
    /*
    Fetch only one visible page of a tree branch

    Args:
        container: Branch content element
        branch: Driver and group metadata
        offset: First run in the branch page
    */
    const token = String(Date.now()) + Math.random()
    container.dataset.request = token
    try {
      const result = await get('/api/explorer?' + new URLSearchParams({driver: branch.driver, group_id: branch.group_id, q: $('search').value, offset, limit: 30}))
      if (!container.isConnected || container.dataset.request !== token) return
      const key = JSON.stringify([offset, result.total, selected, deleting, result.rows.map(run => [run.id, run.model, run.status, run.epochs, run.training_started_at, run.created_at])])
      if (container.dataset.content === key) return
      container.dataset.content = key
      const focused = container.contains(document.activeElement) ? document.activeElement.dataset.runId : null
      container.replaceChildren()
      const scan = node('button', 'View group scan', 'command-button tree-scan-link')
      scan.onclick = () => openScan(branch.driver, branch.group_id)
      container.append(scan)
      for (const run of result.rows) {
        const button = node('button', undefined, 'run' + (run.id === selected ? ' active' : ''))
        button.dataset.runId = run.id
        button.dataset.status = run.status
        button.disabled = deleting
        button.append(node('strong', run.id), node('small', `${run.model || 'Unknown model'} · ${run.status}: ${run.epochs} epochs`, 'state'))
        const started = run.training_started_at || run.created_at
        const date = node('small', formatTime(started))
        date.title = `Latest training start: ${formatTime(started)}`
        button.append(date)
        button.onclick = () => activateRun(button)
        container.append(button)
        if (run.id === focused) button.focus({preventScroll: true})
      }
      if (offset > 0 || result.rows.length < result.total) {
        const pager = node('div', undefined, 'scan-pager')
        for (const [label, next, disabled] of [['‹', offset - 30, offset === 0], ['›', offset + 30, offset + 30 >= result.total]]) {
          const button = node('button', label, 'command-button')
          button.disabled = disabled
          button.setAttribute('aria-label', label === '‹' ? 'Previous experiments' : 'Next experiments')
          button.onclick = () => loadBranch(container, branch, next)
          pager.append(button)
        }
        pager.append(node('span', `${offset + 1}–${Math.min(offset + 30, result.total)} / ${result.total}`))
        container.append(pager)
      }
      container.dataset.offset = offset
    } catch (error) { container.textContent = error.message }
  }

  function renderTree(result) {
    /*
    Render driver branches and lazily load expanded groups

    Args:
        result: Tree counts and index revision
    */
    state.tree = result.branches
    $('count').textContent = result.total.toLocaleString()
    const filtering = !!$('search').value
    const key = JSON.stringify([result.branches, $('search').value])
    if (key === state.treeKey) {
      for (const container of $('runs').querySelectorAll('.tree-content')) {
        if (container.parentElement.open && container.parentElement.parentElement.open && container.dataset.revision !== String(result.revision)) {
          loadBranch(container, JSON.parse(container.dataset.branch), Number(container.dataset.offset || 0))
          container.dataset.revision = result.revision
        }
      }
      for (const button of $('runs').querySelectorAll('button.run')) button.classList.toggle('active', button.dataset.runId === selected)
      return
    }
    state.treeKey = key
    const drivers = new Map()
    for (const branch of result.branches) {
      if (!drivers.has(branch.driver)) drivers.set(branch.driver, [])
      drivers.get(branch.driver).push(branch)
    }
    const scroll = $('runs').scrollTop
    $('runs').replaceChildren()
    for (const [driver, branches] of [...drivers].sort(([a], [b]) => a.localeCompare(b))) {
      const group = node('details', undefined, 'tree-driver')
      const appKey = `tree.driver.${driver}`
      group.open = readPreference(appKey) === 'true' || filtering
      group.append(node('summary', `${driver} (${branches.reduce((sum, branch) => sum + branch.count, 0)})`))
      for (const branch of branches.sort((a, b) => a.group_id.localeCompare(b.group_id))) {
        const detail = node('details', undefined, 'tree-group')
        const preference = `tree.group.${driver}.${branch.group_id}`
        detail.open = readPreference(preference) === 'true' || filtering
        const summary = node('summary', `${branch.group_id} (${branch.count})`)
        summary.title = Object.entries(branch.statuses).map(([status, count]) => `${status}: ${count}`).join(', ')
        const content = node('div', undefined, 'tree-content')
        content.dataset.branch = JSON.stringify(branch)
        detail.append(summary, content)
        detail.ontoggle = () => {
          if (!detail.isConnected) return
          if (!filtering && !$('search').value) writePreference(preference, detail.open)
          if (detail.open && group.open) loadBranch(content, branch)
        }
        group.append(detail)
        if (detail.open && group.open) loadBranch(content, branch)
      }
      group.ontoggle = () => {
        if (!group.isConnected) return
        if (!filtering && !$('search').value) writePreference(appKey, group.open)
        if (group.open) for (const content of group.querySelectorAll('.tree-content')) {
          if (content.parentElement.open) loadBranch(content, JSON.parse(content.dataset.branch))
        }
      }
      $('runs').append(group)
    }
    $('runs').scrollTop = scroll
  }

  async function refreshExplorer() {
    /*
    Refresh summaries and only the visible detail or scan view
    */
    if (refreshController || deleting) return
    const controller = new AbortController()
    refreshController = controller
    try {
      const tree = await get('/api/explorer?' + new URLSearchParams({tree: 1, q: $('search').value}), controller.signal)
      if (controller.signal.aborted) return
      if (!selected && !state.open && !comparisonMode && tree.total) {
        const latest = await get('/api/explorer?' + new URLSearchParams({limit: 1, q: $('search').value}), controller.signal)
        if (controller.signal.aborted) return
        const run = latest.rows[0]
        if (run) {
          selected = run.id
          history.replaceState(null, '', '#' + encodeURIComponent(selected))
          writePreference(`tree.driver.${run.driver}`, true)
          writePreference(`tree.group.${run.driver}.${run.group_id}`, true)
          state.treeKey = ''
        }
      }
      renderTree(tree)
      options('scan-driver', [['', 'All drivers'], ...[...new Set(tree.branches.map(branch => branch.driver))].sort().map(value => [value, value])])
      const groups = tree.branches.filter(branch => !$('scan-driver').value || branch.driver === $('scan-driver').value)
      options('scan-group', [['', 'All groups'], ...[...new Set(groups.map(branch => branch.group_id))].sort().map(value => [value, value])])
      if (state.open) {
        if (state.scanRevision !== tree.revision) await loadScan()
      }
      else if (selected && !comparisonMode) {
        const id = selected
        const run = await get('/api/runs/' + encodeURIComponent(id), controller.signal)
        if (controller.signal.aborted || selected !== id) return
        renderDetail(run)
        await loadObservable(id)
      } else if (!comparisonMode) openScan()
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

  function openScan(driver, group) {
    /*
    Open the scan view at an optional driver and run group

    Args:
        driver: Optional driver filter
        group: Optional run group filter
    */
    comparisonMode = false
    state.open = true
    state.offset = 0
    state.filters = {}
    $('detail').hidden = true
    $('empty').hidden = true
    $('comparison-view').hidden = true
    $('scan-view').hidden = false
    if (driver !== undefined) {
      $('scan-driver').value = driver
      options('scan-group', [['', 'All groups'], ...state.tree.filter(branch => branch.driver === driver).map(branch => [branch.group_id, branch.group_id])], group)
    }
    loadScan()
  }

  async function loadScan() {
    /*
    Load a bounded trial page and the selected plot dimensions
    */
    state.request?.abort()
    scanControls()
    $('scan-budget-label').hidden = $('scan-reduce').value !== 'budget'
    if (!$('scan-budget-label').hidden && !$('scan-budget').reportValidity()) return
    const controller = new AbortController()
    state.request = controller
    try {
      const dimensions = [...new Set([$('scan-x').value, $('scan-y').value, ...[...$('scan-axes').selectedOptions].map(option => option.value)])].filter(Boolean).slice(0, 10)
      const data = await get('/api/explorer?' + query({plot: 1, axes: dimensions.join(','), kind: $('scan-plot').value}), controller.signal)
      if (controller.signal.aborted || !state.open) return
      state.data = data
      state.scanRevision = data.revision
      options('scan-metric', data.metrics.map(value => [value, value]), data.metric)
      const parameters = Object.keys(data.parameters)
      const preferred = parameters.filter(key => key !== 'training.seed')
      options('scan-x', parameters.map(value => [value, value]), $('scan-x').value || (parameters.includes('training.learning_rate') ? 'training.learning_rate' : preferred[0]))
      options('scan-y', parameters.map(value => [value, value]), $('scan-y').value || (parameters.includes('model.width') ? 'model.width' : preferred[1] || preferred[0]))
      const axisControl = $('scan-axes')
      if (axisControl.dataset.options !== JSON.stringify(parameters)) {
        const chosen = new Set([...axisControl.selectedOptions].map(option => option.value))
        const defaults = [...new Set(['training.learning_rate', 'model.width', 'model.depth', 'training.optimizer', 'training.seed', ...parameters])].filter(key => parameters.includes(key)).slice(0, 5)
        axisControl.replaceChildren(...parameters.map((key, index) => {
          const option = node('option', key)
          option.value = key
          option.selected = chosen.size ? chosen.has(key) : defaults.includes(key)
          return option
        }))
        axisControl.dataset.options = JSON.stringify(parameters)
      }
      const needed = [...new Set([$('scan-x').value, $('scan-y').value, ...[...axisControl.selectedOptions].map(option => option.value)])].filter(Boolean).slice(0, 10)
      if (needed.some(key => !dimensions.includes(key))) return loadScan()
      const missing = data.points.filter(point => !Number.isFinite(point.value)).length
      $('scan-summary').textContent = `${data.total.toLocaleString()} trials · ${data.points.length.toLocaleString()} plotted records${data.sampled ? ' (deterministic sample)' : ''} · ${missing} missing metric values · colour: ${$('scan-plot').value === 'scatter' ? $('scan-y').value : data.metric}`
      $('scan-filters').textContent = Object.keys(state.filters).length ? `Active ranges: ${JSON.stringify(state.filters)}` : 'No range filters'
      renderTable()
      drawScan()
      loadCurves()
    } catch (error) {
      if (error.name !== 'AbortError') $('scan-summary').textContent = error.message
    } finally {
      if (state.request === controller) state.request = null
    }
  }

  function toggleTrial(id) {
    /*
    Toggle a trial in the bounded learning-curve selection

    Args:
        id: Run identifier
    */
    if (state.selected.has(id)) state.selected.delete(id)
    else if (state.selected.size < 12) state.selected.add(id)
    else return showActionMessage('Select at most twelve learning curves')
    renderTable()
    drawScan()
    loadCurves()
  }

  function renderTable() {
    /*
    Render one ranked page with selection linked to the plots
    */
    const data = state.data
    if (!data) return
    const keys = Object.keys(data.parameters)
    const table = node('table')
    const head = node('tr')
    for (const label of ['Select', 'Trial', 'Status', data.metric, 'Time', ...keys]) head.append(node('th', label))
    table.append(head)
    const reduction = $('scan-reduce').value === 'best' ? $('scan-mode').value : $('scan-reduce').value
    for (const run of data.rows) {
      const row = node('tr')
      const select = node('input')
      select.type = 'checkbox'
      select.checked = state.selected.has(run.id)
      select.setAttribute('aria-label', `Select ${run.id}`)
      select.onchange = () => toggleTrial(run.id)
      const cell = node('td')
      cell.append(select)
      const title = node('td')
      const open = node('button', run.id, 'command-button')
      open.onclick = () => { location.hash = encodeURIComponent(run.id); selectRun(run.id) }
      title.append(open)
      row.append(cell, title, node('td', run.status), node('td', number(run.metrics[data.metric]?.[reduction])), node('td', formatDuration(run.elapsed_seconds)))
      for (const key of keys) row.append(node('td', String(run.parameters[key] ?? '—')))
      table.append(row)
    }
    $('scan-table').replaceChildren(table)
    $('scan-selection').textContent = `${state.selected.size} / 12 trials selected for learning curves`
    $('scan-page').textContent = data.total ? `${state.offset + 1}–${Math.min(state.offset + 50, data.total)} / ${data.total}` : 'No matching trials'
    $('scan-prev').disabled = state.offset === 0
    $('scan-next').disabled = state.offset + 50 >= data.total
  }

  function canvasFrame(id) {
    /*
    Prepare a canvas at device resolution with fixed plot margins

    Args:
        id: Canvas identifier
    Returns:
        frame: Drawing context and plot dimensions
    */
    const canvas = $(id)
    const width = Math.max(320, canvas.clientWidth)
    const height = 360
    const ratio = window.devicePixelRatio || 1
    canvas.width = Math.round(width * ratio)
    canvas.height = Math.round(height * ratio)
    const ctx = canvas.getContext('2d')
    ctx.scale(ratio, ratio)
    ctx.font = '12px system-ui'
    ctx.fillStyle = '#8e9bb0'
    ctx.strokeStyle = '#283244'
    return {canvas, ctx, width, height, left: 76, right: width - 28, top: 32, bottom: height - 72}
  }

  function scale(values, start, end, log = false) {
    /*
    Map numeric or categorical values to plot coordinates

    Args:
        values: Axis observations
        start: Lower plot coordinate
        end: Upper plot coordinate
        log: Use a logarithmic numeric scale
    Returns:
        axis: Coordinate transform, tick values and inverse range filter
    */
    const present = values.filter(value => value !== null && value !== undefined && (typeof value !== 'number' || Number.isFinite(value)))
    const numeric = present.length > 0 && present.every(value => typeof value === 'number' && Number.isFinite(value))
    const valid = numeric ? present.filter(value => !log || value > 0) : present
    let domain = [...new Set(valid)].sort(numeric ? (a, b) => a - b : (a, b) => String(a).localeCompare(String(b)))
    const transform = value => log && numeric ? Math.log10(value) : value
    const low = numeric ? transform(domain[0] ?? (log ? 1 : 0)) : 0
    const high = numeric ? transform(domain.at(-1) ?? 1) : Math.max(1, domain.length - 1)
    const span = high - low || 1
    const map = value => {
      if (value == null || (numeric && (!Number.isFinite(value) || (log && value <= 0)))) return NaN
      const coordinate = numeric ? transform(value) : domain.indexOf(value)
      if (!numeric && coordinate < 0) return NaN
      return high === low ? (start + end) / 2 : start + (coordinate - low) / span * (end - start)
    }
    if (numeric && domain.length > 6) domain = Array.from({length: 5}, (_, index) => log ? 10 ** (low + index * span / 4) : low + index * span / 4)
    const range = (a, b) => {
      if (!numeric) return valid.filter((value, index) => valid.indexOf(value) === index && map(value) >= Math.min(a, b) && map(value) <= Math.max(a, b))
      const inverse = pixel => {
        const value = high === low ? low : low + (pixel - start) / (end - start) * span
        return log ? 10 ** value : value
      }
      return {min: Math.min(inverse(a), inverse(b)), max: Math.max(inverse(a), inverse(b))}
    }
    return {map, ticks: domain, numeric, range}
  }

  function scanScale(key, values, start, end) {
    /*
    Apply the saved scale for one parameter or metric

    Args:
        key: Parameter name or metric axis key
        values: Axis observations
        start: First plot coordinate
        end: Last plot coordinate
    Returns:
        axis: Scale with log availability and preference key
    */
    const preference = JSON.stringify([key === '$metric' ? 'metric' : 'parameter', key === '$metric' ? state.data.metric : key])
    const linear = scale(values, start, end)
    const canLog = linear.numeric && values.some(value => Number.isFinite(value) && value > 0)
    const log = canLog && logAxes.has(preference)
    return {...(log ? scale(values, start, end, true) : linear), preference, canLog, log}
  }

  function axisButton(frame, key, axis, x, y, width, align = 'center') {
    /*
    Place a keyboard-accessible scale toggle at an axis label

    Args:
        frame: Canvas drawing frame
        key: Parameter name or metric axis key
        axis: Axis scale and log availability
        x: Horizontal label anchor
        y: Upper label coordinate
        width: Maximum label width
        align: Label anchor alignment
    */
    const name = key === '$metric' ? state.data.metric : key
    const button = node('button', undefined, 'scan-axis-button')
    button.type = 'button'
    button.dataset.axis = key
    button.disabled = !axis.canLog
    button.setAttribute('aria-pressed', String(axis.log))
    button.setAttribute('aria-label', `${name}: ${axis.log ? 'log' : 'linear'} scale`)
    button.title = axis.canLog ? `${name}: switch to ${axis.log ? 'linear' : 'log'} scale. Log scales show positive values only` : `${name}: log scale needs positive numeric values`
    button.style.left = `${100 * x / frame.width}%`
    button.style.top = `${100 * y / frame.height}%`
    button.style.width = `${100 * width / frame.width}%`
    button.style.transform = `translateX(${align === 'left' ? 0 : align === 'right' ? -100 : -50}%)`
    button.append(node('span', name), node('small', axis.numeric ? axis.log ? 'log' : 'linear' : 'category'))
    button.onclick = () => {
      if (!axis.canLog) return
      if (axis.log) logAxes.delete(axis.preference)
      else logAxes.add(axis.preference)
      drawScan()
      for (const control of $('scan-axis-buttons').children) if (control.dataset.axis === key) control.focus({preventScroll: true})
    }
    $('scan-axis-buttons').append(button)
  }

  function label(value) {
    /*
    Format compact plot tick values

    Args:
        value: Axis value
    Returns:
        text: Short display label
    */
    return typeof value === 'number' ? Number(value.toPrecision(3)).toString() : String(value).slice(0, 24)
  }

  function axes(frame, x, y, xLabel, yLabel, formatX = label) {
    /*
    Draw labelled horizontal and vertical plot axes

    Args:
        frame: Canvas frame
        x: Horizontal scale
        y: Vertical scale
        xLabel: Horizontal axis name
        yLabel: Vertical axis name
        formatX: Horizontal tick formatter
    */
    const {ctx, left, right, top, bottom, height} = frame
    ctx.strokeStyle = '#283244'
    ctx.fillStyle = '#8e9bb0'
    ctx.textAlign = 'center'
    for (const tick of x.ticks.slice(0, 12)) {
      const px = x.map(tick)
      ctx.beginPath()
      ctx.moveTo(px, top)
      ctx.lineTo(px, bottom)
      ctx.stroke()
      ctx.fillText(formatX(tick), px, bottom + 20, 100)
    }
    ctx.textAlign = 'right'
    for (const tick of y.ticks.slice(0, 12)) {
      const py = y.map(tick)
      ctx.beginPath()
      ctx.moveTo(left, py)
      ctx.lineTo(right, py)
      ctx.stroke()
      ctx.fillText(label(tick), left - 8, py + 4, 65)
    }
    ctx.textAlign = 'center'
    ctx.fillText(xLabel, (left + right) / 2, height - 18)
    ctx.textAlign = 'left'
    ctx.fillText(yLabel, left, 16)
  }

  function color(value, values) {
    /*
    Map numeric or categorical values to a stable colour scale

    Args:
        value: Current parameter or objective
        values: Colour domain
    Returns:
        colour: Canvas stroke and fill colour
    */
    if (value == null) return '#8e9bb0'
    if (!colorDomains.has(values)) colorDomains.set(values, [...new Set(values.filter(item => item != null))].sort(typeof value === 'number' ? (a, b) => a - b : undefined))
    const domain = colorDomains.get(values)
    if (typeof value !== 'number') return palette[Math.max(0, domain.indexOf(value)) % palette.length]
    const t = (value - domain[0]) / (domain.at(-1) - domain[0] || 1)
    return `hsl(${220 - 190 * t} 70% 68%)`
  }

  function drawScan() {
    /*
    Draw scan projections with bounded canvas work
    */
    if (!state.open || !state.data) return
    const frame = canvasFrame('scan-canvas')
    state.frame = frame
    const overlay = $('scan-highlight')
    overlay.width = frame.canvas.width
    overlay.height = frame.canvas.height
    overlay.getContext('2d').setTransform(overlay.width / frame.width, 0, 0, overlay.height / frame.height, 0, 0)
    state.hover = null
    drag = null
    $('scan-tooltip').hidden = true
    frame.canvas.style.cursor = ''
    const {ctx, left, right, top, bottom} = frame
    const points = state.data.points.filter(point => Number.isFinite(point.value))
    const kind = $('scan-plot').value
    frame.canvas.dataset.plot = kind
    scanControls()
    $('scan-axis-buttons').replaceChildren()
    $('scan-summary').textContent = `${state.data.total.toLocaleString()} trials · ${state.data.points.length.toLocaleString()} plotted records${state.data.sampled ? ' (deterministic sample)' : ''} · ${state.data.points.length - points.length} missing metric values · colour: ${kind === 'scatter' ? $('scan-y').value : state.data.metric}`
    state.hits = []
    state.axes = []
    if (!points.length) {
      $('scan-colors').replaceChildren()
      ctx.fillText('No finite values for this selection', left, top + 30)
      return
    }
    const colorValues = kind === 'scatter' ? points.map(point => point.parameters[$('scan-y').value]) : kind === 'heatmap' ? (state.data.aggregates?.groups || []).map(group => group.mean) : points.map(point => point.value)
    const colorTicks = [...new Set(colorValues.filter(value => value != null))].sort(typeof colorValues.find(value => value != null) === 'number' ? (a, b) => a - b : undefined)
    const legendValues = colorTicks.length > 8 ? [colorTicks[0], colorTicks[Math.floor(colorTicks.length / 2)], colorTicks.at(-1)] : colorTicks
    $('scan-colors').hidden = kind === 'distribution'
    $('scan-colors').replaceChildren(...legendValues.map(value => {
      const entry = node('span', `● ${label(value)}`)
      entry.style.color = color(value, colorValues)
      return entry
    }))
    const xKey = $('scan-x').value
    const yKey = $('scan-y').value
    if (kind === 'parallel') {
      const keys = [...$('scan-axes').selectedOptions].map(option => option.value).slice(0, 7)
      keys.push('$metric')
      const dimensions = keys.map((key, index) => ({key, px: left + index * (right - left) / Math.max(1, keys.length - 1), axis: scanScale(key, points.map(point => key === '$metric' ? point.value : point.parameters[key]), bottom, top)}))
      state.axes = dimensions
      const colors = points.map(point => point.value)
      for (const point of points) {
        ctx.strokeStyle = state.selected.has(point.id) ? '#ffffff' : color(point.value, colors)
        ctx.globalAlpha = state.selected.has(point.id) ? 1 : Math.max(0.06, Math.min(0.5, 30 / Math.sqrt(points.length)))
        ctx.lineWidth = state.selected.has(point.id) ? 2.5 : 1
        ctx.beginPath()
        let started = false
        const path = []
        for (const dimension of dimensions) {
          const py = dimension.axis.map(dimension.key === '$metric' ? point.value : point.parameters[dimension.key])
          if (!Number.isFinite(py)) {
            path.push(null)
            started = false
            continue
          }
          path.push({x: dimension.px, y: py})
          if (started) ctx.lineTo(dimension.px, py)
          else ctx.moveTo(dimension.px, py)
          started = true
        }
        ctx.stroke()
        const last = dimensions.at(-1)
        state.hits.push({x: last.px, y: last.axis.map(point.value), path, point, color: ctx.strokeStyle})
      }
      ctx.globalAlpha = 1
      ctx.lineWidth = 1
      dimensions.forEach(({key, px, axis}, index) => {
        ctx.strokeStyle = '#8e9bb0'
        ctx.beginPath()
        ctx.moveTo(px, top)
        ctx.lineTo(px, bottom)
        ctx.stroke()
        ctx.fillStyle = '#e5ebf4'
        ctx.textAlign = index === 0 ? 'left' : index === dimensions.length - 1 ? 'right' : 'center'
        const spacing = (right - left) / Math.max(1, dimensions.length - 1)
        const labelWidth = dimensions.length > 1 && (index === 0 || index === dimensions.length - 1) ? spacing / 2 : spacing
        axisButton(frame, key, axis, px, bottom + 22, Math.min(180, labelWidth - 4), ctx.textAlign)
        ctx.fillStyle = '#8e9bb0'
        for (const tick of axis.ticks.slice(0, 6)) ctx.fillText(label(tick), px, axis.map(tick) - 7, 100)
      })
    } else if (kind === 'heatmap' || kind === 'distribution') {
      drawAggregates(frame, points, kind, xKey, yKey)
    } else {
      const x = scanScale(xKey, points.map(point => point.parameters[xKey]), left, right)
      const y = scanScale('$metric', points.map(point => point.value), bottom, top)
      axes(frame, x, y, '', '')
      axisButton(frame, xKey, x, (left + right) / 2, bottom + 34, Math.min(280, right - left))
      axisButton(frame, '$metric', y, left, 0, Math.min(280, right - left), 'left')
      state.axes = [{key: xKey, axis: x}, {key: '$metric', axis: y}]
      const colors = points.map(point => point.parameters[yKey])
      for (const point of points) {
        const px = x.map(point.parameters[xKey])
        const py = y.map(point.value)
        if (!Number.isFinite(px) || !Number.isFinite(py)) continue
        ctx.fillStyle = color(point.parameters[yKey], colors)
        ctx.globalAlpha = points.length > 1000 ? 0.35 : 0.7
        ctx.beginPath()
        ctx.arc(px, py, state.selected.has(point.id) ? 6 : 3, 0, Math.PI * 2)
        ctx.fill()
        if (state.selected.has(point.id)) {
          ctx.strokeStyle = '#ffffff'
          ctx.stroke()
        }
        state.hits.push({x: px, y: py, point})
      }
      ctx.globalAlpha = 1
    }
  }

  function drawAggregates(frame, points, kind, xKey, yKey) {
    /*
    Draw mean heatmaps or categorical distributions with explicit spread

    Args:
        frame: Canvas drawing frame
        points: Finite trial observations
        kind: Heatmap or distribution
        xKey: First grouping parameter
        yKey: Second grouping parameter
    */
    const {ctx, left, right, top, bottom} = frame
    const aggregates = state.data.aggregates
    if (!aggregates) return
    if (aggregates.error) {
      ctx.fillText(aggregates.error, left, top + 30)
      return
    }
    const stats = aggregates.groups
    const unique = key => [...new Set(stats.map(group => group[key]).filter(value => value != null))].sort((a, b) => typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b)))
    const xs = unique('x')
    const ys = unique('y')
    if (kind === 'heatmap') {
      const w = (right - left) / Math.max(1, xs.length)
      const h = (bottom - top) / Math.max(1, ys.length)
      const colors = stats.map(group => group.mean)
      ctx.textAlign = 'center'
      xs.forEach((value, index) => ctx.fillText(label(value), left + (index + 0.5) * w, bottom + 20, w))
      ctx.textAlign = 'right'
      ys.forEach((value, index) => ctx.fillText(label(value), left - 8, top + (index + 0.5) * h + 4, 65))
      for (const group of stats) {
        const px = left + (xs.indexOf(group.x) + 0.5) * w
        const py = top + (ys.indexOf(group.y) + 0.5) * h
        ctx.fillStyle = color(group.mean, colors)
        ctx.fillRect(px - w / 2 + 1, py - h / 2 + 1, w - 2, h - 2)
        if (w > 60 && h > 25) {
          ctx.fillStyle = '#0d111a'
          ctx.textAlign = 'center'
          ctx.fillText(label(group.mean), px, py + 4)
        }
        state.hits.push({x: px, y: py, width: w, height: h, group, filters: {[xKey]: [group.x], [yKey]: [group.y]}})
      }
      ctx.fillStyle = '#8e9bb0'
      ctx.textAlign = 'left'
      ctx.fillText(`${yKey} · cell mean over all filtered trials and seeds`, left, 16)
      ctx.textAlign = 'center'
      ctx.fillText(xKey, (left + right) / 2, bottom + 48)
    } else {
      const y = scale(stats.flatMap(group => [group.minimum, group.maximum]), bottom, top)
      const x = scale(xs.map(String), left, right)
      axes(frame, x, y, xKey, `${state.data.metric} · min, quartiles, median, max`)
      const w = Math.min(45, (right - left) / Math.max(1, xs.length) * 0.6)
      for (const group of stats) {
        const px = x.map(String(group.x))
        ctx.strokeStyle = '#59d6c8'
        ctx.fillStyle = '#59d6c855'
        ctx.beginPath()
        ctx.moveTo(px, y.map(group.minimum))
        ctx.lineTo(px, y.map(group.maximum))
        ctx.stroke()
        const low = y.map(group.q1)
        const high = y.map(group.q3)
        ctx.fillRect(px - w / 2, high, w, Math.max(1, low - high))
        ctx.strokeRect(px - w / 2, high, w, Math.max(1, low - high))
        ctx.beginPath()
        ctx.moveTo(px - w / 2, y.map(group.median))
        ctx.lineTo(px + w / 2, y.map(group.median))
        ctx.stroke()
        state.hits.push({x: px, y: y.map(group.mean), width: Math.max(w, 20), height: bottom - top, group, filters: {[xKey]: [group.x]}})
      }
    }
  }

  function drawCurves(id, curves, metric, axis) {
    /*
    Render reduced learning curves while retaining missing-value gaps

    Args:
        id: Target canvas identifier
        curves: Run histories
        metric: Observable name
        axis: Horizontal coordinate
    */
    const frame = canvasFrame(id)
    const {ctx, left, right, top, bottom} = frame
    const points = curves.flatMap(curve => curve.points)
    if (!points.length) {
      ctx.fillText('No observations on this axis. Select a metric and one or more trials', left, top + 30)
      return
    }
    const x = scale(points.map(point => point[0]), left, right)
    const y = scale(points.map(point => point[1]), bottom, top)
    axes(frame, x, y, axis === 'step' ? 'Optimizer step' : axis, metric, axis === 'step' ? formatSteps : label)
    curves.forEach((curve, index) => {
      ctx.strokeStyle = palette[index % palette.length]
      ctx.lineWidth = 1.5
      ctx.beginPath()
      let started = false
      for (const [a, b] of curve.points) {
        if (!Number.isFinite(a) || !Number.isFinite(b)) { started = false; continue }
        if (started) ctx.lineTo(x.map(a), y.map(b))
        else ctx.moveTo(x.map(a), y.map(b))
        started = true
      }
      ctx.stroke()
    })
  }

  async function loadCurves() {
    /*
    Fetch histories only for explicitly selected trials
    */
    curveRequest?.abort()
    const controller = new AbortController()
    curveRequest = controller
    const query = new URLSearchParams({metric: $('scan-metric').value, axis: $('scan-time').value})
    for (const id of state.selected) query.append('run', id)
    try {
      const curves = state.selected.size ? await get('/api/curves?' + query, controller.signal) : []
      if (controller.signal.aborted || !state.open) return
      state.curves = curves
      drawCurves('scan-curves', curves, $('scan-metric').value, $('scan-time').value)
      $('scan-legend').replaceChildren(...curves.map((curve, index) => {
        const item = node('p', `${curve.id} · ${curve.count.toLocaleString()} observations, ${curve.points.length} displayed`)
        item.style.color = palette[index % palette.length]
        return item
      }))
    } catch (error) { if (error.name !== 'AbortError') $('scan-legend').textContent = error.message }
  }

  function drawObservable() {
    /*
    Draw the selected observable against its recorded horizontal coordinate
    */
    if (!state.observable || state.observable.id !== selected) return
    const {id, metric, axis, points} = state.observable
    const key = JSON.stringify([id, metric, axis, points, $('observable-x-scale').dataset.scale, $('observable-y-scale').dataset.scale])
    if (key === observableKey) return
    observableKey = key
    const rows = points.map(([x, value]) => ({x, value}))
    const label = metric === 'lr' ? 'Learning rate' : metric
    renderCurve(rows, 'observable-curve', 'observable', label, label, [['value', palette[0]]], axis)
    if (!rows.length) $('observable-curve').append(node('p', 'No observations on this axis', 'subtle'))
  }

  async function loadObservable(id) {
    /*
    Discover arbitrary metrics for the selected experiment

    Args:
        id: Selected experiment identifier
    */
    observableRequest?.abort()
    const controller = new AbortController()
    observableRequest = controller
    try {
      const summary = await get('/api/explorer?' + new URLSearchParams({id}), controller.signal)
      if (controller.signal.aborted || selected !== id || state.open) return
      const run = summary.rows.find(row => row.id === id)
      const names = Object.keys(run?.metrics || {}).sort()
      options('observable-metric', names.map(name => [name, name === 'lr' ? 'Learning rate (lr)' : name]))
      const metric = $('observable-metric').value
      const axis = $('observable-axis').value
      const curves = await get('/api/curves?' + new URLSearchParams({run: id, metric, axis}), controller.signal)
      if (controller.signal.aborted || selected !== id || state.open) return
      state.observable = {id, metric, axis, points: curves[0]?.points || []}
      drawObservable()
      $('observable-note').textContent = run?.synthetic ? 'Synthetic data' : ''
      $('observable-note').hidden = !run?.synthetic
      const table = node('table')
      const header = node('tr')
      for (const title of ['Step', 'Name', 'Value']) header.append(node('th', title))
      table.append(header)
      for (const event of curves[0]?.texts || []) for (const [name, value] of Object.entries(event.values)) {
        const row = node('tr')
        row.append(node('td', formatSteps(event.step)), node('td', name), node('td', String(value)))
        table.append(row)
      }
      $('observable-events').replaceChildren(table)
    } catch (error) {
      if (error.name !== 'AbortError') {
        $('observable-note').textContent = error.message
        $('observable-note').hidden = false
      }
    }
  }

  function highlightScan(hit) {
    /*
    Draw the hovered trial above the scan without rebuilding its axes

    Args:
        hit: Trial path or null to clear the highlight
    */
    const trial = hit?.path ? hit : null
    if (state.hover === trial) return
    state.hover = trial
    const ctx = $('scan-highlight').getContext('2d')
    const frame = state.frame
    if (!frame) return
    ctx.clearRect(0, 0, frame.width, frame.height)
    if (!trial) return
    ctx.beginPath()
    let started = false
    for (const vertex of trial.path) {
      if (!vertex) {
        started = false
        continue
      }
      if (started) ctx.lineTo(vertex.x, vertex.y)
      else ctx.moveTo(vertex.x, vertex.y)
      started = true
    }
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.globalAlpha = 1
    ctx.strokeStyle = '#0d111a'
    ctx.lineWidth = 7
    ctx.stroke()
    ctx.strokeStyle = trial.color
    ctx.lineWidth = 4
    ctx.stroke()
  }

  function scanPointer(event) {
    /*
    Convert screen coordinates to the rendered scan frame

    Args:
        event: Pointer event
    Returns:
        pointer: Plot coordinates and screen scale, or null for a hidden canvas
    */
    const rect = $('scan-canvas').getBoundingClientRect()
    if (!state.frame || rect.width <= 0 || rect.height <= 0) return null
    const sx = rect.width / state.frame.width
    const sy = rect.height / state.frame.height
    return {x: (event.clientX - rect.left) / sx, y: (event.clientY - rect.top) / sy, sx, sy}
  }

  function nearest(event) {
    /*
    Find a nearby trial or aggregate under the pointer

    Args:
        event: Pointer event
    Returns:
        hit: Matching trial or aggregate
    */
    const pointer = scanPointer(event)
    if (!pointer) return null
    const {x, y, sx, sy} = pointer
    if (x < 0 || y < 0 || x > state.frame.width || y > state.frame.height) return null
    let hit = null
    let distance = 144
    for (const item of state.hits) {
      let d = ((item.x - x) * sx) ** 2 + ((item.y - y) * sy) ** 2
      if (item.path) {
        d = Infinity
        let previous = null
        for (const vertex of item.path) {
          if (vertex) {
            const start = previous || vertex
            if (x >= Math.min(start.x, vertex.x) - 12 / sx && x <= Math.max(start.x, vertex.x) + 12 / sx) {
              const ax = (start.x - x) * sx
              const ay = (start.y - y) * sy
              const dx = (vertex.x - start.x) * sx
              const dy = (vertex.y - start.y) * sy
              const length = dx * dx + dy * dy
              const t = length ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / length)) : 0
              d = Math.min(d, (ax + t * dx) ** 2 + (ay + t * dy) ** 2)
            }
          }
          previous = vertex
        }
      }
      if (item.group && Math.abs(item.x - x) <= item.width / 2 && Math.abs(item.y - y) <= item.height / 2) return item
      if (d < distance) { distance = d; hit = item }
    }
    return hit
  }

  let drag = null
  $('scan-canvas').onpointerdown = event => {
    if (event.button !== 0 || event.isPrimary === false || drag) return
    const pointer = scanPointer(event)
    if (!pointer) return
    drag = {...pointer, pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY}
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  $('scan-canvas').onpointermove = event => {
    const hit = nearest(event)
    highlightScan(hit)
    event.currentTarget.style.cursor = hit ? 'pointer' : ''
    const tooltip = $('scan-tooltip')
    if (hit) tooltip.hidden = false
    if (hit?.point) tooltip.textContent = `${hit.point.id}\n${state.data.metric}: ${number(hit.point.value)}\n${Object.entries(hit.point.parameters).map(([key, value]) => `${key}: ${value}`).join('\n')}`
    else if (hit?.group) tooltip.textContent = `n = ${hit.group.count}\nmean = ${number(hit.group.mean)}\nstandard deviation = ${number(hit.group.sd)}\n${JSON.stringify(hit.filters)}`
  }
  $('scan-canvas').onpointerleave = () => {
    highlightScan(null)
    $('scan-canvas').style.cursor = ''
  }
  $('scan-canvas').onpointerup = event => {
    if (!drag || event.pointerId !== drag.pointerId) return
    const start = drag
    drag = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    const end = scanPointer(event)
    if (!end) return
    if (Math.hypot(event.clientX - start.clientX, event.clientY - start.clientY) < 6) {
      const hit = nearest(event)
      if (hit?.point) toggleTrial(hit.point.id)
      else if (hit?.filters) { Object.assign(state.filters, hit.filters); state.offset = 0; loadScan() }
    } else if ($('scan-plot').value === 'parallel') {
      const axis = state.axes.find(axis => Math.abs(axis.px - start.x) * end.sx < 25)
      if (axis) { state.filters[axis.key] = axis.axis.range(start.y, end.y); state.offset = 0; loadScan() }
    } else if ($('scan-plot').value === 'scatter') {
      if (state.axes.length === 2) {
        state.filters[state.axes[0].key] = state.axes[0].axis.range(start.x, end.x)
        state.filters.$metric = state.axes[1].axis.range(start.y, end.y)
        state.offset = 0
        loadScan()
      }
    }
  }
  $('scan-canvas').onpointercancel = $('scan-canvas').onlostpointercapture = () => {
    drag = null
    highlightScan(null)
    $('scan-tooltip').hidden = true
    $('scan-canvas').style.cursor = ''
  }
  $('open-scans').onclick = () => openScan()
  for (const id of ['scan-driver', 'scan-group', 'scan-status', 'scan-metric', 'scan-mode', 'scan-reduce', 'scan-budget']) $(id).onchange = () => {
    state.offset = 0
    state.filters = {}
    if (id === 'scan-driver') options('scan-group', [['', 'All groups'], ...state.tree.filter(branch => !$('scan-driver').value || branch.driver === $('scan-driver').value).map(branch => [branch.group_id, branch.group_id])], '')
    loadScan()
  }
  for (const id of ['scan-x', 'scan-y', 'scan-axes']) $(id).onchange = () => {
    if ($('scan-axes').selectedOptions.length > 7) {
      for (const option of [...$('scan-axes').selectedOptions].slice(7)) option.selected = false
      showActionMessage('Select at most seven parallel parameter axes')
    }
    loadScan()
  }
  $('scan-plot').onchange = () => { scanControls(); loadScan() }
  $('scan-clear').onclick = () => { state.filters = {}; state.offset = 0; loadScan() }
  $('scan-unselect').onclick = () => { state.selected.clear(); renderTable(); drawScan(); loadCurves() }
  $('scan-prev').onclick = () => { state.offset = Math.max(0, state.offset - 50); loadScan() }
  $('scan-next').onclick = () => { state.offset += 50; loadScan() }
  $('scan-time').onchange = loadCurves
  for (const id of ['observable-metric', 'observable-axis']) $(id).onchange = () => loadObservable(selected)
  for (const id of ['observable-x-scale', 'observable-y-scale']) bindAxisScale(id, drawObservable)
  $('search').oninput = () => {
    clearTimeout(searchTimer)
    searchTimer = setTimeout(() => {
      state.offset = 0
      state.scanRevision = null
      refreshController?.abort()
      refreshController = null
      refreshExplorer()
    }, 180)
  }
  new ResizeObserver(() => {
    cancelAnimationFrame(plotFrame)
    plotFrame = requestAnimationFrame(() => {
      if (state.open) {
        drawScan()
        if (state.curves) drawCurves('scan-curves', state.curves, $('scan-metric').value, $('scan-time').value)
      }
    })
  }).observe($('main'))
  window.twistExplorer = {refresh: refreshExplorer, get busy() { return !!state.request }, hide: () => { state.open = false; $('scan-view').hidden = true; state.request?.abort(); curveRequest?.abort() }}
})()
