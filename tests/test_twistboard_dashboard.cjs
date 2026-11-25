// Dashboard refresh and preference regression checks
//
// mikael.mieskolainen@cern.ch (2026)

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const {test} = require('node:test')

function dashboard() {
  /*
  Load dashboard functions without starting browser polling

  Returns:
      context: Isolated dashboard with controlled requests and storage
  */
  const elements = new Map()
  const context = vm.createContext({
    AbortController,
    Intl,
    setTimeout,
    clearTimeout,
    location: {hash: ''},
    window: {matchMedia: () => ({matches: false})},
    document: {
      getElementById: id => {
        if (!elements.has(id)) elements.set(id, {
          textContent: '',
          classList: {add() {}, remove() {}}
        })
        return elements.get(id)
      },
      querySelectorAll: () => []
    },
    localStorage: {
      getItem() { throw new Error('Storage unavailable') },
      setItem() { throw new Error('Storage unavailable') }
    }
  })
  const source = fs.readFileSync(path.join(__dirname, '../twistboard/static/dashboard.js'), 'utf8')
  vm.runInContext(source.split("$('search').oninput = renderList")[0], context)
  return context
}

test('Unavailable preference storage keeps controls functional', () => {
  const context = dashboard()
  assert.equal(vm.runInContext("readPreference('sidebar.hidden')", context), null)
  assert.doesNotThrow(() => vm.runInContext("writePreference('sidebar.hidden', true)", context))
})

test('Experiment header shows exact total and trainable parameter counts', () => {
  const context = dashboard()
  const total = (253177).toLocaleString()
  assert.equal(context.formatParameterCount({total: 253177, trainable: 253177}), `${total} parameters`)
  assert.equal(context.formatParameterCount({total: 253177, trainable: 0}), `${total} parameters · 0 trainable`)
  assert.equal(context.formatParameterCount(), 'Parameter count unavailable')
})

test('Optimizer steps use thousands and retain single-step precision', () => {
  const context = dashboard()
  const steps = vm.runInContext('formatSteps', context)
  assert.equal(steps(0), '0k')
  assert.equal(steps(1), '0.001k')
  assert.equal(steps(1250), '1.25k')
  assert.equal(steps(12345), '12.345k')
  assert.equal(steps(null), '—')
})

test('Force deletion confirms the selected run before sending an explicit request', async () => {
  const context = dashboard()
  const elements = new Map()
  const element = () => ({dataset: {}, classList: {toggle() {}}, append() {}, replaceChildren() {}})
  context.document.createElement = element
  context.document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, element())
    return elements.get(id)
  }
  context.renderCurves = context.renderEvaluations = context.renderFeatureImportance = () => {}
  const requests = []
  const confirmations = []
  context.fetch = async (path, options) => {
    requests.push([path, options])
    return {ok: true, json: async () => ({deleted: 'active run'})}
  }
  context.refresh = async () => {}
  context.showActionMessage = () => {}
  context.window.confirm = message => {
    confirmations.push(message)
    return false
  }
  const run = {id: 'active run', directory: '/runs/active run', status: 'evaluation_evaluating',
    delete_blocked: 'Stop evaluation before deleting this directory', history: [], checkpoints: {}, files: []}
  context.renderDetail(run)
  const remove = elements.get('delete-run')
  assert.equal(remove.textContent, 'Force delete experiment')
  assert.equal(remove.disabled, false)
  await remove.onclick()
  assert.equal(requests.length, 0)
  assert.match(confirmations[0], /Force delete experiment "active run"/)
  assert.match(confirmations[0], /Other jobs may keep running/)
  context.window.confirm = () => true
  await remove.onclick()
  await context.deleteDirectory('evaluation', 'run/evaluations/test', '/runs/run/evaluations/test')
  assert.equal(requests[0][0], '/api/runs/active%20run')
  assert.equal(requests[0][1].headers['X-Twistboard-Action'], 'force-delete')
  assert.equal(requests[1][1].headers['X-Twistboard-Action'], 'delete')
  context.renderDetail({...run, status: 'trained', delete_blocked: null})
  assert.equal(remove.textContent, 'Delete experiment')
})

test('Feature rankings appear when available and support paging and filtering', async () => {
  const context = dashboard()
  const elements = new Map()
  const element = tag => ({
    children: [],
    attributes: {},
    style: {setProperty(name, value) { this[name] = value }},
    setAttribute(name, value) { this.attributes[name] = value },
    append(...children) { this.children.push(...children) },
    replaceChildren(...children) {
      this.children = children
      if (tag === 'importance-checkpoint') this.value = children[0]?.value
    }
  })
  context.document.createElement = element
  context.document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, element(id))
    return elements.get(id)
  }
  const requested = []
  context.get = async path => {
    requested.push(path)
    return {features: Array.from({length: 35}, (_, index) => ({
      rank: index + 1,
      feature: `shower_${index + 1}`,
      score: index === 34 ? 0 : 35 - index
    }))}
  }
  context.renderFeatureImportance({id: 'run', files: []})
  assert.equal(elements.get('feature-importance').hidden, true)
  const files = ['checkpoints/best_feature_importance.json', 'checkpoints/last_feature_importance.json',
    'plots/best_feature_importance.pdf'].map(path => ({path: `run/${path}`}))
  const run = {id: 'run', files}
  await context.renderFeatureImportance(run)
  assert.equal(elements.get('feature-importance').hidden, false)
  assert.equal(elements.get('importance-files').children.length, 2)
  const rows = elements.get('importance-rows')
  const bar = row => row.children[2].children[0].children[0]
  const barWidth = row => bar(row).style['--gain-width']
  assert.equal(rows.children.length, 32)
  assert.equal(barWidth(rows.children[0]), '100%')
  assert.equal(bar(rows.children[0]).attributes['aria-hidden'], 'true')
  elements.get('importance-next').onclick()
  assert.equal(rows.children.length, 3)
  assert.equal(rows.children[0].children[0].textContent, '33')
  assert.equal(barWidth(rows.children[0]), `${3 / 35 * 100}%`)
  const filter = elements.get('importance-filter')
  filter.value = 'SHOWER_33'
  filter.oninput()
  assert.equal(barWidth(rows.children[0]), `${3 / 35 * 100}%`)
  filter.value = 'SHOWER_35'
  filter.oninput()
  assert.equal(rows.children.length, 1)
  assert.equal(rows.children[0].children[1].textContent, 'shower_35')
  assert.equal(barWidth(rows.children[0]), '0%')
  await context.renderFeatureImportance(run)
  assert.equal(requested.length, 1)
  const select = elements.get('importance-checkpoint')
  select.value = files[1].path
  await select.onchange()
  assert.deepEqual(requested, files.slice(0, 2).map(file => `/files/${file.path}`))
  assert.equal(elements.get('importance-files').children.length, 1)
  context.renderFeatureImportance({id: 'other', files: []})
  assert.equal(elements.get('feature-importance').hidden, true)
})

test('Training distributions keep the displayed plot while another variable loads', async () => {
  const context = dashboard()
  const elements = new Map()
  const element = tag => ({
    children: [], style: {setProperty() {}},
    append(...children) { this.children.push(...children) },
    replaceChildren(...children) {
      this.children = children
      if (['distribution-category', 'distribution-feature'].includes(tag)) this.value = children[0]?.value ?? ''
    }
  })
  context.document.createElement = element
  context.document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, element(id))
    return elements.get(id)
  }
  const requests = []
  const plots = []
  const pending = []
  const scales = new Map()
  const meta = {features: ['spectral_depth', 'log_sum_energy'], classes: ['pi0', 'gamma'],
    categories: ['All training events', '0–7.5 GeV'], events: [[10, 8], [5, 4]]}
  const histogram = {edges: [0, 1, 2], weights: [[[4, 6], [3, 5]], [[2, 3], [0, 4]]]}
  context.get = async (path, signal) => {
    requests.push(path)
    if (path.endsWith('feature=1')) return new Promise(resolve => pending.push({resolve, signal}))
    return path.endsWith('.json') ? meta : histogram
  }
  context.bindAxisScale = (id, redraw) => scales.set(id, redraw)
  context.drawTrainingDistribution = (...args) => {
    plots.push(args)
    context.document.getElementById('distribution-plot').replaceChildren(element('svg'))
  }
  await context.renderTrainingDistributions({id: 'run', files: []})
  assert.equal(elements.get('training-distributions').hidden, true)
  const files = ['json', 'npz'].map(extension => ({path: `run/plots/training_distributions.${extension}`}))
  await context.renderTrainingDistributions({id: 'run', files})
  assert.equal(elements.get('training-distributions').hidden, false)
  assert.equal(requests.length, 2)
  assert.equal(requests[1], '/api/training-distributions/run?feature=0')
  assert.deepEqual(plots.at(-1)[2], histogram.weights[0])
  const category = elements.get('distribution-category')
  category.value = '1'
  category.onchange()
  assert.equal(requests.length, 2)
  assert.deepEqual(plots.at(-1)[2], histogram.weights[1])
  const filter = elements.get('distribution-filter')
  const chart = elements.get('distribution-plot')
  const visible = chart.children[0]
  filter.value = 'energy'
  const firstLoad = filter.oninput()
  assert.equal(chart.children[0], visible)
  assert.equal(elements.get('distribution-legend').children.length, 2)
  scales.get('distribution-y-scale')()
  assert.equal(plots.at(-1)[0], 'spectral_depth')
  assert.equal(requests.at(-1), '/api/training-distributions/run?feature=1')
  const current = chart.children[0]
  filter.value = 'spectral'
  await filter.oninput()
  assert.equal(pending[0].signal.aborted, true)
  pending[0].resolve(histogram)
  await firstLoad
  assert.equal(chart.children[0], current)
  assert.equal(requests.length, 3)
  filter.value = 'energy'
  const secondLoad = filter.oninput()
  assert.equal(chart.children[0], current)
  pending[1].resolve(histogram)
  await secondLoad
  assert.equal(plots.at(-1)[0], 'log_sum_energy')
  assert.deepEqual(plots.at(-1)[2], histogram.weights[1])
  assert.notEqual(chart.children[0], current)
})

test('Logarithmic histograms omit invalid bins and keep gaps at zero height', () => {
  const context = dashboard()
  const elements = new Map()
  const element = tag => ({
    tag, children: [], attributes: {}, dataset: {scale: 'log10'},
    setAttribute(name, value) { this.attributes[name] = value },
    append(...children) { this.children.push(...children) },
    replaceChildren(...children) { this.children = children }
  })
  context.document.createElement = element
  context.document.createElementNS = (_, tag) => element(tag)
  context.document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, element(id))
    return elements.get(id)
  }
  context.drawTrainingDistribution('feature', [-4, 0, 1, 10, 100, 1000], [[4, 10, 2, 0, 8], [0, 0, 0, 1, 0]],
    ['pi0', 'gamma'], false)
  const chart = elements.get('distribution-plot')
  const svg = chart.children[0]
  const paths = svg.children.filter(child => child.tag === 'path').map(child => child.attributes.d)
  assert.equal(paths.length, 2)
  assert.equal(paths[0].match(/M/g).length, 2)
  assert.equal(paths[1].match(/M/g).length, 1)
  assert.doesNotMatch(paths.join(' '), /Infinity|NaN/)
  assert.match(svg.attributes['aria-label'], /log10 X axis and log10 Y axis/)
  assert.match(elements.get('distribution-axis-note').textContent, /nonpositive edges.*zero-height bins/)
  context.drawTrainingDistribution('feature', [-4, 0, 1], [[4, 10], [0, 1]], ['pi0', 'gamma'], false)
  assert.match(chart.children[0].textContent, /Try a linear X axis/)
})

test('Validation loss cards use the last finite loss and the run minimum', () => {
  const context = dashboard()
  const losses = context.validationLosses([{val_loss: 0.8}, {val_loss: 0.5}, {val_loss: 0.7}, {}])
  assert.equal(losses.last, 0.7)
  assert.equal(losses.best, 0.5)
  assert.equal(context.validationLosses([{}]).last, null)
  assert.equal(context.validationLosses([]).best, null)
})

test('Evaluation cards and class metrics use the stored statistical error', () => {
  const context = dashboard()
  const cards = new Map()
  const element = () => ({append() {}, replaceChildren() {}})
  context.document.createElement = element
  context.document.getElementById = element
  context.metricCard = (label, value) => {
    cards.set(label, value)
    return element()
  }
  const metrics = {accuracy: 0.97, macro_auc: 0.996, uncertainties: {accuracy: {std: 0.012}, macro_auc: {std: 0.00034}}}
  context.renderEvaluations({id: 'run', evaluations: [{directory: 'test', metrics}]})
  const fixed = (value, digits) => value.toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits})
  assert.equal(cards.get('Accuracy'), `${fixed(0.97, 3)} ± ${fixed(0.012, 3)}`)
  assert.equal(cards.get('Macro AUC'), `${fixed(0.996, 5)} ± ${fixed(0.00034, 5)}`)
  const rows = context.metricRows({classes: {gamma: {auc: 0.995, uncertainties: {auc: {std: 0.003}}}}})
  assert.equal(rows.length, 1)
  assert.equal(rows[0][0], 'classes.gamma.auc')
  assert.equal(rows[0][2].std, 0.003)
  assert.equal(context.evaluationMetric(0.5), (0.5).toLocaleString())
})

test('Rejection cards follow the saved signal efficiencies in order', () => {
  const context = dashboard()
  const cards = new Map()
  const element = tag => ({
    tag,
    children: [],
    append(...children) { this.children.push(...children) },
    replaceChildren() { this.children = [] }
  })
  context.document.createElement = element
  context.document.getElementById = element
  context.metricCard = (label, value) => {
    assert.equal(label.className, 'rejection-label')
    const text = label.children.map(part => part.tag === 'sub' ? `_${part.textContent}` : part).join('')
    cards.set(text, value)
    return element()
  }
  const render = points => context.renderEvaluations({
    id: 'run',
    evaluations: [{directory: 'test', metrics: {signal_operating_points: points}}]
  })
  const at30 = `1/ϵ_B at ϵ_S = ${(0.3).toLocaleString()}`
  const at50 = `1/ϵ_B at ϵ_S = ${(0.5).toLocaleString()}`
  const at725 = `1/ϵ_B at ϵ_S = ${(0.725).toLocaleString()}`
  const at90 = `1/ϵ_B at ϵ_S = ${(0.9).toLocaleString()}`
  const zero = {requested_efficiency: 0.3, physical: true, zero_background: true, rejection: null}
  render([{requested_efficiency: 0.725, physical: true, rejection: 5250}, zero,
    {requested_efficiency: 0.9, physical: true, rejection: 24}])
  assert.deepEqual([...cards.keys()], [at725, at30, at90])
  assert.equal(cards.get(at30), 'Zero background efficiency')
  assert.equal(cards.get(at725), (5250).toLocaleString())
  assert.equal(cards.get(at90), '24')
  cards.clear()
  render([{...zero, physical: false}])
  assert.equal(cards.get(at30), 'Undefined')
  assert.deepEqual([...cards.keys()], [at30])
  cards.clear()
  render([])
  assert.equal(cards.size, 0)
  const rejection = {std: null, lower: 8000, upper: null, upper_unbounded: true, confidence: .68,
    valid_resamples: 30, zero_background_resamples: 13}
  const interval = `[${(8000).toLocaleString()}, ${Infinity.toLocaleString()}]`
  render([{...zero, zero_background: false, rejection: 8652, uncertainties: {rejection}},
    {...zero, requested_efficiency: 0.5, uncertainties: {rejection}}])
  assert.equal(cards.get(at30), `${(8652).toLocaleString()} ${interval}`)
  assert.equal(cards.get(at50), `${Infinity.toLocaleString()} ${interval}`)
  const row = context.metricRows({...zero, uncertainties: {rejection}}).find(([key]) => key === 'rejection')
  assert.equal(row[1], Infinity)
  assert.equal(context.evaluationMetric(row[1], row[2]), cards.get(at50))
})

for (const failed of [false, true]) {
  test(`Host evaluation sends a relative checkpoint and restores controls, failed=${failed}`, async () => {
    const context = dashboard()
    await vm.runInContext(`(async () => {
      let refreshed = 0
      let calls = 0
      const control = {disabled: false}
      document.querySelectorAll = () => [control]
      refresh = async () => { refreshed += 1 }
      let finish
      fetch = async (url, request) => {
        calls += 1
        if (url !== '/api/evaluations' || request.method !== 'POST') throw new Error('Wrong endpoint')
        if (request.headers['X-Twistboard-Action'] !== 'evaluate') throw new Error('Missing action header')
        if (JSON.parse(request.body).checkpoint !== 'run with spaces/checkpoints/best.pt') throw new Error('Wrong checkpoint')
        await new Promise(resolve => { finish = resolve })
        return {ok: !${failed}, json: async () => ${failed} ? {error: 'Host is busy'} : {id: 'run with spaces/evaluations/new'}}
      }
      const pending = runEvaluation('run with spaces', 'best.pt')
      if (!control.disabled) throw new Error('Control remained enabled')
      await runEvaluation('run with spaces', 'best.pt')
      if (calls !== 1) throw new Error('Duplicate launch')
      finish()
      await pending
      if (control.disabled || startingEvaluation) throw new Error('Launch state leaked')
      if (refreshed !== 1) throw new Error('Results were not refreshed')
      if (${failed}) {
        if ($('action-message').textContent !== 'Host is busy') throw new Error('Failure was hidden')
        if (evaluationSelection.has('run with spaces')) throw new Error('Failed result selected')
      } else if (evaluationSelection.get('run with spaces') !== 'run with spaces/evaluations/new') {
        throw new Error('New evaluation was not selected')
      }
    })()`, context)
  })
}

for (const failed of [false, true]) {
  test(`Superseded run-list request cannot replace current state, failed=${failed}`, async () => {
    const context = dashboard()
    await vm.runInContext(`(async () => {
      const pending = []
      get = (path, signal) => new Promise((resolve, reject) => pending.push({path, signal, resolve, reject}))
      renderList = () => {}
      let rendered = null
      renderDetail = run => { rendered = run.id }
      const old = refresh()
      const current = selectRun('new')
      pending[1].resolve([{id: 'new'}])
      await Promise.resolve()
      pending[2].resolve({id: 'new'})
      await current
      if (${failed}) pending[0].reject(new Error('Stale request failed'))
      else pending[0].resolve([{id: 'old'}])
      await old
      if (selected !== 'new' || runs[0].id !== 'new' || rendered !== 'new') throw new Error('Stale selection applied')
      if ($('connection').textContent !== '● Live') throw new Error('Stale error applied')
      if (refreshController !== null) throw new Error('Refresh lock leaked')
    })()`, context)
  })
}

test('Run-list refresh retains nodes, scroll and keyboard focus', () => {
  const context = dashboard()
  vm.runInContext(`
    document.createElement = () => ({
      dataset: {}, children: [],
      append(...children) { this.children.push(...children) },
      focus() { document.activeElement = this }
    })
    const list = $('runs')
    list.children = []
    list.scrollTop = 120
    list.contains = element => list.children.includes(element)
    list.append = element => list.children.push(element)
    list.replaceChildren = () => { list.children = [] }
    $('search').value = ''
    runs = [{id: 'demo', model: 'pelican', driver: 'toptag', task: 'classification', status: 'training', epochs: 1}]
    renderList()
    const button = list.children[0]
    document.activeElement = button
    renderList()
    if (list.children[0] !== button) throw new Error('Unchanged list rebuilt')
    runs[0].epochs = 2
    renderList()
    if (list.children[0] === button) throw new Error('Updated list was not rendered')
    if (document.activeElement !== list.children[0]) throw new Error('Keyboard focus lost')
    if (list.scrollTop !== 120) throw new Error('Scroll position lost')
  `, context)
})

test('Arrow navigation follows filtered experiments and stops at list boundaries', () => {
  const context = dashboard()
  vm.runInContext(`
    let clicked = 0
    let scrolled = null
    selectRun = id => { selected = id; clicked += 1 }
    const buttons = ['first', 'third'].map(id => ({
      dataset: {runId: id},
      focus() { document.activeElement = this },
      scrollIntoView() { scrolled = id },
      click() { activateRun(this) }
    }))
    $('runs').querySelectorAll = () => buttons
    selected = 'first'
    document.activeElement = buttons[0]
    const press = (key, options = {}) => {
      let prevented = false
      navigateRuns({key, preventDefault() { prevented = true }, ...options})
      return prevented
    }
    if (!press('ArrowDown') || selected !== 'third') throw new Error('Filtered next experiment not selected')
    if (document.activeElement !== buttons[1] || scrolled !== 'third') throw new Error('Selection not focused and scrolled')
    press('ArrowDown')
    if (clicked !== 1) throw new Error('Boundary caused a duplicate load')
    press('ArrowUp')
    if (selected !== 'first') throw new Error('Previous experiment not selected')
    press('ArrowUp')
    if (clicked !== 2) throw new Error('First boundary caused a duplicate load')
    for (const options of [{ctrlKey: true}, {altKey: true}, {metaKey: true}, {shiftKey: true}, {isComposing: true}]) {
      if (press('ArrowDown', options)) throw new Error('Modified key was consumed')
    }
    if (press('Enter')) throw new Error('Unrelated key was consumed')
    deleting = true
    if (press('ArrowDown')) throw new Error('Navigation allowed during deletion')
    deleting = false
    document.activeElement = $('search')
    selected = 'filtered-out'
    press('ArrowDown')
    if (selected !== 'first') throw new Error('Search did not enter first result')
    document.activeElement = $('search')
    selected = 'filtered-out'
    press('ArrowUp')
    if (selected !== 'third') throw new Error('Search did not enter last result')
    $('runs').querySelectorAll = () => []
    if (press('ArrowDown')) throw new Error('Empty list consumed navigation')
  `, context)
})

test('Mouse and arrow activation have the same selection, URL, focus and scroll effects', () => {
  const context = dashboard()
  vm.runInContext(`
    const effects = []
    document.createElement = () => ({
      dataset: {}, children: [],
      append(...children) { this.children.push(...children) },
      focus(options) {
        document.activeElement = this
        effects.push(['focus', this.dataset.runId, options])
      },
      scrollIntoView(options) { effects.push(['scroll', this.dataset.runId, options]) },
      click() { this.onclick() }
    })
    const list = $('runs')
    list.children = []
    list.scrollTop = 0
    list.contains = element => list.children.includes(element)
    list.append = element => list.children.push(element)
    list.replaceChildren = () => { list.children = [] }
    list.querySelectorAll = () => list.children
    $('search').value = ''
    runs = ['first', 'next run'].map(id => ({id, model: 'pelican', driver: 'toptag', status: 'trained', epochs: 1}))
    selected = 'first'
    selectRun = id => {
      selected = id
      effects.push(['select', id, location.hash])
    }
    renderList()
    list.children[1].click()
    const mouse = JSON.stringify(effects)
    if (location.hash !== 'next%20run' || selected !== 'next run') throw new Error('Mouse activation did not select run')
    selected = 'first'
    document.activeElement = list.children[0]
    effects.length = 0
    navigateRuns({key: 'ArrowDown', preventDefault() {}})
    if (JSON.stringify(effects) !== mouse) throw new Error('Mouse and keyboard activation differ')
    if (document.activeElement !== list.children[1]) throw new Error('Keyboard focus differs')
    if (location.hash !== 'next%20run' || selected !== 'next run') throw new Error('Keyboard selection differs')
  `, context)
})

test('Linear epoch ticks include short runs and use a constant integer stride', () => {
  const context = dashboard()
  const ticks = (first, last) => Array.from(context.integerTicks(first, last))
  assert.deepEqual(ticks(1, 14), Array.from({length: 14}, (_, index) => index + 1))
  assert.deepEqual(ticks(0.5, 1.5), [1])
  for (const last of [23, 35, 100, 500, 10000]) {
    const values = ticks(1, last)
    assert.ok(values.length <= 16)
    const step = values[1] - values[0]
    assert.ok(Number.isInteger(step) && step > 0)
    assert.ok(values.every((value, index) => value <= last && (index === 0 || value - values[index - 1] === step)))
  }
})
