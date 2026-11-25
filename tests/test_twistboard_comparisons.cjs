// Classifier comparison selection and rendering checks
//
// mikael.mieskolainen@cern.ch (2026)

const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const {test} = require('node:test')

function page() {
  /* Create a DOM fixture for comparison controls and SVG output */
  class Element {
    constructor(tag) {
      this.tagName = tag
      this.children = []
      this.style = {}
      this.dataset = {}
      this.attributes = {}
      this.value = ''
      this.hidden = false
    }
    append(...children) {
      this.children.push(...children)
      if (this.tagName === 'select' && !this.value) this.value = children[0]?.value || ''
    }
    replaceChildren(...children) { this.children = []; this.append(...children) }
    setAttribute(key, value) { this.attributes[key] = value }
    getScreenCTM() { return {a: 1, d: 1, e: 0, f: 0} }
    querySelectorAll(tag) { return this.children.filter(child => child.tagName === tag) }
    querySelector(tag) { return this.children.find(child => child.tagName === tag) }
    addEventListener() {}
  }
  const elements = new Map()
  const node = (tag, text, className) => Object.assign(new Element(tag), {textContent: text, className})
  const $ = id => {
    if (!elements.has(id)) elements.set(id, new Element(id.startsWith('comparison-') && ['choice', 'selection', 'reference', 'class', 'plot', 'x', 'y'].some(suffix => id.endsWith('-' + suffix)) ? 'select' : 'div'))
    return elements.get(id)
  }
  const context = vm.createContext({
    $, node, link: (path, text) => node('a', text),
    number: value => value == null ? '—' : String(value), formatTime: value => value,
    document: {createElementNS: (ns, tag) => new Element(tag)},
    AbortController, comparisonMode: false, setTimeout, setInterval() {},
    openFigure() {},
  })
  vm.runInContext(fs.readFileSync('twistboard/static/comparisons.js', 'utf8'), context)
  return {context, $}
}

test('Checkpoint selection survives different experiment rows and supports removal', () => {
  const {context, $} = page()
  vm.runInContext(`
    const first = node('td'), second = node('td')
    addComparisonCheckpoint(first, {id: 'a', command_directory: 'runs/a'}, {file: 'best.pt'})
    addComparisonCheckpoint(second, {id: 'b', command_directory: 'runs/b'}, {file: 'last.pt'})
    first.children[0].onclick()
    second.children[0].onclick()
    if (comparisonCheckpoints.size !== 2) throw new Error('Selection lost')
    if ($('comparison-card').disabled) throw new Error('Card download disabled')
    $('comparison-checkpoints').children[0].children[0].onclick()
    if (comparisonCheckpoints.size !== 1) throw new Error('Remove failed')
  `, context)
  assert.equal($('comparison-card').disabled, true)
})

test('Shared selection renders paired metrics and distinct model curves', () => {
  const {context, $} = page()
  vm.runInContext(`
    $('comparison-reference').value = 'A'
    $('comparison-class').value = '1'
    $('comparison-plot').value = 'roc'
    $('comparison-x').dataset.scale = 'linear'
    $('comparison-y').dataset.scale = 'linear'
    comparisonRecord = {
      status: 'completed', schema: {name: 'classification', outputs: ['background', 'signal']},
      selection: {label: 'Inclusive', support: {events: 12}, models: ['A', 'B'].map((label, index) => ({
        label, metrics: {classes: {signal: {auc: .8 + index * .1, operating_points: [{requested_efficiency: .3, rejection: 10, physical: true}]}}, cross_entropy: .2, brier: .1},
        curves: {'1': {fpr: [0, .2, 1], tpr: [0, .8, 1], recall: [0, .8, 1], precision: [1, .8, .5]}}
      })), paired: {A: {B: {'auc:signal': {estimate: .1, lower: .05, upper: .15}}}}
      }
    }
    renderComparison()
    const table = $('comparison-table').children[0]
    if (table.children[1].children[0].children[0].textContent !== 'B') throw new Error('AUC sorting failed')
    if (!table.children[1].children[0].children[2].textContent.includes('[0.05, 0.15]')) throw new Error('Interval missing')
    const svg = $('comparison-chart').children[0]
    const paths = svg.children.filter(child => child.tagName === 'path')
    if (paths[0].attributes.stroke === paths[1].attributes.stroke) throw new Error('Identical colors')
    comparisonHidden.add('A')
    renderComparison()
    if ($('comparison-table').children[0].children[1].children.length !== 1) throw new Error('Hidden model remains')
    $('comparison-plot').value = 'rejection'
    $('comparison-y').dataset.scale = 'log10'
    renderComparison()
  `, context)
  assert.equal($('comparison-chart').children[0].tagName, 'svg')
})

test('Superseded comparison responses cannot replace the current selection', async () => {
  const {context} = page()
  await vm.runInContext(`(async () => {
    const pending = []
    get = (url, signal) => new Promise(resolve => pending.push({url, signal, resolve}))
    renderComparison = () => {}
    $('comparison-choice').value = 'old'
    const old = loadComparison()
    $('comparison-choice').value = 'new'
    const current = loadComparison()
    const record = id => ({id, name: id, status: 'completed', models: [], schema: {name: 'classification', outputs: ['a', 'b']}, selections: [{id: 'inclusive', label: 'Inclusive'}]})
    pending[1].resolve(record('new'))
    await Promise.resolve()
    pending[2].resolve({...record('new'), selection: {models: []}})
    await current
    pending[0].resolve(record('old'))
    await Promise.resolve()
    if (pending[3]) pending[3].resolve({...record('old'), selection: {models: []}})
    await old
    if (comparisonRecord.id !== 'new') throw new Error('Stale response applied')
  })()`, context)
})

test('Run comparison sends selected file paths and opens the new job', async () => {
  const {context} = page()
  await vm.runInContext(`(async () => {
    comparisonCheckpoints.set('runs/a/checkpoints/best.pt', {label: 'A', checkpoint: 'a/checkpoints/best.pt'})
    comparisonCheckpoints.set('runs/b/checkpoints/best.pt', {label: 'B', checkpoint: 'b/checkpoints/best.pt'})
    let opened = null
    loadComparisons = async id => { opened = id }
    fetch = async (url, request) => {
      if (url !== '/api/comparisons' || request.method !== 'POST') throw new Error('Job was not launched')
      const body = JSON.parse(request.body)
      if (body.models[1].checkpoint !== 'b/checkpoints/best.pt') throw new Error('Wrong file path')
      return {ok: true, json: async () => ({id: 'comparisons/new'})}
    }
    await $('comparison-run').onclick()
    if (opened !== 'comparisons/new') throw new Error('New comparison did not open')
    if ($('comparison-run').disabled) throw new Error('Launch control stayed disabled')
  })()`, context)
})

test('Regression view shows RMSE differences and physical residual curves', () => {
  const {context, $} = page()
  vm.runInContext(`
    $('comparison-reference').value = 'A'
    $('comparison-class').value = '0'
    $('comparison-plot').value = 'residual'
    comparisonRecord = {status: 'completed', schema: {name: 'gaussian_regression', outputs: ['energy'], units: ['GeV']}, selection: {label: 'Inclusive', support: {events: 20}, models: [{label: 'A', metrics: {energy: {rmse: 2, mae: 1, bias: 0, nll: .4, coverage_1sigma: .68, coverage_2sigma: .95}}, curves: {'0': {residual: {x: [-1, 0, 1], y: [.2, .6, .2]}}}}], paired: {A: {A: {'rmse:energy': {estimate: 0, lower: 0, upper: 0}}}}}}
    renderComparison()
    const row = $('comparison-table').children[0].children[1].children[0]
    if (row.children[1].textContent !== '2') throw new Error('RMSE is missing')
    if (row.children[2].textContent !== '0 [0, 0]') throw new Error('Paired regression interval is missing')
  `, context)
  assert.equal($('comparison-chart').children[0].tagName, 'svg')
})
