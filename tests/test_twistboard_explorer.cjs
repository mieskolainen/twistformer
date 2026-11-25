// Scan coordinate and pointer regression checks
//
// mikael.mieskolainen@cern.ch (2026)

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const {test} = require('node:test')

function explorer(ratio = 1, width = 600, height = 360) {
  /*
  Load scan rendering and pointer handlers with a controlled canvas

  Args:
      ratio: Device pixel ratio
      width: Displayed canvas width
      height: Displayed canvas height
  Returns:
      fixture: Scan state, controls and pointer event factory
  */
  const elements = new Map()
  const node = (tag, textContent, className) => ({
    tag, textContent, className, style: {}, dataset: {}, children: [], attributes: {},
    append(...children) { this.children.push(...children) },
    replaceChildren(...children) { this.children = children },
    setAttribute(key, value) { this.attributes[key] = value },
    focus() { this.focused = true }
  })
  const $ = id => {
    if (!elements.has(id)) {
      const calls = []
      const ctx = new Proxy({calls}, {get: (target, key) => target[key] ?? ((...args) => {
        calls.push({method: key, args, color: target.strokeStyle, width: target.lineWidth, alpha: target.globalAlpha})
      })})
      elements.set(id, {
        ...node('div'),
        value: '', style: {}, dataset: {}, selectedOptions: [], clientWidth: 600,
        getContext: () => ctx,
        getBoundingClientRect: () => ({left: 40, top: 20, width, height}),
        setPointerCapture(id) { this.capture = id },
        hasPointerCapture(id) { return this.capture === id },
        releasePointerCapture() { this.capture = null }
      })
    }
    return elements.get(id)
  }
  const context = vm.createContext({
    $, window: {devicePixelRatio: ratio}, number: String,
    node, URLSearchParams, bindAxisScale() {}, ResizeObserver: class { observe() {} }
  })
  const source = fs.readFileSync(path.join(__dirname, '../twistboard/static/explorer.js'), 'utf8')
  vm.runInContext(source.replace(/\}\)\(\)\s*$/, `
    globalThis.fixture = {state, drawScan, nearest, query, clicks: [], loads: 0}
    toggleTrial = id => fixture.clicks.push(id)
    loadScan = () => { fixture.loads += 1 }
  })()`), context)
  const fixture = context.fixture
  $('scan-plot').value = 'parallel'
  $('scan-x').value = 'a'
  $('scan-axes').selectedOptions = [{value: 'a'}, {value: 'b'}]
  fixture.state.open = true
  fixture.state.data = {metric: 'loss', total: 2, points: [
    {id: 'A', value: 0, parameters: {a: 0, b: 10}},
    {id: 'B', value: 10, parameters: {a: 10, b: 0}}
  ]}
  fixture.drawScan()
  const event = (x, y, extra = {}) => ({
    clientX: 40 + x * width / 600, clientY: 20 + y * height / 360,
    currentTarget: $('scan-canvas'), pointerId: 1, button: 0, isPrimary: true, ...extra
  })
  return {...fixture, $, event}
}

test('Fractional budgets in thousands request the exact optimizer step', () => {
  const {$, query} = explorer()
  $('scan-budget').value = '1.001'
  assert.equal(new URLSearchParams(query()).get('budget'), '1001')
})

for (const [ratio, width, height] of [[1, 600, 360], [2, 600, 360], [2, 300, 270]]) {
  test(`Parallel hover follows every segment at pixel ratio ${ratio} and size ${width} x ${height}`, () => {
    const {nearest, $, event} = explorer(ratio, width, height)
    for (const [x, y] of [[76, 288], [138, 224], [262, 96], [324, 32], [386, 96], [510, 224], [572, 288]]) {
      assert.equal(nearest(event(x, y))?.point.id, 'A')
      $('scan-canvas').onpointermove(event(x, y))
      assert.equal($('scan-tooltip').hidden, false)
      assert.match($('scan-tooltip').textContent, /^A\nloss: 0\na: 0\nb: 10$/)
      assert.equal($('scan-canvas').style.cursor, 'pointer')
    }
    assert.equal(nearest(event(138, 96))?.point.id, 'B')
    assert.equal(nearest(event(200, 40)), null)
    assert.equal(nearest(event(-1, 288)), null)
    $('scan-canvas').onpointermove(event(200, 40))
    assert.equal($('scan-tooltip').hidden, false)
    assert.equal($('scan-canvas').style.cursor, '')
  })
}

test('Hover tolerance stays in screen pixels for a compressed canvas', () => {
  const {nearest, event, state, drawScan} = explorer(2, 300)
  state.data.points[0].parameters.b = 0
  state.data.points[1].parameters.b = 10
  drawScan()
  assert.equal(nearest(event(200, 278))?.point.id, 'A')
  assert.equal(nearest(event(200, 274)), null)
})

test('Missing and logarithmically excluded values do not create hit segments', () => {
  const {state, drawScan, nearest, $, event} = explorer()
  state.data.points[0].parameters.b = null
  drawScan()
  assert.equal(nearest(event(200, 288)), null)
  assert.equal(nearest(event(448, 288)), null)
  assert.equal(nearest(event(572, 288))?.point.id, 'A')
  state.data.points[0].parameters.b = 10
  drawScan()
  $('scan-axis-buttons').children.find(button => button.dataset.axis === 'a').onclick()
  assert.equal(nearest(event(138, 224)), null)
  assert.equal(nearest(event(386, 96))?.point.id, 'A')
})

test('Segment clicks select trials and axis drags retain coordinate filters', () => {
  const {state, $, event, clicks} = explorer(2, 300, 270)
  const canvas = $('scan-canvas')
  canvas.onpointerdown(event(138, 224))
  canvas.onpointerup(event(138, 224))
  assert.deepEqual(Array.from(clicks), ['A'])
  assert.equal(canvas.capture, null)
  canvas.onpointerdown(event(76, 224))
  canvas.onpointerup(event(76, 96))
  assert.equal(state.filters.a.min, 2.5)
  assert.equal(state.filters.a.max, 7.5)
  assert.equal(clicks.length, 1)
})

test('Cancelled, secondary and stale pointer gestures cannot select a trial', () => {
  const {state, drawScan, $, event, clicks} = explorer()
  const canvas = $('scan-canvas')
  for (const handler of ['onpointercancel', 'onlostpointercapture']) {
    canvas.onpointerdown(event(138, 224))
    canvas.onpointermove(event(138, 224))
    canvas[handler](event(138, 224))
    canvas.onpointerup(event(138, 224))
    assert.equal($('scan-tooltip').hidden, true)
  }
  canvas.onpointerdown(event(138, 224, {button: 2}))
  canvas.onpointerup(event(138, 224, {button: 2}))
  canvas.onpointerdown(event(138, 224, {isPrimary: false}))
  canvas.onpointerup(event(138, 224))
  canvas.onpointerdown(event(138, 224))
  canvas.onpointerup(event(138, 224, {pointerId: 2}))
  assert.equal(clicks.length, 0)
  drawScan()
  canvas.onpointerup(event(138, 224))
  assert.equal(clicks.length, 0)
  canvas.onpointermove(event(138, 224))
  canvas.onpointerleave()
  assert.equal($('scan-tooltip').hidden, false)
  state.data.points = []
  drawScan()
  canvas.onpointermove(event(138, 224))
  assert.equal($('scan-tooltip').hidden, true)
})

test('Scatter points and aggregate cells retain hover detection', () => {
  const {state, drawScan, nearest, $, event} = explorer()
  $('scan-plot').value = 'scatter'
  drawScan()
  assert.equal(nearest(event(76, 288))?.point.id, 'A')
  assert.equal(nearest(event(138, 224)), null)
  $('scan-plot').value = 'heatmap'
  $('scan-y').value = 'b'
  state.data.aggregates = {groups: [{x: 0, y: 10, mean: 5, count: 2, sd: 1}]}
  drawScan()
  assert.equal(nearest(event(100, 50))?.group.count, 2)
})

test('Hover draws the full trial in bold above the plot without rebuilding it', () => {
  const {state, $, event, drawScan} = explorer(2)
  const canvas = $('scan-canvas')
  const overlay = $('scan-highlight')
  const base = canvas.getContext('2d').calls
  const calls = overlay.getContext('2d').calls
  const baseCount = base.length
  canvas.onpointermove(event(138, 224))
  assert.equal(state.hover.point.id, 'A')
  assert.equal(base.length, baseCount)
  assert.equal(overlay.width, canvas.width)
  assert.equal(overlay.height, canvas.height)
  assert.deepEqual(calls.filter(call => ['moveTo', 'lineTo'].includes(call.method)).map(call => [call.method, ...call.args]), [
    ['moveTo', 76, 288], ['lineTo', 324, 32], ['lineTo', 572, 288]
  ])
  const stroke = calls.at(-1)
  assert.equal(stroke.method, 'stroke')
  assert.equal(stroke.width, 4)
  assert.equal(stroke.alpha, 1)
  assert.equal(stroke.color, state.hits[0].color)
  const count = calls.length
  canvas.onpointermove(event(262, 96))
  assert.equal(calls.length, count)
  canvas.onpointermove(event(138, 96))
  assert.equal(calls[count].method, 'clearRect')
  assert.equal(state.hover.point.id, 'B')
  canvas.onpointerleave()
  assert.equal(state.hover, null)
  assert.equal(calls.at(-1).method, 'clearRect')
  assert.equal($('scan-tooltip').hidden, false)
  assert.match($('scan-tooltip').textContent, /^B\n/)
  drawScan()
  assert.equal($('scan-tooltip').hidden, true)
})

test('Hover highlight retains missing-axis gaps and clears on cancellation', () => {
  const {state, $, event, drawScan} = explorer()
  state.data.points[0].parameters.b = null
  drawScan()
  const calls = $('scan-highlight').getContext('2d').calls
  $('scan-canvas').onpointermove(event(572, 288))
  assert.equal(calls.filter(call => call.method === 'lineTo').length, 0)
  assert.equal(calls.filter(call => call.method === 'moveTo').length, 2)
  $('scan-canvas').onpointercancel(event(572, 288))
  assert.equal(state.hover, null)
  assert.equal(calls.at(-1).method, 'clearRect')
})

test('Axis names toggle independent log scales and retain them across redraws', () => {
  const {state, $, drawScan} = explorer()
  state.data.points = [1, 10, 100].map(value => ({id: String(value), value, parameters: {a: value, b: 100 / value}}))
  drawScan()
  const button = key => $('scan-axis-buttons').children.find(control => control.dataset.axis === key)
  const axis = key => state.axes.find(item => item.key === key).axis
  assert.equal(button('a').attributes['aria-pressed'], 'false')
  assert.ok(axis('a').map(10) > 260)
  button('a').onclick()
  assert.equal(axis('a').map(10), 160)
  assert.equal(axis('b').log, false)
  assert.equal(axis('$metric').log, false)
  assert.equal(button('a').attributes['aria-pressed'], 'true')
  assert.equal(button('a').focused, true)
  button('$metric').onclick()
  assert.equal(axis('$metric').map(10), 160)
  assert.equal(axis('a').log, true)
  const range = axis('a').range(224, 96)
  assert.ok(Math.abs(range.min - Math.sqrt(10)) < 1e-10)
  assert.ok(Math.abs(range.max - Math.sqrt(1000)) < 1e-10)
  drawScan()
  assert.equal(axis('a').log, true)
  state.data.metric = 'accuracy'
  drawScan()
  assert.equal(axis('$metric').log, false)
  assert.equal(axis('a').log, true)
  state.data.metric = 'loss'
  drawScan()
  assert.equal(axis('$metric').log, true)
  button('a').onclick()
  assert.equal(axis('a').log, false)
  assert.equal(axis('$metric').log, true)
})

test('Log toggles exclude nonpositive values and disable categorical or nonpositive axes', () => {
  const {state, $, drawScan} = explorer()
  state.data.points = [-1, 0, 10].map(value => ({id: String(value), value, parameters: {a: value, b: 'category'}}))
  drawScan()
  assert.equal($('scan-axis-buttons').children[1].disabled, true)
  $('scan-axis-buttons').children[0].onclick()
  assert.equal(state.hits[0].path[0], null)
  assert.equal(state.hits[1].path[0], null)
  assert.equal(state.hits[2].path[0].y, 160)
  const range = state.axes[0].axis.range(200, 100)
  assert.equal(range.min, 10)
  assert.equal(range.max, 10)
  state.data.points = state.data.points.slice(0, 2)
  drawScan()
  assert.equal($('scan-axis-buttons').children[0].disabled, true)
  assert.equal(state.axes[0].axis.log, false)
  assert.ok(state.axes[0].axis.ticks.every(value => Number.isFinite(state.axes[0].axis.map(value))))
})

test('Parallel coordinates hide X and Y controls and scatter retains axis toggles', () => {
  const {state, $, drawScan} = explorer()
  assert.equal($('scan-x-label').hidden, true)
  assert.equal($('scan-y-label').hidden, true)
  assert.equal($('scan-axis-label').hidden, false)
  $('scan-plot').value = 'scatter'
  $('scan-plot').onchange()
  assert.equal($('scan-x-label').hidden, false)
  assert.equal($('scan-y-label').hidden, false)
  assert.equal($('scan-axis-label').hidden, true)
  drawScan()
  assert.deepEqual($('scan-axis-buttons').children.map(button => button.dataset.axis), ['a', '$metric'])
  $('scan-axis-buttons').children[0].onclick()
  assert.equal(state.axes[0].axis.log, true)
  $('scan-plot').value = 'parallel'
  drawScan()
  assert.equal(state.axes[0].axis.log, true)
  state.data.points = []
  drawScan()
  assert.equal($('scan-axis-buttons').children.length, 0)
})
