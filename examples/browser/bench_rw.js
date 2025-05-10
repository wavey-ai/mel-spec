'use strict'
const { Worker } = require('worker_threads')
const { performance } = require('perf_hooks')
const path = require('path')
const { sharedbuffer, ringbuffer } = require('./ringbuffer')

const ELEMENT = Int16Array
const FRAME_SIZE = 320
const RING_FRAMES = 8192
const WINDOW_MS = 5000

const rbPath = path.join(__dirname, 'ringbuffer.js')
const sab = sharedbuffer(FRAME_SIZE, RING_FRAMES, ELEMENT)
ringbuffer(sab, FRAME_SIZE, RING_FRAMES, ELEMENT)

const stopSab = new SharedArrayBuffer(4)
const stopFlag = new Uint32Array(stopSab)

const workerSrc = role => `
  const { parentPort, workerData } = require('worker_threads')
  const { ringbuffer } = require(workerData.rbPath)
  const RB = ringbuffer(workerData.sab,
                        workerData.frame,
                        workerData.ring,
                        global[workerData.type])
  const stop = new Uint32Array(workerData.stopSab)
  const frame = new global[workerData.type](workerData.frame)
  let pushes = 0, pops = 0, dropped0 = RB.dropped_count()
  if (${role === 'prod'}) {
    while (!Atomics.load(stop,0)) { RB.push(frame); pushes++ }
    parentPort.postMessage({ pushes })
  } else {
    while (!Atomics.load(stop,0)) if (RB.pop()) pops++
    const dropped = RB.dropped_count() - dropped0
    parentPort.postMessage({ pops, dropped })
  }
`
const mkWorker = r => new Worker(workerSrc(r), {
  eval: true,
  workerData: { rbPath, sab, stopSab, frame: FRAME_SIZE, ring: RING_FRAMES, type: ELEMENT.name }
})
const producer = mkWorker('prod')
const consumer = mkWorker('cons')

const t0 = performance.now()
setTimeout(() => Atomics.store(stopFlag, 0, 1), WINDOW_MS)

const res = {}
function done(role, msg) {
  res[role] = msg
  if (res.prod && res.cons) {
    const sec = (performance.now() - t0) / 1000
    const total = res.prod.pushes
    const pushes_s = res.prod.pushes / sec
    const pops_s = res.cons.pops / sec
    const drops_s = res.cons.dropped / sec
    const pop_pct = (res.cons.pops / total * 100).toFixed(1)
    const drop_pct = (res.cons.dropped / total * 100).toFixed(1)
    const nf = x => Intl.NumberFormat('en-US').format(Math.round(x))
    console.log('\nelapsed  |  writes (ops/s)  reads (ops/s)  drops (ops/s)  read%   drop%')
    console.log(`${sec.toFixed(2)}s | ${nf(pushes_s).padStart(16)}  ${nf(pops_s).padStart(14)}  ${nf(drops_s).padStart(14)}  ${pop_pct.padStart(5)}  ${drop_pct.padStart(6)}`)
  }
}
producer.on('message', m => done('prod', m))
consumer.on('message', m => done('cons', m))

