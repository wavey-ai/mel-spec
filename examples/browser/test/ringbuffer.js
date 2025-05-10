const test = require('tap').test;
const { sharedbuffer, ringbuffer } = require('./../ringbuffer');

// Empty buffer pop
test('pop should return undefined when buffer is empty', t => {
  const frame_size = 4;
  const max_frames = 8;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  t.equal(rb.pop(), undefined, 'Popping from an empty buffer should return undefined');
  t.end();
});

// FIFO for Uint8Array
test('fifo push and pop', t => {
  const frame_size = 4;
  const max_frames = 128;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const frames = [
    new Uint8Array([1, 2, 3, 4]),
    new Uint8Array([5, 6, 7, 8]),
    new Uint8Array([9, 10, 11, 12]),
    new Uint8Array([13, 14, 15, 16]),
    new Uint8Array([2, 2, 3, 4]),
    new Uint8Array([3, 6, 7, 8]),
    new Uint8Array([4, 10, 11, 12]),
    new Uint8Array([5, 14, 15, 16]),
  ];

  for (const frame of frames) rb.push(frame);
  for (let i = 0; i < frames.length; i++) {
    t.same(rb.pop(), frames[i], `frame ${i} should match`);
  }

  t.end();
});

// FIFO for Float32Array
test('fifo push and pop float', t => {
  const frame_size = 4;
  const max_frames = 128;
  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const frames = [
    new Float32Array([1, 2, 3.2, 4]),
    new Float32Array([5, 6.44, 7, 8]),
    new Float32Array([9.1, 10.22, 11, 12]),
    new Float32Array([13, 14, 15, 16]),
    new Float32Array([2, 2.2, 3, 4]),
  ];

  for (const frame of frames) rb.push(frame);
  for (let i = 0; i < frames.length; i++) {
    t.same(rb.pop(), frames[i], `frame ${i} should match`);
  }

  t.end();
});

// Wrapping without overwrite
test('push and pop with wrapping', t => {
  const frame_size = 4;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const [f1, f2, f3, f4] = [
    new Uint8Array([1, 2, 3, 4]),
    new Uint8Array([5, 6, 7, 8]),
    new Uint8Array([9, 10, 11, 12]),
    new Uint8Array([13, 14, 15, 16]),
  ];

  rb.push(f1);
  rb.push(f2);
  t.same(rb.pop(), f1);
  t.same(rb.pop(), f2);

  rb.push(f3);
  rb.push(f4);
  t.same(rb.pop(), f3);
  t.same(rb.pop(), f4);

  t.end();
});

// Wrapping with overwrite
test('push and pop with wrapping overwrite', t => {
  const frame_size = 4;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const [f1, f2, f3, f4] = [
    new Uint8Array([1, 2, 3, 4]),
    new Uint8Array([5, 6, 7, 8]),
    new Uint8Array([9, 10, 11, 12]),
    new Uint8Array([13, 14, 15, 16]),
  ];

  rb.push(f1);
  rb.push(f2);
  rb.push(f3);
  rb.push(f4);

  t.same(rb.pop(), f3);
  t.same(rb.pop(), f4);
  t.equal(rb.pop(), undefined);
  t.equal(rb.dropped_count(), 2);

  t.end();
});

// Multiple overwrite
test('push and pop with multiple wrapping overwrite', t => {
  const frame_size = 4;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const [f1, f2, f3, f4] = [
    new Uint8Array([1, 2, 3, 4]),
    new Uint8Array([5, 6, 7, 8]),
    new Uint8Array([9, 10, 11, 12]),
    new Uint8Array([13, 14, 15, 16]),
  ];

  for (const f of [f1, f2, f3, f4, f1, f2, f3, f4]) rb.push(f);

  t.same(rb.pop(), f3);
  t.same(rb.pop(), f4);
  t.equal(rb.pop(), undefined);
  t.equal(rb.dropped_count(), 6);

  t.end();
});

// Float32 overwrite
test('float data type multiple', t => {
  const frame_size = 4;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const [f1, f2, f3, f4] = [
    new Float32Array([1, 2, 3, 4.23]),
    new Float32Array([5, 6.44, 2, 7]),
    new Float32Array([9, 10, 11, 12]),
    new Float32Array([13.2, 14, 15, 16]),
  ];

  for (const f of [f1, f2, f3, f4, f1, f2, f3, f4]) rb.push(f);

  t.same(rb.pop(), f3);
  t.same(rb.pop(), f4);
  t.equal(rb.pop(), undefined);
  t.equal(rb.dropped_count(), 6);

  t.end();
});

// Float32 simple
test('float data type', t => {
  const frame_size = 4;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const [f1, f2, f3, f4] = [
    new Float32Array([1, 2, 3, 4.23]),
    new Float32Array([5, 6.44, 2, 7]),
    new Float32Array([9, 10, 11, 12]),
    new Float32Array([13.2, 14, 15, 16]),
  ];

  for (const f of [f1, f2, f3, f4]) rb.push(f);

  t.same(rb.pop(), f3);
  t.same(rb.pop(), f4);
  t.equal(rb.pop(), undefined);
  t.equal(rb.dropped_count(), 2);

  t.end();
});

// Paired push/pop
test('push and pop with paired pushes', t => {
  const frame_size = 2;
  const max_frames = 64;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const a = [...Array(100).keys()].map(i => new Uint8Array([i, 0]));
  const b = [...Array(100).keys()].map(i => new Uint8Array([i, 1]));

  for (let i = 0; i < a.length; i++) {
    rb.push(a[i]);
    rb.push(b[i]);

    const res = [];
    while (true) {
      const r = rb.pop();
      if (r) res.push(r);
      else break;
    }

    t.same(res[0], a[i]);
    t.same(res[1], b[i]);
    t.equal(res.length, 2);
  }

  t.end();
});

// count() accuracy across wrap
test('count reflects pushes/pops across wrap', t => {
  const frame_size = 3;
  const max_frames = 4;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  t.equal(rb.count(), 0, 'starts empty');
  rb.push(new Uint8Array([1, 2, 3]));
  t.equal(rb.count(), 1);
  rb.push(new Uint8Array([4, 5, 6]));
  rb.pop();
  t.equal(rb.count(), 1, 'one left after one pop');
  rb.push(new Uint8Array([7, 8, 9])); // wrap
  rb.push(new Uint8Array([10, 11, 12])); // overwrite
  t.equal(rb.count(), 3, 'full again after wrap overwrite');
  rb.pop(); rb.pop(); rb.pop();
  t.equal(rb.count(), 0, 'back to empty');

  t.end();
});

// single-slot buffer
test('single-slot buffer behaves correctly', t => {
  const frame_size = 1;
  const max_frames = 1;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  rb.push(new Uint8Array([42]));
  t.same(rb.pop(), new Uint8Array([42]));
  rb.push(new Uint8Array([99])); // overwrite
  rb.push(new Uint8Array([100])); // overwrite again
  t.equal(rb.dropped_count(), 1, 'one frame dropped');
  t.same(rb.pop(), new Uint8Array([100]));
  t.equal(rb.pop(), undefined);

  t.end();
});

// synthetic counter wraparound
test('counter wraparound still yields correct count', t => {
  const frame_size = 1;
  const max_frames = 2;
  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const in_b = new Uint32Array(sb, 0, 1);
  const out_b = new Uint32Array(sb, 4, 1);
  in_b[0] = 0xFFFFFFFE;
  out_b[0] = 0xFFFFFFFE;

  rb.push(new Uint8Array([1]));
  t.equal(rb.count(), 1, 'handles 0xFFFFFFFF → 0 wrap');
  rb.pop();
  t.equal(rb.count(), 0);
  t.end();
});

// multi-thread smoke
const { Worker } = require('worker_threads');
const path = require('path');

test('concurrent producer / consumer', { timeout: 20000 }, t => {
  const frameSize = 8;
  const loops = 200000;
  const maxFrames = 1 << Math.ceil(Math.log2(loops + 32));
  const sb = sharedbuffer(frameSize, maxFrames, Uint8Array);
  const rb = ringbuffer(sb, frameSize, maxFrames, Uint8Array);
  const rbPath = path.join(__dirname, '..', 'ringbuffer.js');

  const worker = new Worker(`
    const { parentPort, workerData } = require('worker_threads');
    const { ringbuffer } = require(workerData.rbPath);
    const rb = ringbuffer(workerData.sb, workerData.frameSize, workerData.maxFrames, Uint8Array);
    const payload = new Uint8Array(workerData.frameSize);
    for (let i = 0; i < workerData.loops; i++) rb.push(payload);
    parentPort.postMessage('done');
  `, { eval: true, workerData: { sb, frameSize, maxFrames, loops, rbPath } });

  let popped = 0;
  (function drain() {
    while (rb.pop()) popped++;
    if (popped >= loops) {
      worker.terminate();
      t.equal(rb.count(), 0);
      t.equal(rb.dropped_count(), 0);
      t.end();
    } else {
      setImmediate(drain);
    }
  })();
});

// lap test: reader always gets latest
test('producer overtakes reader ⇒ reader still gets latest value', { timeout: 10000 }, t => {
  const frameSize = 1;
  const maxFrames = 128;
  const totalWrites = 10000;

  const sb = sharedbuffer(frameSize, maxFrames, Uint32Array);
  const rb = ringbuffer(sb, frameSize, maxFrames, Uint32Array);

  for (let i = 0; i < totalWrites; i++) {
    rb.push(new Uint32Array([i]));
  }

  let last = -1;
  let popped = 0;
  let f;
  while ((f = rb.pop())) {
    last = f[0];
    popped++;
  }

  t.equal(popped + rb.dropped_count(), totalWrites, 'all writes accounted for');
  t.equal(last, totalWrites - 1, 'reader sees the very latest value');
  t.end();
});

