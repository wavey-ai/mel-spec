const test = require("tap").test;
const { sharedbuffer, ringbuffer } = require("./../ringbuffer");

test("pop should return undefined when buffer is empty", (t) => {
  const frame_size = 4;
  const max_frames = 8;

  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);
  const poppedFrame = rb.pop();
  t.equal(
    poppedFrame,
    undefined,
    "Popping from an empty buffer should return undefined"
  );

  t.end();
});

test("fifo push and pop", (t) => {
  const frame_size = 4;
  const max_frames = 100;
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

  for (let i = 0; i < frames.length; i++) {
    rb.push(frames[i]);
  }

  for (let i = 0; i < frames.length; i++) {
    const res = rb.pop();
    t.same(res, frames[i], `frame ${i} should match`);
  }

  t.end();
});

test("fifo push and pop float", (t) => {
  const frame_size = 4;
  const max_frames = 100;
  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const frames = [
    new Float32Array([1, 2, 3.2, 4]),
    new Float32Array([5, 6.44, 7, 8]),
    new Float32Array([9.1, 10.22, 11, 12]),
    new Float32Array([13, 14, 15, 16]),
    new Float32Array([2, 2.2, 3, 4]),
  ];

  for (let i = 0; i < frames.length; i++) {
    rb.push(frames[i]);
  }

  for (let i = 0; i < frames.length; i++) {
    const res = rb.pop();
    t.same(res, frames[i], `frame ${i} should match`);
  }

  t.end();
});

test("push and pop with wrapping", (t) => {
  const frame_size = 4;
  const max_frames = 2;

  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const frame1 = new Uint8Array([1, 2, 3, 4]);
  const frame2 = new Uint8Array([5, 6, 7, 8]);
  const frame3 = new Uint8Array([9, 10, 11, 12]);
  const frame4 = new Uint8Array([13, 14, 15, 16]);

  rb.push(frame1);
  rb.push(frame2);

  t.same(rb.pop(), frame1);
  t.same(rb.pop(), frame2);

  rb.push(frame3);
  rb.push(frame4);

  t.same(rb.pop(), frame3);
  t.same(rb.pop(), frame4);

  t.end();
});

test("push and pop with wrapping overwrite", (t) => {
  const frame_size = 4;
  const max_frames = 2;

  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const frame1 = new Uint8Array([1, 2, 3, 4]);
  const frame2 = new Uint8Array([5, 6, 7, 8]);
  const frame3 = new Uint8Array([9, 10, 11, 12]);
  const frame4 = new Uint8Array([13, 14, 15, 16]);

  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);

  t.same(rb.pop(), frame3);
  t.same(rb.pop(), frame4);
  t.same(rb.pop(), undefined);

  t.same(rb.dropped_count(), 2);

  t.end();
});

test("push and pop with multiple wrapping overwrite", (t) => {
  const frame_size = 4;
  const max_frames = 2;

  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const frame1 = new Uint8Array([1, 2, 3, 4]);
  const frame2 = new Uint8Array([5, 6, 7, 8]);
  const frame3 = new Uint8Array([9, 10, 11, 12]);
  const frame4 = new Uint8Array([13, 14, 15, 16]);

  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);
  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);

  t.same(rb.pop(), frame3);
  t.same(rb.pop(), frame4);
  t.same(rb.pop(), undefined);

  t.same(rb.dropped_count(), 6);
  t.end();
});

test("float data type multiple", (t) => {
  const frame_size = 4;
  const max_frames = 2;

  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const frame1 = new Float32Array([1, 2, 3, 4.23]);
  const frame2 = new Float32Array([5, 6.44, 2, 7]);
  const frame3 = new Float32Array([9, 10, 11, 12]);
  const frame4 = new Float32Array([13.2, 14, 15, 16]);

  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);
  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);

  t.same(rb.pop(), frame3);
  t.same(rb.pop(), frame4);
  t.same(rb.pop(), undefined);

  t.same(rb.dropped_count(), 6);

  t.end();
});

test("float data type", (t) => {
  const frame_size = 4;
  const max_frames = 2;

  const sb = sharedbuffer(frame_size, max_frames, Float32Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Float32Array);

  const frame1 = new Float32Array([1, 2, 3, 4.23]);
  const frame2 = new Float32Array([5, 6.44, 2, 7]);
  const frame3 = new Float32Array([9, 10, 11, 12]);
  const frame4 = new Float32Array([13.2, 14, 15, 16]);

  rb.push(frame1);
  rb.push(frame2);
  rb.push(frame3);
  rb.push(frame4);

  t.same(rb.pop(), frame3);
  t.same(rb.pop(), frame4);
  t.same(rb.pop(), undefined);

  t.same(rb.dropped_count(), 2);

  t.end();
});

test("push and pop with", (t) => {
  const frame_size = 2;
  const max_frames = 64;

  const sb = sharedbuffer(frame_size, max_frames, Uint8Array);
  const rb = ringbuffer(sb, frame_size, max_frames, Uint8Array);

  const a = [];
  for (let i = 0; i < 100; i++) {
    a.push(new Uint8Array([i, 0]));
  }
  const b = [];
  for (let i = 0; i < 100; i++) {
    b.push(new Uint8Array([i, 1]));
  }

  for (let i = 0; i < a.length; i++) {
    rb.push(a[i]);
    rb.push(b[i]);

    const res = [];

    while (true) {
      let r = rb.pop();
      if (r) {
        res.push(r);
      } else {
        break;
      }
    }

    t.same(res[0], a[i]);
    t.same(res[1], b[i]);
    t.same(res.length, 2);
  }

  t.end();
});
