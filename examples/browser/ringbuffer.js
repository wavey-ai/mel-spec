if (typeof module !== 'undefined') {
  module.exports = { ringbuffer, sharedbuffer };
}

function sharedbuffer(frame_size, max_frames, dataType) {
  if ((max_frames & (max_frames - 1)) !== 0) {
    throw new RangeError('max_frames must be a power of two');
  }
  const data_offset = 6 * 4;
  const el_bytes = dataType.BYTES_PER_ELEMENT;
  const sab = new SharedArrayBuffer((frame_size * el_bytes * max_frames) + data_offset);
  return sab;
}

function ringbuffer(sab, frame_size, max_frames, dataType) {
  const mask = max_frames - 1;
  const hdrBytes = 6 * 4;
  const in_b = new Uint32Array(sab, 0, 1);
  const out_b = new Uint32Array(sab, 4, 1);
  const dropped_b = new Uint32Array(sab, 8, 1);
  const w_ptr_b = new Uint32Array(sab, 12, 1);
  const r_ptr_b = new Uint32Array(sab, 16, 1);
  const wrap_flag_b = new Uint32Array(sab, 20, 1);
  const data_b = new dataType(sab, hdrBytes, max_frames * frame_size);

  const in_count = () => Atomics.load(in_b, 0);
  const out_count = () => Atomics.load(out_b, 0);
  const w_ptr = () => Atomics.load(w_ptr_b, 0);
  const r_ptr = () => Atomics.load(r_ptr_b, 0);
  const wrap_flag = () => Atomics.load(wrap_flag_b, 0) === 1;

  const wrapping_add = i => (i + 1) & mask;
  const current_offset = i => i * frame_size;

  const push = frame => {
    const write = w_ptr();
    data_b.set(frame, write * frame_size);
    if (write === 0) Atomics.store(wrap_flag_b, 0, 1);
    if (wrap_flag() && r_ptr() === write) {
      Atomics.store(wrap_flag_b, 0, 0);
      const dropped = dropped_b[0] + (in_count() - out_count());
      Atomics.store(dropped_b, 0, dropped);
      Atomics.store(out_b, 0, in_count());
    }
    Atomics.add(in_b, 0, 1);
    Atomics.store(w_ptr_b, 0, wrapping_add(write));
    return true;
  };

  const count = () => {
    return in_count() - out_count();
  };

  const pop = () => {
    if (in_count() - out_count() === 0) return;
    const read = r_ptr();
    const start = read * frame_size;
    const end = start + frame_size;
    const res = data_b.slice(start, end);
    Atomics.add(out_b, 0, 1);
    Atomics.store(r_ptr_b, 0, wrapping_add(read));
    return res;
  };


  return { sab, push, pop, dropped_count: () => Atomics.load(dropped_b, 0), count: () => in_count() - out_count() };
}


