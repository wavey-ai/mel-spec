build:
	rm -rf dist && mkdir dist
	cd ../../mel_spec_pipeline && wasm-pack build --target no-modules --no-typescript --out-dir ../examples/browser/dist --debug
	cat ringbuffer.js worklet.js > dist/worklet.js
