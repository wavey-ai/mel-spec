build:
	rm -rf dist && mkdir dist
	cd ../../mel_spec_pipeline && wasm-pack build --target no-modules --no-typescript --out-dir ../examples/browser/dist --debug
	cd ../../mel_spec_audio && wasm-pack build --target no-modules --no-typescript --out-dir ../examples/browser/dist --debug
	sed -i.bak "s/wasm_bindgen/wasm_bindgen_wav/g" dist/mel_spec_audio.js
	cat ringbuffer.js worklet.js > dist/worklet.js
