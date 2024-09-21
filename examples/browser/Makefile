.PHONEY:
clean: 
	rm -rf dist _soundkit && mkdir dist

.PHONEY:
wasm:
	cd ../../ && wasm-pack build --features=wasm --target no-modules --no-typescript --out-dir ./examples/browser/dist --debug
	git clone git@github.com:wavey-ai/soundkit.git _soundkit && cd _soundkit && wasm-pack build --features=wasm --target no-modules --no-typescript --out-dir ../dist --debug
	rm -rf _soundkit

.PHONEY:
build: clean wasm
	sed -i.bak "s/wasm_bindgen/wasm_bindgen_wav/g" dist/soundkit.js
	cat ringbuffer.js worklet.js > dist/worklet.js
