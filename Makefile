.PHONY: format

format:
	isort torchacc/
	yapf -i -r *.py torchacc/ tests/ benchmarks/

test:
	PJRT_USE_TORCH_ALLOCATOR=true python -m pytest -v --maxfail=1 -k 'not flash_attn' ./tests/
	PJRT_USE_TORCH_ALLOCATOR=true python -m pytest -v --maxfail=1 -k 'flash_attn' -n 4 ./tests/
