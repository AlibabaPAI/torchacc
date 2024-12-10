.PHONY: format

format:
	isort torchacc/
	yapf -i -r *.py torchacc/ tests/ benchmarks/

test:
	PJRT_USE_TORCH_ALLOCATOR=true python -m pytest -v -k 'not flash_attn' ./tests/
	PJRT_USE_TORCH_ALLOCATOR=true python -m pytest -v -k -n 4 'flash_attn' ./tests/
