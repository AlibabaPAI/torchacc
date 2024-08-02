.PHONY: format

format:
	isort torchacc/
	yapf -i -r *.py torchacc/ tests/ benchmarks/
