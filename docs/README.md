# Update Docs

## Prepare environment

```
apt-get update && apt-get install -y latexmk texlive-xetex fonts-noto fonts-freefont-otf xindy
pip install -r requirements.txt
```

## build

```
make latexpdf
```

## (optional) update api docs
If you add modules, you need to generate the corresponding api-doc

```
sphinx-apidoc -o ./source/apis ../torchacc/{module_name}
# update the contents based on exsiting api docs
```