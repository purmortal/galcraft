# Contributing and maintaining the documentation

## Build locally

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a web browser. The API pages are generated from NumPy-style Python docstrings using Sphinx autodoc and Napoleon.

## Publish with GitHub Pages

The repository includes `.github/workflows/docs.yml` using the GitHub Pages deployment actions. On GitHub, open **Settings → Pages**, then choose **GitHub Actions** as the build and deployment source. Push to `main` to trigger a build.

For the repository `purmortal/galcraft`, the intended published URL is:

`https://purmortal.github.io/galcraft/`

You do not need to copy HTML manually into your personal-website repository. The project repository can publish its own site under `/galcraft/`.

## Docstring conventions

Use NumPy-style sections such as `Parameters`, `Returns`, `Raises`, and `Notes`. Include units (Å, kpc, km/s, degrees) and shape/order for array arguments when known. Do not describe an operation that the code does not perform. Distinguish return values from in-place updates.

## Tests and changes

- Run `python -m compileall GalCraft` after editing code or docstrings.
- Run the repository's existing CI/test-kit integration checks for any change to scientific behavior.
- When changing configuration keys or output formats, update the corresponding pages in `docs/`.
- Do not store large SSP libraries or simulation outputs in the documentation source tree.
