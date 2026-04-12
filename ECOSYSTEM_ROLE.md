# Ecosystem Role

**Repository role:** NumPy-native package for array-based RGE256 workflows

This repository keeps the NumPy-facing counter-mode implementation separate from the broader multi-variant suite and from the browser demos. It is the cleanest choice for scientific Python workflows that need deterministic array generation without PyTorch.

## Keep Here
- the installable NumPy package
- reproducible array-generation APIs
- package-level tests and release notes
- user documentation for scientific Python workflows

## Keep Out
- broader family comparisons better handled in `rge256`
- browser-facing demo work
- the full research-program narrative

## Ecosystem Links
- Main research program: https://github.com/RRG314/Recursive-Geometric-Entropy-research
- Core suite: https://github.com/RRG314/rge256
- PyTorch package: https://github.com/RRG314/torchrge256
- Full demo app: https://github.com/RRG314/RGE-256-app
