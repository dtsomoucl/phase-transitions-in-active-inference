# About this Third-Party Code

This folder contains local source copies of external packages used by the active-inference simulations in this workspace.

## `pymdp`

- Source repository: `https://github.com/infer-actively/pymdp`
- Local reference commit used for this port: `23c206f`
- Local package path: `third_party/pymdp`
- License file copied to: `third_party/pymdp_LICENSE`

The copy is included here only to make the balance-scale `pymdp` port runnable inside this workspace without depending on a temporary `/tmp` checkout.

## Local compatibility note

This vendored copy includes a small compatibility patch: optional plotting dependencies such as `seaborn` are not required merely to import the core `pymdp` agent code. If you instead copy the upstream GitHub repository verbatim, you may need to install `seaborn` (and related plotting packages) or apply the same patch to `pymdp/utils.py` and `pymdp/envs/grid_worlds.py`.
