# MaleCNS input files

Place the official MaleCNS v1.0 Feather downloads in this folder using these exact filenames:

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`

They are intentionally excluded from source control. Build the sparse artifact with:

```powershell
python brain.py --build-male-cns
```

The output is written to `artifacts/male_cns_connectome.npz` together with a body-ID index vector. Do not attempt to make this graph dense.
