# Orca-Engine: R/D Analysis of Orca Vocalisations

The tenth engine. The first whose kernel must be **discovered**, not defined.

## Architecture

Same R/D substrate as lor-engine, quenya-engine, sindarin-engine, velahn-engine.
Same seven opcodes. Same 7×7 Genesis table. Same Gray-Scott dynamics.

The key innovation: **dual semantic field** — acoustic topology (what calls
sound like) and context topology (when calls are used) running in parallel.
Cross-pod alignment via Procrustes rotation IS translation between dialects.

## Modules

| Module | Status | Source |
|---|---|---|
| `address.py` | Verbatim copy | lor-engine |
| `kernel.py` | Adapted (OrcaRoot, CSV parse) | lor-engine |
| `features.py` | New (50D acoustic features) | — |
| `field.py` | Adapted (dual-field R/D) | lor-engine |
| `phonology.py` | Copy (Genesis) + new (boundary smoothing) | lor-engine |
| `opcodes.py` | New (DBSCAN discovery) + fallback (Genesis) | lor-engine |
| `compounds.py` | New (sequence detection) | lor-engine pattern |
| `alignment.py` | New (Procrustes cross-pod) | — |
| `trie.py` | Near-verbatim | lor-engine |
| `lookup.py` | Adapted | lor-engine |
| `navigate.py` | Adapted | lor-engine |
| `cli.py` | Adapted | lor-engine |

## Usage

```bash
cd orca-engine
python -m orca              # interactive REPL
python -m orca "S1"         # single query
python -m orca "align:J:K"  # cross-pod alignment
python -m pytest tests/ -v  # run tests
```

## Next Steps

1. Load real SRKW call catalogues from published data
2. Extract acoustic features from hydrophone recordings (librosa)
3. Run opcode discovery on variable vs discrete call pairs
4. Cross-pod alignment between J, K, L pods with real data
5. Validate: do aligned calls share behavioural contexts?
