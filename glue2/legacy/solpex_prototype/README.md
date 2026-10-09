# SOLPEx B2.5 Coupler

Prototype interface for a gated SOLSTICE surrogate inside B2.5, intended to
reduce EIRENE calls while retaining EIRENE as the authoritative fallback.

> **Prototype status:** this code is not yet safe for closed-loop coupling. The
> audited data contract, known implementation issues, and a staged
> EIRENE-fallback plan are documented in
> [the SOLPS--EIRENE--SOLSTICE coupling note](docs/solps_eirene_solstice_coupling_note.pdf)
> ([LaTeX source](docs/solps_eirene_solstice_coupling_note.tex)).

The current 5-field-in/4-field-out socket is a reduced legacy prototype. It
does **not** reproduce the raw EIRENE return contract or B2's downstream
regularisation. See the executed
[`fort.31` explorer notebook](notebooks/fort31_explorer.ipynb) for the complete
B2-to-EIRENE background record inventory in an example run.

The observation-only raw data logger now under development is described in
[EIRENE training dump](docs/eirene_training_dump.md). It is intentionally
separate from the legacy socket prototype.

## Architecture

The diagram below documents the existing prototype, not the recommended final
contract. The production design will route between NN and EIRENE while keeping
the full raw return arrays and the normal B2 postprocessing path.

```
B2.5 (Fortran)                     SOLPEx (Python/PyTorch)
     |                                    |
  b2mod_eirene_nn.F                solpex_b2_server.py
     |  pack Te,Ti,ne,ni,ua              |  load GNN model
     |                                    |  listen on socket
     +---> nn_interface.c ---[socket]---->+
     |     send 5*ny*nx f64              |  build graph
     |                                    |  GNN forward pass (~1ms)
     +<--- nn_interface.c <--[socket]----+
     |     recv 4*ny*nx f64              |  return Sp,Qe,Qi,Sm
  unpack into sna0,smo0,she0,shi0        |
     |                                    |
  continue B2.5 time step                |
```

## Files

| File | Language | Purpose |
|------|----------|---------|
| `solpex_b2_server.py` | Python | Socket server, holds GNN in GPU memory |
| `nn_interface.c` | C | Socket client, called from Fortran |
| `b2mod_eirene_nn.F` | Fortran | B2.5 wrapper: pack/unpack arrays |

## Usage

### 1. Start the server

```bash
python coupler/solpex_b2_server.py \
    --model scripts/outputs/eirene_gnn.pt \
    --mesh /path/to/coupling_dataset.npz \
    --socket /tmp/solpex.sock \
    --device cuda
```

### 2. Run B2.5

Add to `b2mn.dat`:
```
b2mndr_eirene_nn  1
b2mndr_nn_socket  '/tmp/solpex.sock'
```

### 3. Build

Compile `nn_interface.c` and link with B2.5:
```bash
gcc -c -O2 coupler/nn_interface.c -o nn_interface.o
# Add nn_interface.o and b2mod_eirene_nn.o to B2.5 link step
```

## Protocol

Binary, little-endian over Unix domain socket:

1. **Client sends** header: `nx(i32), ny(i32), ns(i32)` (12 bytes)
2. **Client sends** plasma: `5 * ny * nx` float64 (Te, Ti, ne, ni, ua)
3. **Server sends** sources: `4 * ny * nx` float64 (Sp, Qe, Qi, Sm)

Arrays are Fortran column-major (interior cells only, no guard cells).
