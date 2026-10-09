#!/usr/bin/env python3
"""
SOLPEx socket server for B2.5-GNN coupling.

Replaces EIRENE inside the B2.5 iteration loop. Listens on a Unix domain
socket, receives plasma state arrays from the Fortran side, runs the
EIRENE-replacement GNN, and returns volumetric source terms.

Protocol (binary, little-endian float64):
  1. Client sends header:  nx(i32), ny(i32), ns(i32)
  2. Client sends plasma:  5 * ny * nx float64  (Te, Ti, ne, ni, ua)
  3. Server sends sources: 4 * ny * nx float64  (Sp, Qe, Qi, Sm)

Usage:
  python coupler/solpex_b2_server.py --model eirene_gnn.pt --socket /tmp/solpex.sock
  python coupler/solpex_b2_server.py --model eirene_gnn.pt --socket /tmp/solpex.sock --device cuda
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn.eirene_gnn import EireneGNN
from gnn.data import _build_edges_masked


# ======================================================================
# Model loading
# ======================================================================

def load_eirene_gnn(model_path, device="cpu"):
    """Load trained EIRENE-replacement GNN and its normalizers."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = EireneGNN(
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        edge_dim=3,
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Input normalizer
    inp = ckpt.get("input_norm", {})
    x_mu = inp.get("x_mu", torch.zeros(14))
    x_std = inp.get("x_std", torch.ones(14))
    ea_mu = inp.get("ea_mu", torch.zeros(3))
    ea_std = inp.get("ea_std", torch.ones(3))

    # Target normalizer
    tgt = ckpt.get("target_norm", {})
    t_mu = tgt.get("mu", torch.zeros(9))
    t_std = tgt.get("std", torch.ones(9))

    return model, {
        "x_mu": x_mu.to(device), "x_std": x_std.to(device),
        "ea_mu": ea_mu.to(device), "ea_std": ea_std.to(device),
        "t_mu": t_mu.to(device), "t_std": t_std.to(device),
    }


# ======================================================================
# Graph construction for a single plasma state
# ======================================================================

def build_inference_graph(plasma_2d, mask_2d, R, Z, vol, norms, device):
    """Build a single PyG Data object from 2D plasma arrays.

    Parameters
    ----------
    plasma_2d : dict with keys Te, Ti, ne, ni, ua — each (ny, nx)
    mask_2d : (ny, nx) bool
    R, Z : (ny, nx) cell center coordinates
    vol : (ny, nx) cell volumes
    norms : dict with x_mu, x_std, ea_mu, ea_std

    Returns
    -------
    torch_geometric.data.Data ready for model forward pass
    """
    from torch_geometric.data import Data

    ny, nx = mask_2d.shape
    n_nodes = int(mask_2d.sum())

    # Build edges (4-connected on native grid)
    edges = _build_edges_masked(mask_2d)
    edge_index = torch.from_numpy(edges).to(device)

    # Edge attributes
    R_flat = R[mask_2d]
    Z_flat = Z[mask_2d]
    if edges.shape[1] > 0:
        dR = R_flat[edges[1]] - R_flat[edges[0]]
        dZ = Z_flat[edges[1]] - Z_flat[edges[0]]
        dist = np.sqrt(dR**2 + dZ**2)
        edge_attr = torch.from_numpy(
            np.stack([dR, dZ, dist], axis=-1).astype(np.float32)
        ).to(device)
    else:
        edge_attr = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # Node features: all 14 plasma channels
    # For the EIRENE GNN, input is [Te, Ti, ne, ni, ua, vol, hx, hy, bb0-bb3, R, Z]
    # From B2.5 we get Te, Ti, ne, ni, ua. Fill geometry from stored mesh.
    Te = plasma_2d["Te"][mask_2d]
    Ti = plasma_2d["Ti"][mask_2d]
    ne = plasma_2d["ne"][mask_2d]
    ni = plasma_2d["ni"][mask_2d]
    ua = plasma_2d["ua"][mask_2d]
    v = vol[mask_2d]
    r = R[mask_2d]
    z = Z[mask_2d]

    # hx, hy, bb0-bb3 are geometry — stored from training dataset
    # For now, zero-fill (they'll be provided by the mesh metadata)
    zeros = np.zeros(n_nodes, dtype=np.float32)

    x = torch.from_numpy(np.stack([
        Te, Ti, ne, ni, ua, v,
        zeros, zeros,             # hx, hy (placeholder)
        zeros, zeros, zeros, zeros,  # bb0-bb3 (placeholder)
        r, z,
    ], axis=-1).astype(np.float32)).to(device)

    # Normalize
    x = (x - norms["x_mu"]) / norms["x_std"]
    edge_attr = (edge_attr - norms["ea_mu"]) / norms["ea_std"]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_nodes,
    )


def inverse_symlog(y_n, mu, std):
    """Inverse symlog normalization."""
    y_t = y_n * std + mu
    return torch.sign(y_t) * (torch.exp(torch.abs(y_t).clamp(max=80)) - 1.0)


# ======================================================================
# Socket I/O helpers
# ======================================================================

def recv_exact(conn, nbytes):
    """Receive exactly nbytes from socket."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def send_all(conn, data):
    """Send all bytes."""
    conn.sendall(data)


# ======================================================================
# Server
# ======================================================================

class SolpexServer:
    """Unix socket server for B2.5-GNN coupling."""

    def __init__(self, model, norms, mesh_meta, device, socket_path):
        self.model = model
        self.norms = norms
        self.mesh_meta = mesh_meta  # dict with R, Z, vol, mask from training data
        self.device = device
        self.socket_path = socket_path
        self.call_count = 0
        self.total_time = 0.0

    def handle_request(self, conn):
        """Handle one predict request from B2.5."""
        # Read header: nx, ny, ns (3 x int32)
        header = recv_exact(conn, 12)
        nx, ny, ns = struct.unpack("<iii", header)

        # Read plasma arrays: 5 * ny * nx float64
        n_vals = 5 * ny * nx
        plasma_bytes = recv_exact(conn, n_vals * 8)
        plasma_flat = np.frombuffer(plasma_bytes, dtype=np.float64)

        # Unpack: Te, Ti, ne, ni, ua — each (ny, nx), Fortran column-major
        stride = ny * nx
        plasma_2d = {
            "Te": plasma_flat[0*stride:1*stride].reshape(ny, nx, order='F').astype(np.float32),
            "Ti": plasma_flat[1*stride:2*stride].reshape(ny, nx, order='F').astype(np.float32),
            "ne": plasma_flat[2*stride:3*stride].reshape(ny, nx, order='F').astype(np.float32),
            "ni": plasma_flat[3*stride:4*stride].reshape(ny, nx, order='F').astype(np.float32),
            "ua": plasma_flat[4*stride:5*stride].reshape(ny, nx, order='F').astype(np.float32),
        }

        # Build graph and predict
        t0 = time.time()
        mask = self.mesh_meta["mask"]
        graph = build_inference_graph(
            plasma_2d, mask,
            self.mesh_meta["R"], self.mesh_meta["Z"], self.mesh_meta["vol"],
            self.norms, self.device,
        )

        with torch.no_grad():
            pred_n = self.model(graph)

        # Inverse normalize to physical units
        sources_phys = inverse_symlog(
            pred_n, self.norms["t_mu"], self.norms["t_std"]
        ).cpu().numpy().astype(np.float64)

        elapsed = time.time() - t0
        self.call_count += 1
        self.total_time += elapsed

        # Unpack: Sp(0), Qe(2), Qi(3), Sm(4) — indices in SOURCE_KEYS
        # Map from GNN output [Sp, Sne, Qe, Qi, Sm, dab2, dmb2, tab2, tmb2]
        # to B2.5 arrays [Sp, Qe, Qi, Sm]
        n_nodes = mask.sum()
        Sp = np.zeros((ny, nx), dtype=np.float64)
        Qe = np.zeros((ny, nx), dtype=np.float64)
        Qi = np.zeros((ny, nx), dtype=np.float64)
        Sm = np.zeros((ny, nx), dtype=np.float64)

        Sp[mask] = sources_phys[:, 0]   # Sp
        Qe[mask] = sources_phys[:, 2]   # Qe
        Qi[mask] = sources_phys[:, 3]   # Qi
        Sm[mask] = sources_phys[:, 4]   # Sm

        # Pack as Fortran column-major flat array: 4 * ny * nx float64
        out = np.concatenate([
            Sp.flatten(order='F'),
            Qe.flatten(order='F'),
            Qi.flatten(order='F'),
            Sm.flatten(order='F'),
        ])
        send_all(conn, out.tobytes())

        if self.call_count % 10 == 0:
            avg_ms = (self.total_time / self.call_count) * 1000
            print(f"  [solpex] call {self.call_count}: {elapsed*1000:.1f}ms "
                  f"(avg {avg_ms:.1f}ms)", flush=True)

    def serve(self):
        """Main server loop."""
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        print(f"[solpex] Listening on {self.socket_path}", flush=True)
        print(f"[solpex] Model on {self.device}, waiting for B2.5...", flush=True)

        # Handle Ctrl+C gracefully
        def cleanup(sig, frame):
            print("\n[solpex] Shutting down...")
            sock.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            sys.exit(0)
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        while True:
            conn, _ = sock.accept()
            print("[solpex] B2.5 connected", flush=True)
            try:
                while True:
                    self.handle_request(conn)
            except ConnectionError:
                print(f"[solpex] B2.5 disconnected after {self.call_count} calls "
                      f"({self.total_time:.1f}s total inference)", flush=True)
                conn.close()


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="SOLPEx B2.5 socket server")
    ap.add_argument("--model", required=True, help="Path to eirene_gnn.pt")
    ap.add_argument("--mesh", required=True,
                    help="Path to coupling_dataset.npz (for mesh geometry)")
    ap.add_argument("--socket", default="/tmp/solpex.sock",
                    help="Unix socket path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"[solpex] Loading model from {args.model}...")
    model, norms = load_eirene_gnn(args.model, device=str(device))
    print(f"[solpex] Model loaded on {device}")

    # Load mesh geometry from training dataset
    print(f"[solpex] Loading mesh from {args.mesh}...")
    d = np.load(args.mesh, allow_pickle=True)
    # Use first run's geometry as reference
    mask = d["mask"][0].astype(bool)
    plasma = d["plasma"][0]  # (14, H, W)
    mesh_meta = {
        "mask": mask,
        "R": plasma[12].astype(np.float32),   # R is channel 12
        "Z": plasma[13].astype(np.float32),   # Z is channel 13
        "vol": plasma[5].astype(np.float32),  # vol is channel 5
    }
    ny, nx = mask.shape
    print(f"[solpex] Mesh: {ny}x{nx}, {mask.sum()} valid nodes")

    server = SolpexServer(model, norms, mesh_meta, device, args.socket)
    server.serve()


if __name__ == "__main__":
    main()
