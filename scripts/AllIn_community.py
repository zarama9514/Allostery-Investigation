from __future__ import annotations

from pathlib import Path
from typing import Sequence

import MDAnalysis as mda
import networkx as nx
import numpy as np

from AllIn_DCCM import DCCMAnalyzer


class CommunityAnalyzer:
    def __init__(self, psf: str, dcd: str | Sequence[str]):
        self.psf = psf
        self.dcd = dcd

    @staticmethod
    def _graph_from_dccm(dccm: np.ndarray, threshold: float) -> nx.Graph:
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        graph = nx.Graph()
        n = int(dccm.shape[0])
        graph.add_nodes_from(range(n))
        for i in range(n):
            row = dccm[i]
            for j in range(i + 1, n):
                w = float(abs(row[j]))
                if w > threshold:
                    graph.add_edge(i, j, weight=w)
        return graph

    @staticmethod
    def _communities(graph: nx.Graph, seed: int = 42) -> dict[int, int]:
        if graph.number_of_nodes() == 0:
            return {}
        if graph.number_of_edges() == 0:
            return {int(node): int(node) for node in graph.nodes}
        communities = nx.community.louvain_communities(graph, seed=seed, weight="weight")
        return {int(node): int(cid) for cid, nodes in enumerate(communities) for node in nodes}

    def run(
        self,
        output_pdb: str,
        selection: str = "protein and name CA",
        align_selection: str = "protein and backbone",
        step: int = 10,
        skip_first_n_frames: int = 0,
        threshold: float = 0.5,
        seed: int = 42,
        output_frame_index: int = 0,
    ) -> dict[str, int | str]:
        dccm_result = DCCMAnalyzer(psf1=self.psf, dcd1=self.dcd).calculate(
            selection1=selection,
            align_selection1=align_selection,
            step=step,
            skip_first_n_frames=skip_first_n_frames,
        )
        dccm = np.asarray(dccm_result["dccm"], dtype=float)
        graph = self._graph_from_dccm(dccm, threshold=threshold)
        community_map = self._communities(graph, seed=seed)
        universe = mda.Universe(self.psf, self.dcd)
        universe.trajectory[output_frame_index]
        try:
            _ = universe.atoms.tempfactors
        except Exception:
            universe.add_TopologyAttr("tempfactors")
        universe.atoms.tempfactors = 0.0
        ca_atoms = universe.select_atoms(selection)
        if ca_atoms.n_atoms != dccm.shape[0]:
            raise ValueError("Selected atom count does not match DCCM size")
        for idx, atom in enumerate(ca_atoms):
            community_id = int(community_map.get(idx, -1))
            atom.residue.atoms.tempfactors = float(community_id)
        output_path = Path(output_pdb)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        universe.atoms.write(str(output_path))
        return {
            "output_pdb": str(output_path),
            "n_communities": int(len(set(community_map.values()))),
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "n_frames_used": int(dccm_result["n_frames_used"]),
        }
