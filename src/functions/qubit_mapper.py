# ============================================================
# SQM Research Project - Qubit Mapper (Hardware-Aware)
# Authors: Danny Valerio-Ramirez & Santiago Nunez-Corrales
# ============================================================

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx

# Configure matplotlib to use non-GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os

class QubitMapper:
    # Cost-function weights for noise-aware mapping
    _W_READOUT = 1.0
    _W_T1 = 1e-6  # scales T1 (in seconds) to ~µs contribution

    def __init__(self, backend):
        self.backend = backend
        self.coupling_map = backend.configuration().coupling_map
        self.n_qubits = backend.configuration().n_qubits
        self.graph = self._build_connectivity_graph()
        self.available_qubits: Set[int] = set(range(self.n_qubits))
        self.allocation_map: Dict[str, List[int]] = {}
        # Cache calibration data for noise-aware mapping
        self._props, self._gate_name = self._get_calibration_data()

    def _build_connectivity_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_qubits))
        for edge in self.coupling_map:
            graph.add_edge(edge[0], edge[1])
        return graph

    # ------------------------------------------------------------------
    # Calibration-aware helpers
    # ------------------------------------------------------------------

    def _get_calibration_data(self):
        """Cache backend.properties() and detect the native 2q gate name."""
        props = None
        gate_name = None
        try:
            if hasattr(self.backend, 'properties'):
                props = self.backend.properties()
        except Exception:
            pass
        if props is not None:
            gate_names_in_backend = {g.gate for g in props.gates}
            for candidate in ('cz', 'cx', 'ecr'):
                if candidate in gate_names_in_backend:
                    gate_name = candidate
                    break
        return props, gate_name

    def _evaluate_chain_cost(self, chain: List[int], is_sqm: bool) -> float:
        """Noise-aware cost: 2q-gate error + readout(ancilla) + 1/T1 decay."""
        if self._props is None or self._gate_name is None:
            return 0.0  # fallback: topology-only selection
        cost = 0.0
        # Sum 2q-gate error over consecutive edges in the chain
        for i in range(len(chain) - 1):
            u, v = chain[i], chain[i + 1]
            try:
                cost += self._props.gate_error(self._gate_name, [u, v])
            except Exception:
                try:
                    cost += self._props.gate_error(self._gate_name, [v, u])
                except Exception:
                    cost += 1.0  # penalise missing calibration
        # Readout error on ancilla (SQM chain[2] = tele_ancilla)
        if is_sqm and len(chain) >= 3:
            try:
                cost += self._W_READOUT * self._props.readout_error(chain[2])
            except Exception:
                pass
        # T1 decay penalty across all qubits in the chain
        for q in chain:
            try:
                t1_s = self._props.t1(q)
                if t1_s and t1_s > 0:
                    cost += self._W_T1 / t1_s
            except Exception:
                pass
        return cost

    def _enumerate_sqm_chains(self, exclude: Set[int]) -> List[List[int]]:
        """Enumerate all 4-qubit linear chains (q_work<->mem_orig<->tele<->mem_backup)
        with strict consecutive adjacency in available_qubits - exclude."""
        pool = self.available_qubits - exclude
        chains: List[List[int]] = []
        for a in pool:
            for b in self.graph.neighbors(a):
                if b not in pool or b == a:
                    continue
                for c in self.graph.neighbors(b):
                    if c not in pool or c in (a, b):
                        continue
                    for d in self.graph.neighbors(c):
                        if d not in pool or d in (a, b, c):
                            continue
                        chains.append([a, b, c, d])
        return chains

    def _enumerate_swap_chains(self, exclude: Set[int]) -> List[List[int]]:
        """Enumerate all 2-qubit pairs (q_work<->mem_0) with direct edge."""
        pool = self.available_qubits - exclude
        chains: List[List[int]] = []
        for a in pool:
            for b in self.graph.neighbors(a):
                if b in pool and b != a:
                    chains.append([a, b])
        return chains

    def find_connected_subgraph(self, size: int, preferred_start: Optional[int] = None) -> Optional[List[int]]:
        if size > len(self.available_qubits):
            return None
        if size == 0:
            return []
        start = (preferred_start if preferred_start in self.available_qubits else min(self.available_qubits))
        visited = {start}
        queue = [start]
        while queue and len(visited) < size:
            current = queue.pop(0)
            for neighbor in self.graph.neighbors(current):
                if neighbor in self.available_qubits and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) == size:
                        break
        return list(visited) if len(visited) == size else None

    def allocate_chain_topology(self, chain_config: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        is_sqm = any('tele_ancilla' in reg_name for reg_name, _ in chain_config)
        R = n = 0
        for reg_name, size in chain_config:
            if reg_name == "q_work":
                n = size
            elif reg_name.startswith("mem_orig_"):
                try:
                    reg_num = int(reg_name.split("_")[2])
                    R = max(R, reg_num + 1)
                except (IndexError, ValueError):
                    pass
            elif reg_name.startswith("mem_") and not reg_name.startswith("mem_orig") and not reg_name.startswith("mem_backup"):
                try:
                    reg_num = int(reg_name.split("_")[1])
                    R = max(R, reg_num + 1)
                except (IndexError, ValueError):
                    pass
        if R == 0 or n == 0:
            return self._allocate_generic_chain(chain_config)
        if is_sqm:
            return self.allocate_sqm_per_bit_topology(R=R, n=n)
        else:
            return self.allocate_swap_per_bit_topology(R=R, n=n)

    def allocate_sqm_per_bit_topology(self, R: int, n: int) -> Dict[str, List[int]]:
        result = {}
        result["q_work"] = []
        for r_idx in range(R):
            result[f"mem_orig_{r_idx}"] = []
            result[f"mem_backup_{r_idx}"] = []
            result[f"tele_ancilla_{r_idx}"] = []
        print(f"[QubitMapper] SQM Per-Bit Allocation (Noise-Aware)")
        print(f"  Target: R={R} memory pairs, n={n} qubits per register")
        total_qubits_needed = n * (1 + 3 * R)
        if total_qubits_needed > len(self.available_qubits):
            raise RuntimeError(f"[QubitMapper] Need {total_qubits_needed} qubits, have {len(self.available_qubits)}")
        for bit_idx in range(n):
            # Collect qubits already committed in this allocation pass
            committed: Set[int] = set()
            for v in result.values():
                committed.update(v)
            # --- R=0 chain: q_work <-> mem_orig_0 <-> tele_ancilla_0 <-> mem_backup_0 ---
            candidates = self._enumerate_sqm_chains(committed)
            if not candidates:
                raise RuntimeError(f"[QubitMapper] No valid 4-qubit SQM chain for bit {bit_idx}")
            # Select chain with minimum noise cost
            best = min(candidates, key=lambda ch: self._evaluate_chain_cost(ch, is_sqm=True))
            q_center, mem_orig_0, tele_0, mem_backup_0 = best
            result["q_work"].append(q_center)
            result["mem_orig_0"].append(mem_orig_0)
            result["tele_ancilla_0"].append(tele_0)
            result["mem_backup_0"].append(mem_backup_0)
            for q in best:
                self.available_qubits.discard(q)
            # --- Additional memory pairs (r_idx >= 1) branch from q_center ---
            for r_idx in range(1, R):
                committed_now: Set[int] = set()
                for v in result.values():
                    committed_now.update(v)
                # Need 3-qubit branch: q_center <-> mem_orig_r <-> tele_r <-> mem_backup_r
                pool = self.available_qubits - committed_now
                branch_candidates: List[List[int]] = []
                for b in self.graph.neighbors(q_center):
                    if b not in pool:
                        continue
                    for c in self.graph.neighbors(b):
                        if c not in pool or c == b:
                            continue
                        for d in self.graph.neighbors(c):
                            if d not in pool or d in (b, c):
                                continue
                            branch_candidates.append([q_center, b, c, d])
                if not branch_candidates:
                    raise RuntimeError(f"[QubitMapper] No valid branch for mem pair r={r_idx}, bit {bit_idx}")
                best_branch = min(branch_candidates,
                                  key=lambda ch: self._evaluate_chain_cost(ch, is_sqm=True))
                _, mem_orig_r, tele_r, mem_backup_r = best_branch
                result[f"mem_orig_{r_idx}"].append(mem_orig_r)
                result[f"tele_ancilla_{r_idx}"].append(tele_r)
                result[f"mem_backup_{r_idx}"].append(mem_backup_r)
                for q in (mem_orig_r, tele_r, mem_backup_r):
                    self.available_qubits.discard(q)
        for reg_id, qubits in result.items():
            self.allocation_map[reg_id] = qubits
        print(f"[QubitMapper] Per-Bit Allocation Complete:")
        for reg_id in sorted(result.keys()):
            print(f"  [{reg_id:20s}] qubits {sorted(result[reg_id])}")
        return result

    def allocate_swap_per_bit_topology(self, R: int, n: int) -> Dict[str, List[int]]:
        result = {}
        result["q_work"] = []
        for r_idx in range(R):
            result[f"mem_{r_idx}"] = []
        print(f"[QubitMapper] SWAP Per-Bit Allocation (Noise-Aware)")
        print(f"  Target: R={R} memory registers, n={n} qubits per register")
        total_qubits_needed = n * (1 + R)
        if total_qubits_needed > len(self.available_qubits):
            raise RuntimeError(f"Need {total_qubits_needed} qubits, have {len(self.available_qubits)}")
        for bit_idx in range(n):
            committed: Set[int] = set()
            for v in result.values():
                committed.update(v)
            # --- q_work <-> mem_0 (direct edge) ---
            candidates = self._enumerate_swap_chains(committed)
            if not candidates:
                raise RuntimeError(f"[QubitMapper] No valid SWAP pair for bit {bit_idx}")
            best = min(candidates, key=lambda ch: self._evaluate_chain_cost(ch, is_sqm=False))
            q_center, mem_0 = best
            result["q_work"].append(q_center)
            result["mem_0"].append(mem_0)
            self.available_qubits.discard(q_center)
            self.available_qubits.discard(mem_0)
            # --- Additional memory registers (r_idx >= 1): direct edge from q_center ---
            for r_idx in range(1, R):
                committed_now: Set[int] = set()
                for v in result.values():
                    committed_now.update(v)
                pool = self.available_qubits - committed_now
                mem_candidates = [
                    [q_center, nb]
                    for nb in self.graph.neighbors(q_center)
                    if nb in pool
                ]
                if not mem_candidates:
                    raise RuntimeError(f"[QubitMapper] No neighbor for mem_{r_idx}[{bit_idx}]")
                best_mem = min(mem_candidates,
                               key=lambda ch: self._evaluate_chain_cost(ch, is_sqm=False))
                result[f"mem_{r_idx}"].append(best_mem[1])
                self.available_qubits.discard(best_mem[1])
        for reg_id, qubits in result.items():
            self.allocation_map[reg_id] = qubits
        print(f"[QubitMapper] Per-Bit Allocation Complete:")
        for reg_id in sorted(result.keys()):
            print(f"  [{reg_id:20s}] qubits {sorted(result[reg_id])}")
        return result

    def _find_next_available_qubit(self, exclude: Optional[Set[int]] = None) -> Optional[int]:
        exclude_set = exclude if exclude is not None else set()
        available = self.available_qubits - exclude_set
        return min(available) if available else None

    def _find_connected_qubit_near(self, qubit: int, exclude: Optional[Set[int]] = None) -> Optional[int]:
        exclude_set = exclude if exclude is not None else set()
        for neighbor in self.graph.neighbors(qubit):
            if neighbor in self.available_qubits and neighbor not in exclude_set:
                return neighbor
        return None

    def _allocate_generic_chain(self, chain_config: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        result = {}
        total_length = sum(size for _, size in chain_config)
        chain = self._find_linear_chain_simple(total_length)
        if chain is None:
            raise RuntimeError(f"Cannot find chain of length {total_length}")
        offset = 0
        for register_id, size in chain_config:
            segment = chain[offset:offset + size]
            for q in segment:
                self.available_qubits.discard(q)
            self.allocation_map[register_id] = segment
            result[register_id] = segment
            offset += size
        return result

    def _find_linear_chain_simple(self, chain_length: int) -> Optional[List[int]]:
        if chain_length <= 0:
            return None
        if chain_length == 1 and self.available_qubits:
            return [min(self.available_qubits)]
        adj = {q: set() for q in self.available_qubits}
        for a, b in self.coupling_map:
            if a in self.available_qubits and b in self.available_qubits:
                adj[a].add(b)
                adj[b].add(a)
        def dfs_path(current: int, target_len: int, visited: Set[int], path: List[int]) -> Optional[List[int]]:
            if len(path) == target_len:
                return path.copy()
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs_path(neighbor, target_len, visited, path)
                    if result:
                        return result
                    path.pop()
                    visited.remove(neighbor)
            return None
        for start in sorted(self.available_qubits):
            result = dfs_path(start, chain_length, {start}, [start])
            if result:
                return result
        return None

    @staticmethod
    def compare_mappers(mapper1, mapper2, output_file: Optional[str] = None) -> None:
        """Compare two QubitMapper allocations with connectivity matrices and allocation tables."""
        fig, ((ax1_matrix, ax1_table), (ax2_matrix, ax2_table)) = plt.subplots(2, 2, figsize=(18, 14))
        
        color_map = {
            'q_work': '#FF6B6B',
            'mem_orig': '#4ECDC4',
            'mem_backup': '#45B7D1',
            'mem_': '#FFA07A',
            'tele_ancilla': '#FFB347',
        }
        
        def draw_mapper(mapper, ax_matrix, ax_table, title, compiler_name):
            """Draw connectivity matrix and allocation table for one mapper."""
            all_allocated = set()
            for qubits in mapper.allocation_map.values():
                all_allocated.update(qubits)
            
            sorted_qubits = sorted(all_allocated)
            n = len(sorted_qubits)
            
            if n == 0:
                ax_matrix.text(0.5, 0.5, 'No qubits allocated', ha='center', va='center')
                ax_matrix.axis('off')
                ax_table.axis('off')
                return
            
            # === LEFT: Connectivity Matrix ===
            ax_matrix.set_title(f"{title}\nConnectivity Matrix (Used Qubits)", 
                             fontsize=12, fontweight='bold')
            
            qubit_index = {q: i for i, q in enumerate(sorted_qubits)}
            matrix = [[0] * n for _ in range(n)]
            for i, q1 in enumerate(sorted_qubits):
                for j, q2 in enumerate(sorted_qubits):
                    if mapper.graph.has_edge(q1, q2):
                        matrix[i][j] = 1
            
            im = ax_matrix.imshow(matrix, cmap='YlOrRd', aspect='auto', alpha=0.7)
            
            # Add grid
            for i in range(n + 1):
                ax_matrix.axhline(i - 0.5, color='black', linewidth=0.5)
                ax_matrix.axvline(i - 0.5, color='black', linewidth=0.5)
            
            ax_matrix.set_xticks(range(n))
            ax_matrix.set_yticks(range(n))
            ax_matrix.set_xticklabels(sorted_qubits, rotation=45, fontsize=8)
            ax_matrix.set_yticklabels(sorted_qubits, fontsize=8)
            ax_matrix.set_xlabel("Physical Qubit", fontsize=10, fontweight='bold')
            ax_matrix.set_ylabel("Physical Qubit", fontsize=10, fontweight='bold')
            
            # Add text annotations
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] == 1:
                        ax_matrix.text(j, i, '1', ha='center', va='center', 
                                    color='white', fontsize=7, fontweight='bold')
            
            plt.colorbar(im, ax=ax_matrix, label='Connected')
            
            # === RIGHT: Allocation Table ===
            ax_table.axis('tight')
            ax_table.axis('off')
            
            table_data = [[f"{compiler_name} Allocation", 'Physical Qubits', 'Count']]
            for reg_name in sorted(mapper.allocation_map.keys()):
                qubits = mapper.allocation_map[reg_name]
                qubit_str = '[' + ', '.join(map(str, sorted(qubits)[:8]))
                if len(qubits) > 8:
                    qubit_str += f', ... ({len(qubits)} total)'
                qubit_str += ']'
                table_data.append([reg_name, qubit_str, str(len(qubits))])
            
            total_allocated = sum(len(q) for q in mapper.allocation_map.values())
            table_data.append(['', '', ''])
            table_data.append(['TOTAL ALLOCATED', '', str(total_allocated)])
            table_data.append(['BACKEND QUBITS', '', str(mapper.n_qubits)])
            table_data.append(['USED QUBITS', '', str(len(all_allocated))])
            
            table = ax_table.table(cellText=table_data, cellLoc='left', loc='center',
                                  colWidths=[0.35, 0.45, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.2)
            
            # Style header row
            for i in range(3):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style data rows
            for row_idx, (reg_name, _, _) in enumerate(table_data[1:-4], 1):
                color = '#FF6B6B' if 'q_work' in reg_name else \
                       '#4ECDC4' if 'mem_orig' in reg_name else \
                       '#45B7D1' if 'mem_backup' in reg_name else \
                       '#FFB347' if 'tele_ancilla' in reg_name else \
                       '#FFA07A'
                for col in range(3):
                    table[(row_idx, col)].set_facecolor(color)
                    table[(row_idx, col)].set_alpha(0.3)
            
            # Style summary rows
            summary_start = len(table_data) - 4
            for row in range(summary_start, len(table_data)):
                for col in range(3):
                    if row == summary_start:
                        table[(row, col)].set_facecolor('#F0F0F0')
                    else:
                        table[(row, col)].set_facecolor('#E8E8E8')
                        table[(row, col)].set_text_props(weight='bold')
        
        # Draw both mappers
        draw_mapper(mapper1, ax1_matrix, ax1_table, "SQM Allocation (Dual-Register)", "SQM")
        draw_mapper(mapper2, ax2_matrix, ax2_table, "SWAP Allocation (Single-Register)", "SWAP")
        
        # Add main title
        fig.suptitle('Qubit Allocation Comparison: SQM vs SWAP', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#FF6B6B', label='Operation Register (q_work)'),
            mpatches.Patch(facecolor='#4ECDC4', label='Memory Original'),
            mpatches.Patch(facecolor='#45B7D1', label='Memory Backup'),
            mpatches.Patch(facecolor='#FFB347', label='Teleportation Ancilla'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
                  bbox_to_anchor=(0.5, -0.02), frameon=True)
        
        plt.tight_layout(rect=(0, 0.02, 1, 0.96))
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"[QubitMapper] Comparison visualization saved to: {output_file}")
        
        plt.close()
