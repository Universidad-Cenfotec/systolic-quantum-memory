# ============================================================
# SQTM Research Project - Qubit Mapper (Hardware-Aware)
# Authors: Danny Valerio-Ramirez & Santiago Nunez-Corrales
# ============================================================

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

class QubitMapper:
    def __init__(self, backend):
        self.backend = backend
        self.coupling_map = backend.configuration().coupling_map
        self.n_qubits = backend.configuration().n_qubits
        self.graph = self._build_connectivity_graph()
        self.available_qubits: Set[int] = set(range(self.n_qubits))
        self.allocation_map: Dict[str, List[int]] = {}

    def _build_connectivity_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_qubits))
        for edge in self.coupling_map:
            graph.add_edge(edge[0], edge[1])
        return graph

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

    def find_contiguous_block(self, size: int) -> Optional[List[int]]:
        available_subgraph = self.graph.subgraph(self.available_qubits).copy()
        degrees = dict(available_subgraph.degree())
        if not degrees:
            return None
        start = max(degrees, key=lambda x: degrees[x])
        return self.find_connected_subgraph(size, preferred_start=start)

    def allocate_register(self, register_type: str, register_id: str, size: int, preferred_location: Optional[List[int]] = None) -> List[int]:
        """Allocate a register by finding a connected subgraph of desired size."""
        if preferred_location:
            if len(preferred_location) == size and all(q in self.available_qubits for q in preferred_location):
                subgraph = self.graph.subgraph(preferred_location)
                try:
                    allocated = preferred_location if nx.is_connected(subgraph) else self.find_connected_subgraph(size)
                except:
                    allocated = self.find_connected_subgraph(size)
            else:
                allocated = self.find_connected_subgraph(size)
        else:
            allocated = self.find_connected_subgraph(size)
        if allocated is None:
            raise RuntimeError(f"Cannot allocate {size} qubits for {register_id}.")
        for qubit in allocated:
            self.available_qubits.discard(qubit)
        self.allocation_map[register_id] = allocated
        return allocated

    def allocate_chain_topology(self, chain_config: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        is_sqtm = any('tele_ancilla' in reg_name for reg_name, _ in chain_config)
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
        if is_sqtm:
            return self.allocate_sqtm_per_bit_topology(R=R, n=n)
        else:
            return self.allocate_swap_per_bit_topology(R=R, n=n)

    def allocate_sqtm_per_bit_topology(self, R: int, n: int) -> Dict[str, List[int]]:
        result = {}
        result["q_work"] = []
        for r_idx in range(R):
            result[f"mem_orig_{r_idx}"] = []
            result[f"mem_backup_{r_idx}"] = []
            result[f"tele_ancilla_{r_idx}"] = []
        print(f"[QubitMapper] SQTM Per-Bit Allocation (Qubit-Centric Center)")
        print(f"  Target: R={R} memory pairs, n={n} qubits per register")
        total_qubits_needed = n * (1 + 3 * R)
        if total_qubits_needed > len(self.available_qubits):
            raise RuntimeError(f"[QubitMapper] Need {total_qubits_needed} qubits, have {len(self.available_qubits)}")
        for bit_idx in range(n):
            q_center = self._find_next_available_qubit()
            if q_center is None:
                raise RuntimeError(f"[QubitMapper] Cannot allocate q_work[{bit_idx}]")
            result["q_work"].append(q_center)
            self.available_qubits.discard(q_center)
            exclude_set = set(result["q_work"])
            mem_orig_0 = self._find_connected_qubit_near(q_center, exclude_set) or self._find_next_available_qubit(exclude_set)
            if mem_orig_0 is None:
                raise RuntimeError(f"Cannot allocate mem_orig_0[{bit_idx}]")
            result["mem_orig_0"].append(mem_orig_0)
            self.available_qubits.discard(mem_orig_0)
            exclude_set.add(mem_orig_0)
            tele_0 = self._find_connected_qubit_near(mem_orig_0, exclude_set) or self._find_next_available_qubit(exclude_set)
            if tele_0 is None:
                raise RuntimeError(f"Cannot allocate tele_ancilla_0[{bit_idx}]")
            result["tele_ancilla_0"].append(tele_0)
            self.available_qubits.discard(tele_0)
            exclude_set.add(tele_0)
            mem_backup_0 = self._find_connected_qubit_near(tele_0, exclude_set) or self._find_next_available_qubit(exclude_set)
            if mem_backup_0 is None:
                raise RuntimeError(f"Cannot allocate mem_backup_0[{bit_idx}]")
            result["mem_backup_0"].append(mem_backup_0)
            self.available_qubits.discard(mem_backup_0)
            exclude_set.add(mem_backup_0)
            for r_idx in range(1, R):
                mem_orig_r = self._find_connected_qubit_near(q_center, exclude_set) or self._find_next_available_qubit(exclude_set)
                if mem_orig_r is None:
                    raise RuntimeError(f"Cannot allocate mem_orig_{r_idx}[{bit_idx}]")
                result[f"mem_orig_{r_idx}"].append(mem_orig_r)
                self.available_qubits.discard(mem_orig_r)
                exclude_set.add(mem_orig_r)
                tele_r = self._find_connected_qubit_near(mem_orig_r, exclude_set) or self._find_next_available_qubit(exclude_set)
                if tele_r is None:
                    raise RuntimeError(f"Cannot allocate tele_ancilla_{r_idx}[{bit_idx}]")
                result[f"tele_ancilla_{r_idx}"].append(tele_r)
                self.available_qubits.discard(tele_r)
                exclude_set.add(tele_r)
                mem_backup_r = self._find_connected_qubit_near(tele_r, exclude_set) or self._find_next_available_qubit(exclude_set)
                if mem_backup_r is None:
                    raise RuntimeError(f"Cannot allocate mem_backup_{r_idx}[{bit_idx}]")
                result[f"mem_backup_{r_idx}"].append(mem_backup_r)
                self.available_qubits.discard(mem_backup_r)
                exclude_set.add(mem_backup_r)
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
        print(f"[QubitMapper] SWAP Per-Bit Allocation (Qubit-Centric Center)")
        print(f"  Target: R={R} memory registers, n={n} qubits per register")
        total_qubits_needed = n * (1 + R)
        if total_qubits_needed > len(self.available_qubits):
            raise RuntimeError(f"Need {total_qubits_needed} qubits, have {len(self.available_qubits)}")
        for bit_idx in range(n):
            q_center = self._find_next_available_qubit()
            if q_center is None:
                raise RuntimeError(f"Cannot allocate q_work[{bit_idx}]")
            result["q_work"].append(q_center)
            self.available_qubits.discard(q_center)
            exclude_set = set(result["q_work"])
            mem_0 = self._find_connected_qubit_near(q_center, exclude_set) or self._find_next_available_qubit(exclude_set)
            if mem_0 is None:
                raise RuntimeError(f"Cannot allocate mem_0[{bit_idx}]")
            result["mem_0"].append(mem_0)
            self.available_qubits.discard(mem_0)
            exclude_set.add(mem_0)
            for r_idx in range(1, R):
                mem_r = self._find_connected_qubit_near(q_center, exclude_set) or self._find_next_available_qubit(exclude_set)
                if mem_r is None:
                    raise RuntimeError(f"Cannot allocate mem_{r_idx}[{bit_idx}]")
                result[f"mem_{r_idx}"].append(mem_r)
                self.available_qubits.discard(mem_r)
                exclude_set.add(mem_r)
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
        draw_mapper(mapper1, ax1_matrix, ax1_table, "SQTM Allocation (Dual-Register)", "SQTM")
        draw_mapper(mapper2, ax2_matrix, ax2_table, "SWAP Allocation (Single-Register)", "SWAP")
        
        # Add main title
        fig.suptitle('Qubit Allocation Comparison: SQTM vs SWAP', 
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
