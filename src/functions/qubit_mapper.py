# ============================================================
# SQTM Research Project - Qubit Mapper (Hardware-Aware)
# Authors: Danny Valerio-Ramirez & Santiago Nunez-Corrales
# ============================================================

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx
from networkx.algorithms import isomorphism
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
        if preferred_location:
            if (len(preferred_location) == size and all(q in self.available_qubits for q in preferred_location)):
                subgraph = self.graph.subgraph(preferred_location)
                allocated = (preferred_location if nx.is_connected(subgraph) else self.find_connected_subgraph(size))
            else:
                allocated = self.find_connected_subgraph(size)
        else:
            allocated = self.find_connected_subgraph(size)
        if allocated is None:
            raise RuntimeError(f"Cannot allocate {size} qubits for {register_id}.")
        for qubit in allocated:
            self.available_qubits.remove(qubit)
        self.allocation_map[register_id] = allocated
        return allocated

    def allocate_sqtm_topology(self, R: int, n: int) -> Dict[str, List[int]]:
        """
        Allocate SQTM topology with shared operation register.
        
        Structure:
        - q_work: 1 shared register with n qubits
        - For each logical address i (0 to R-1):
          - mem_orig_i: n qubits
          - mem_backup_i: n qubits
          - tele_ancilla_i: n qubits
        
        Total: n + R*3*n = n*(1 + 3*R) qubits
        """
        result = {}
        
        print(f"[QubitMapper] SQTM Hardware-Aware Allocation (Linear Chain)")
        print(f"  Target: R={R} memory pairs, n={n} qubits per register")
        print(f"  Structure: q_work(n) + R x [mem_orig(n) + mem_backup(n) + tele_ancilla(n)]")
        print(f"  Total qubits needed: {n * (1 + 3 * R)}")
        print(f"  Available: {len(self.available_qubits)} / {self.n_qubits}")
        
        # Calculate total qubits needed: 1 q_work (shared) + 3*R memory registers
        total_qubits_needed = n * (1 + 3 * R)
        if total_qubits_needed > len(self.available_qubits):
            raise RuntimeError(f"[QubitMapper] Need {total_qubits_needed} qubits, have {len(self.available_qubits)}")
        
        # Find a linear chain of qubits
        chain = self._find_linear_chain_simple(total_qubits_needed)
        if chain is None:
            raise RuntimeError(f"[QubitMapper] Cannot find chain of length {total_qubits_needed}")
        
        offset = 0
        
        # 1. Allocate shared q_work (first n qubits of chain)
        result["q_work"] = chain[offset:offset + n]
        offset += n
        
        # 2. Allocate memory registers for each logical address
        for r_idx in range(R):
            # mem_orig_r_idx
            result[f"mem_orig_{r_idx}"] = chain[offset:offset + n]
            offset += n
            
            # mem_backup_r_idx
            result[f"mem_backup_{r_idx}"] = chain[offset:offset + n]
            offset += n
            
            # tele_ancilla_r_idx
            result[f"tele_ancilla_{r_idx}"] = chain[offset:offset + n]
            offset += n
        
        # Remove allocated qubits from available pool
        for qubits in result.values():
            for q in qubits:
                self.available_qubits.discard(q)
        
        # Update allocation_map for tracking
        for reg_id, qubits in result.items():
            self.allocation_map[reg_id] = qubits
        
        print(f"[QubitMapper] Allocation Complete:")
        for reg_id in sorted(result.keys()):
            qubits = result[reg_id]
            print(f"  [{reg_id:20s}] qubits {sorted(qubits)}")
        
        return result

    def allocate_chain_topology(self, chain_config: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        result = {}
        print(f"[QubitMapper] Allocating chain topology (Hardware-Aware):")
        print(f"  Config: {chain_config}")
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
        if R == 0 or n == 0:
            return self._allocate_generic_chain(chain_config)
        print(f"  Detected SQTM: R={R}, n={n}")
        sqtm_result = self.allocate_sqtm_topology(R=R, n=n)
        
        # Copy results directly (q_work is now a single shared register)
        for reg_name, _ in chain_config:
            if reg_name in sqtm_result:
                result[reg_name] = sqtm_result[reg_name]
        
        return result

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

    def allocate_multi_chain_topology(self, num_chains: int, chain_template: List[Tuple[str, int]], base_chain_length: Optional[int] = None) -> Dict[str, List[int]]:
        result = {}
        for chain_idx in range(num_chains):
            indexed_config = [(f"{base_id}_{chain_idx}", size) for base_id, size in chain_template]
            chain_length = sum(size for _, size in indexed_config)
            chain = self._find_linear_chain_simple(chain_length)
            if chain is None:
                raise RuntimeError(f"Cannot find chain {chain_idx}")
            offset = 0
            for reg_id, size in indexed_config:
                segment = chain[offset:offset + size]
                for q in segment:
                    self.available_qubits.discard(q)
                self.allocation_map[reg_id] = segment
                result[reg_id] = segment
                offset += size
        return result

    def visualize_mapping(self, output_file: str = "results/qubit_mapping.png") -> None:
        """
        Visualize the qubit allocation mapping with connectivity matrix.
        
        Creates a visualization showing:
        - Connectivity matrix of used qubits (1 = connected, 0 = not connected)
        - Color-coded register allocations
        - Physical qubit allocation summary table
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass
        
        # Get all used qubits
        used_qubits = set()
        for qubits in self.allocation_map.values():
            used_qubits.update(qubits)
        
        used_qubits = sorted(list(used_qubits))
        
        if not used_qubits:
            print("[QubitMapper] No qubits allocated. Skipping visualization.")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # ===== LEFT PLOT: Connectivity Matrix =====
        ax1.set_title("Qubit Connectivity Matrix (Used Qubits Only)", 
                      fontsize=14, fontweight='bold')
        
        # Create connectivity matrix for used qubits only
        qubit_to_idx = {q: i for i, q in enumerate(used_qubits)}
        n_used = len(used_qubits)
        connectivity_matrix = [[0] * n_used for _ in range(n_used)]
        
        # Populate connectivity matrix
        for a, b in self.coupling_map:
            if a in qubit_to_idx and b in qubit_to_idx:
                i, j = qubit_to_idx[a], qubit_to_idx[b]
                connectivity_matrix[i][j] = 1
                connectivity_matrix[j][i] = 1
        
        # Create color map for allocations
        allocation_colors = {}
        color_map = {
            'q_work': '#FF6B6B',           # Red
            'mem_orig': '#4ECDC4',        # Teal
            'mem_backup': '#45B7D1',      # Blue
            'tele_ancilla': '#FFA07A',    # Light salmon
        }
        
        for reg_name, qubits in self.allocation_map.items():
            color = color_map.get('q_work' if 'q_work' in reg_name else
                                 'mem_orig' if 'mem_orig' in reg_name else
                                 'mem_backup' if 'mem_backup' in reg_name else
                                 'tele_ancilla', '#D3D3D3')
            for q in qubits:
                allocation_colors[q] = color
        
        # Draw the matrix as an image with grid
        im = ax1.imshow(connectivity_matrix, cmap='YlOrRd', aspect='auto', alpha=0.7)
        
        # Add grid lines
        for i in range(n_used + 1):
            ax1.axhline(i - 0.5, color='black', linewidth=0.5)
            ax1.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # Add qubit labels on axes
        ax1.set_xticks(range(n_used))
        ax1.set_yticks(range(n_used))
        ax1.set_xticklabels(used_qubits, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(used_qubits, fontsize=9)
        ax1.set_xlabel("Physical Qubit", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Physical Qubit", fontsize=11, fontweight='bold')
        
        # Add text annotations for connections
        for i in range(n_used):
            for j in range(n_used):
                if connectivity_matrix[i][j] == 1:
                    text_color = 'white' if connectivity_matrix[i][j] > 0.5 else 'black'
                    ax1.text(j, i, '1', ha='center', va='center', 
                            color=text_color, fontsize=8, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Connected (1 = Yes, 0 = No)', fontsize=10)
        
        # ===== RIGHT PLOT: Register Allocation Table =====
        ax2.axis('tight')
        ax2.axis('off')
        
        # Create allocation table
        table_data = [['Register', 'Physical Qubits', 'Count']]
        
        for reg_name in sorted(self.allocation_map.keys()):
            qubits = self.allocation_map[reg_name]
            qubit_str = '[' + ', '.join(map(str, sorted(qubits))) + ']'
            table_data.append([reg_name, qubit_str, str(len(qubits))])
        
        # Add summary
        total_allocated = sum(len(q) for q in self.allocation_map.values())
        table_data.append(['', '', ''])
        table_data.append(['TOTAL ALLOCATED', '', str(total_allocated)])
        table_data.append(['BACKEND QUBITS', '', str(self.n_qubits)])
        table_data.append(['USED QUBITS', '', str(len(used_qubits))])
        
        # Create table
        table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.5, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with register colors
        for row_idx, (reg_name, _, _) in enumerate(table_data[1:-4], 1):
            color = color_map.get('opreg' if 'opreg' in reg_name else
                                 'mem_orig' if 'mem_orig' in reg_name else
                                 'mem_backup' if 'mem_backup' in reg_name else
                                 'tele_ancilla', '#F0F0F0')
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
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=color_map['opreg'], label='Operation Register'),
            mpatches.Patch(facecolor=color_map['mem_orig'], label='Memory Original'),
            mpatches.Patch(facecolor=color_map['mem_backup'], label='Memory Backup'),
            mpatches.Patch(facecolor=color_map['tele_ancilla'], label='Teleportation Ancilla'),
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
        ax2.set_title("Qubit Allocation Summary", fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[QubitMapper] Visualization saved to: {output_file}")
        
        plt.close()

    @staticmethod
    def compare_mappers(sqtm_mapper: 'QubitMapper', swap_mapper: 'QubitMapper', 
                       output_file: str = "results/qubit_mapping_comparison.png") -> None:
        """
        Compare qubit allocations between SQTM and SWAP mappers using connectivity matrices.
        
        Creates a visualization showing both allocations with connectivity matrices
        for easy comparison.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass
        
        fig, ((ax1_matrix, ax1_table), (ax2_matrix, ax2_table)) = plt.subplots(2, 2, figsize=(18, 14))
        
        color_map = {
            'q_work': '#FF6B6B',           # Red
            'mem_orig': '#4ECDC4',        # Teal
            'mem_backup': '#45B7D1',      # Blue
            'tele_ancilla': '#FFA07A',    # Light salmon
        }
        
        # Helper function to draw mapper
        def draw_mapper(mapper, ax_matrix, ax_table, title, compiler_name):
            # Get used qubits for this mapper
            used_qubits = set()
            for qubits in mapper.allocation_map.values():
                used_qubits.update(qubits)
            
            used_qubits = sorted(list(used_qubits))
            
            if not used_qubits:
                ax_matrix.text(0.5, 0.5, 'No qubits allocated', 
                              ha='center', va='center', fontsize=12)
                ax_matrix.axis('off')
                ax_table.axis('off')
                return
            
            # ===== LEFT: Connectivity Matrix =====
            ax_matrix.set_title(f"{title}\nConnectivity Matrix (Used Qubits)", 
                             fontsize=12, fontweight='bold')
            
            # Create connectivity matrix for used qubits only
            qubit_to_idx = {q: i for i, q in enumerate(used_qubits)}
            n_used = len(used_qubits)
            connectivity_matrix = [[0] * n_used for _ in range(n_used)]
            
            # Populate connectivity matrix
            for a, b in mapper.coupling_map:
                if a in qubit_to_idx and b in qubit_to_idx:
                    i, j = qubit_to_idx[a], qubit_to_idx[b]
                    connectivity_matrix[i][j] = 1
                    connectivity_matrix[j][i] = 1
            
            # Draw the matrix as an image with grid
            im = ax_matrix.imshow(connectivity_matrix, cmap='YlOrRd', aspect='auto', alpha=0.7)
            
            # Add grid lines
            for i in range(n_used + 1):
                ax_matrix.axhline(i - 0.5, color='black', linewidth=0.5)
                ax_matrix.axvline(i - 0.5, color='black', linewidth=0.5)
            
            # Add qubit labels on axes
            ax_matrix.set_xticks(range(n_used))
            ax_matrix.set_yticks(range(n_used))
            ax_matrix.set_xticklabels(used_qubits, rotation=45, ha='right', fontsize=8)
            ax_matrix.set_yticklabels(used_qubits, fontsize=8)
            ax_matrix.set_xlabel("Physical Qubit", fontsize=10, fontweight='bold')
            ax_matrix.set_ylabel("Physical Qubit", fontsize=10, fontweight='bold')
            
            # Add text annotations for connections
            for i in range(n_used):
                for j in range(n_used):
                    if connectivity_matrix[i][j] == 1:
                        text_color = 'white' if connectivity_matrix[i][j] > 0.5 else 'black'
                        ax_matrix.text(j, i, '1', ha='center', va='center', 
                                    color=text_color, fontsize=7, fontweight='bold')
            
            # ===== RIGHT: Allocation Table =====
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
            table_data.append(['USED QUBITS', '', str(len(used_qubits))])
            
            table = ax_table.table(cellText=table_data, cellLoc='left', loc='center',
                                  colWidths=[0.35, 0.45, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.2)
            
            # Style header row
            for i in range(3):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style data rows with register colors
            for row_idx, (reg_name, _, _) in enumerate(table_data[1:-4], 1):
                color = color_map.get('q_work' if 'q_work' in reg_name else
                                     'mem_orig' if 'mem_orig' in reg_name else
                                     'mem_backup' if 'mem_backup' in reg_name else
                                     'tele_ancilla', '#F0F0F0')
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
        draw_mapper(sqtm_mapper, ax1_matrix, ax1_table, "SQTM Allocation (Dual-Register)", "SQTM")
        draw_mapper(swap_mapper, ax2_matrix, ax2_table, "SWAP Allocation (Single-Register)", "SWAP")
        
        # Add main title
        fig.suptitle('Qubit Allocation Comparison: SQTM vs SWAP', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend at the bottom
        legend_elements = [
            mpatches.Patch(facecolor=color_map['q_work'], label='Operation Register'),
            mpatches.Patch(facecolor=color_map['mem_orig'], label='Memory Original'),
            mpatches.Patch(facecolor=color_map['mem_backup'], label='Memory Backup'),
            mpatches.Patch(facecolor=color_map['tele_ancilla'], label='Teleportation Ancilla'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, 
                  bbox_to_anchor=(0.5, -0.02), frameon=True)
        
        plt.tight_layout(rect=(0, 0.02, 1, 0.96))
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[QubitMapper] Comparison visualization saved to: {output_file}")
        
        plt.close()

