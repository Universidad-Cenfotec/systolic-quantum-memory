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
        result = {}
        logical_via = nx.Graph()
        logical_via.add_edges_from([("opreg", "mem_orig"), ("mem_orig", "tele_ancilla"), ("tele_ancilla", "mem_backup")])
        print(f"[QubitMapper] SQTM Hardware-Aware Allocation (Subgraph Isomorphism)")
        print(f"  Target: {R} registers x {n} qubits = {R*n} parallel vias")
        print(f"  Available: {len(self.available_qubits)} / {self.n_qubits}")
        if R * n * 4 > len(self.available_qubits):
            raise RuntimeError(f"[QubitMapper] Need {R*n*4} qubits, have {len(self.available_qubits)}")
        for r_idx in range(R):
            for bit_idx in range(n):
                physical_subgraph = self.graph.subgraph(list(self.available_qubits)).copy()
                if len(physical_subgraph) < 4:
                    raise RuntimeError(f"[QubitMapper] Insufficient qubits for R={r_idx}, n={bit_idx}")
                matcher = isomorphism.GraphMatcher(physical_subgraph, logical_via)
                try:
                    match = next(matcher.subgraph_isomorphisms_iter())
                except StopIteration:
                    raise RuntimeError(f"[QubitMapper] No 4-qubit path found for R={r_idx}, bit={bit_idx}")
                inv_match = {v: k for k, v in match.items()}
                mem_orig_id = f"mem_orig_{r_idx}"
                mem_backup_id = f"mem_backup_{r_idx}"
                tele_ancilla_id = f"tele_ancilla_{r_idx}"
                opreg_id = f"opreg_{r_idx}"
                for reg_id in [mem_orig_id, mem_backup_id, tele_ancilla_id, opreg_id]:
                    if reg_id not in result:
                        result[reg_id] = []
                result[mem_orig_id].append(inv_match["mem_orig"])
                result[mem_backup_id].append(inv_match["mem_backup"])
                result[tele_ancilla_id].append(inv_match["tele_ancilla"])
                result[opreg_id].append(inv_match["opreg"])
                for physical_qubit in match.keys():
                    self.available_qubits.discard(physical_qubit)
        print(f"[QubitMapper] Allocation Complete:")
        for reg_id in sorted(result.keys()):
            qubits = result[reg_id]
            self.allocation_map[reg_id] = qubits
            print(f"  [{reg_id:20s}] qubits {sorted(qubits)}")
        return result

    def allocate_chain_topology(self, chain_config: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        result = {}
        print(f"[QubitMapper] Allocating chain topology (Hardware-Aware):")
        print(f"  Config: {chain_config}")
        R = n = 0
        for reg_name, size in chain_config:
            if reg_name == "opreg":
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
        
        # Consolidate results: opereg gets exactly n qubits (first allocation)
        # Others keep their per-register allocation
        for reg_name, _ in chain_config:
            if reg_name == "opreg":
                # Take ONLY the first n qubits from opreg allocations 
                # (from the first register only, since opreg is shared)
                key_0 = "opreg_0"
                if key_0 in sqtm_result:
                    result[reg_name] = sqtm_result[key_0][:n]
            elif reg_name in sqtm_result:
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
        Visualize the backend topology and qubit allocation mapping.
        
        Creates a graph visualization showing:
        - Physical qubit topology (nodes and edges)
        - Color-coded register allocations
        - Physical qubit indices and register assignments
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ===== LEFT PLOT: Backend Topology =====
        ax1.set_title(f"Backend Topology: {self.backend.name}\n({self.n_qubits} qubits)", 
                      fontsize=14, fontweight='bold')
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        # Draw all edges first
        nx.draw_networkx_edges(self.graph, pos, ax=ax1, width=1.5, 
                              edge_color='lightgray', alpha=0.6)
        
        # Draw nodes with color coding by allocation
        node_colors = []
        color_map = {
            'opreg': '#FF6B6B',           # Red
            'mem_orig': '#4ECDC4',        # Teal
            'mem_backup': '#45B7D1',      # Blue
            'tele_ancilla': '#FFA07A',    # Light salmon
            'unallocated': '#D3D3D3'      # Light gray
        }
        
        for node in self.graph.nodes():
            assigned = False
            for reg_name, qubits in self.allocation_map.items():
                if node in qubits:
                    # Determine color based on register type
                    if 'opreg' in reg_name:
                        node_colors.append(color_map['opreg'])
                    elif 'mem_orig' in reg_name:
                        node_colors.append(color_map['mem_orig'])
                    elif 'mem_backup' in reg_name:
                        node_colors.append(color_map['mem_backup'])
                    elif 'tele_ancilla' in reg_name:
                        node_colors.append(color_map['tele_ancilla'])
                    assigned = True
                    break
            if not assigned:
                node_colors.append(color_map['unallocated'])
        
        nx.draw_networkx_nodes(self.graph, pos, ax=ax1, node_color=node_colors,
                              node_size=500, alpha=0.85)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, ax=ax1, font_size=8, font_weight='bold')
        
        ax1.axis('off')
        
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
        table_data.append(['TOTAL', '', str(total_allocated)])
        table_data.append(['REMAINING', '', str(len(self.available_qubits))])
        
        # Create table
        table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.25, 0.55, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style summary rows
        for row in [len(table_data)-3, len(table_data)-2, len(table_data)-1]:
            for col in range(3):
                if row == len(table_data)-3:
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
            mpatches.Patch(facecolor=color_map['unallocated'], label='Unallocated')
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
        Compare qubit allocations between SQTM and SWAP mappers side-by-side.
        
        Creates a visualization showing both allocations for easy comparison.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass
        
        fig, ((ax1_top, ax2_top), (ax1_bot, ax2_bot)) = plt.subplots(2, 2, figsize=(18, 14))
        
        color_map = {
            'opreg': '#FF6B6B',           # Red
            'mem_orig': '#4ECDC4',        # Teal
            'mem_backup': '#45B7D1',      # Blue
            'tele_ancilla': '#FFA07A',    # Light salmon
            'unallocated': '#D3D3D3'      # Light gray
        }
        
        # Helper function to draw mapper
        def draw_mapper(mapper, ax_graph, ax_table, title, compiler_name):
            ax_graph.set_title(f"{title}\nTopology ({mapper.n_qubits} qubits)", 
                             fontsize=12, fontweight='bold')
            
            pos = nx.spring_layout(mapper.graph, k=0.5, iterations=50, seed=42)
            nx.draw_networkx_edges(mapper.graph, pos, ax=ax_graph, width=1.5, 
                                  edge_color='lightgray', alpha=0.6)
            
            # Build color map for all nodes
            node_colors = []
            node_list = sorted(mapper.graph.nodes())  # Ensure consistent ordering
            
            for node in node_list:
                assigned = False
                for reg_name, qubits in mapper.allocation_map.items():
                    if node in qubits:
                        if 'opreg' in reg_name:
                            node_colors.append(color_map['opreg'])
                        elif 'mem_orig' in reg_name:
                            node_colors.append(color_map['mem_orig'])
                        elif 'mem_backup' in reg_name:
                            node_colors.append(color_map['mem_backup'])
                        elif 'tele_ancilla' in reg_name:
                            node_colors.append(color_map['tele_ancilla'])
                        else:
                            node_colors.append(color_map['unallocated'])
                        assigned = True
                        break
                if not assigned:
                    node_colors.append(color_map['unallocated'])
            
            # Ensure we have exactly one color per node
            if len(node_colors) != len(node_list):
                node_colors = [color_map['unallocated']] * len(node_list)
            
            nx.draw_networkx_nodes(mapper.graph, pos, ax=ax_graph, node_color=node_colors,
                                  node_size=400, alpha=0.85)
            nx.draw_networkx_labels(mapper.graph, pos, ax=ax_graph, font_size=7, font_weight='bold')
            
            ax_graph.axis('off')
            
            # Draw allocation table
            ax_table.axis('tight')
            ax_table.axis('off')
            
            table_data = [[f"{compiler_name} Allocation", 'Physical Qubits', 'Count']]
            for reg_name in sorted(mapper.allocation_map.keys()):
                qubits = mapper.allocation_map[reg_name]
                qubit_str = '[' + ', '.join(map(str, sorted(qubits)[:5]))
                if len(qubits) > 5:
                    qubit_str += f', ... ({len(qubits)} total)'
                qubit_str += ']'
                table_data.append([reg_name, qubit_str, str(len(qubits))])
            
            total_allocated = sum(len(q) for q in mapper.allocation_map.values())
            table_data.append(['', '', ''])
            table_data.append(['TOTAL ALLOCATED', '', str(total_allocated)])
            table_data.append(['REMAINING', '', str(len(mapper.available_qubits))])
            
            table = ax_table.table(cellText=table_data, cellLoc='left', loc='center',
                                  colWidths=[0.35, 0.45, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            for i in range(3):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for row in [len(table_data)-3, len(table_data)-2, len(table_data)-1]:
                for col in range(3):
                    if row == len(table_data)-3:
                        table[(row, col)].set_facecolor('#F0F0F0')
                    else:
                        table[(row, col)].set_facecolor('#E8E8E8')
                        table[(row, col)].set_text_props(weight='bold')
        
        # Draw both mappers
        draw_mapper(sqtm_mapper, ax1_top, ax1_bot, "SQTM Allocation\n(Dual-Register)", "SQTM")
        draw_mapper(swap_mapper, ax2_top, ax2_bot, "SWAP Allocation\n(Single-Register)", "SWAP")
        
        # Add main title
        fig.suptitle('Qubit Allocation Comparison: SQTM vs SWAP', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend at the bottom
        legend_elements = [
            mpatches.Patch(facecolor=color_map['opreg'], label='Operation Register'),
            mpatches.Patch(facecolor=color_map['mem_orig'], label='Memory Original'),
            mpatches.Patch(facecolor=color_map['mem_backup'], label='Memory Backup'),
            mpatches.Patch(facecolor=color_map['tele_ancilla'], label='Teleportation Ancilla'),
            mpatches.Patch(facecolor=color_map['unallocated'], label='Unallocated')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10, 
                  bbox_to_anchor=(0.5, -0.02), frameon=True)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[QubitMapper] Comparison visualization saved to: {output_file}")
        
        plt.close()

