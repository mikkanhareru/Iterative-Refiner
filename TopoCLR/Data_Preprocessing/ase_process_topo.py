import numpy as np
import torch
from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList
from collections import defaultdict, deque

# Standard valence for common elements
STANDARD_VALENCE = {
    'H': 1, 'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
}

# Alternative valences (sp3, sp2, sp, other oxidation states)
ALTERNATIVE_VALENCE = {
    'C': [4, 3, 2],
    'N': [3, 2, 1, 4],
    'O': [2, 1],
    'S': [2, 4, 6],
    'P': [3, 5],
}

def gaussian_rbf(distances, centers, width):
    distances = np.array(distances).reshape(-1, 1)
    centers = np.array(centers).reshape(1, -1)
    return np.exp(-((distances - centers) ** 2) / (2 * width ** 2))

class MolecularGraphBuilder:
    """
    Build molecular graph from ASE Atoms object with structural features.
    
    Optimized for structure prediction tasks with focus on:
    - Geometric features (distance, RBF, unit vectors)
    - Topological features (degree, ring membership)
    - Chemical validation
    """
    
    def __init__(self, atoms, multiplier=1.0, rbf_params=None):
        """
        Initialize with ASE Atoms object.
        
        Parameters:
        -----------
        atoms : ase.Atoms
            Molecular structure from ase.read() or created manually
        multiplier : float
            Multiplier for natural cutoffs (default 1.0)
            Increase if missing bonds, decrease if too many
        rbf_params : dict, optional
            RBF parameters: {'start': 0.0, 'stop': 6.0, 'num_centers': 50, 'width': 0.3}
        """
        self.atoms = atoms
        self.multiplier = multiplier
        self.bonds = []
        self.bond_lengths = {}
        self.connectivity = defaultdict(list)
        
        # RBF parameters
        if rbf_params is None:
            rbf_params = {
                'start': 0.0,
                'stop': 6.0,
                'num_centers': 50,
                'width': 0.3
            }
        self.rbf_params = rbf_params
        self.rbf_centers = np.linspace(
            rbf_params['start'], 
            rbf_params['stop'], 
            rbf_params['num_centers']
        )
        
        # Ring detection cache
        self._rings = None
        self._atom_in_ring = None
        
    def build_connectivity(self):
        """
        Build connectivity using ASE natural cutoffs.
        Returns list of bonds as (atom_i, atom_j) tuples.
        """
        cutoffs = natural_cutoffs(self.atoms, mult=self.multiplier)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(self.atoms)
        
        bonds_set = set()
        for i in range(len(self.atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                bond = tuple(sorted([i, j]))
                if bond not in bonds_set:
                    bonds_set.add(bond)
                    dist = self.atoms.get_distance(i, j, mic=True)
                    self.bonds.append(bond)
                    self.bond_lengths[bond] = dist
        
        self._update_connectivity()
        #print(f"Built connectivity: {len(self.bonds)} bonds")
        return self.bonds
    
    def _update_connectivity(self):
        """Update connectivity dictionary."""
        self.connectivity = defaultdict(list)
        for i, j in self.bonds:
            self.connectivity[i].append(j)
            self.connectivity[j].append(i)

    def _canonicalize_cycle(self, cycle):
        """
        Return a canonical tuple for a simple cycle (no repeated start/end).
        We pick the lexicographically smallest rotation among both directions.
        """
        c = cycle[:]
        if len(c) >= 2 and c[0] == c[-1]:
            c = c[:-1]
        n = len(c)
        # rotate so the smallest index is first; check both directions
        m = min(c)
        starts = [i for i, x in enumerate(c) if x == m]
        candidates = []
        for i in starts:
            rot = c[i:] + c[:i]
            candidates.append(tuple(rot))
            candidates.append(tuple(reversed(rot)))
        return min(candidates)

    def detect_rings(self, max_ring_size=8):
        """
        Detect simple rings using a Paton-style cycle-basis search on an
        undirected graph. Returns a list of node-index lists.
        """
        if getattr(self, "_rings", None) is not None:
            return self._rings

        n = len(self.atoms)
        adj = self.connectivity  # list[list[int]], undirected neighbors

        visited = [False] * n
        parent = [-1] * n
        rings_set = set()

        for root in range(n):
            if visited[root]:
                continue
            # DFS stack
            parent[root] = root
            stack = [root]
            visited[root] = True

            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if w == parent[v]:
                        continue
                    if not visited[w]:
                        visited[w] = True
                        parent[w] = v
                        stack.append(w)
                    else:
                        # Back/cross edge found: form cycle along parent chains
                        # Path v -> LCA and w -> LCA
                        u = v
                        path_u = [u]
                        seen_u = {u}
                        while parent[u] != u:
                            u = parent[u]
                            path_u.append(u)
                            seen_u.add(u)

                        x = w
                        path_w = [x]
                        while x not in seen_u:
                            x = parent[x]
                            path_w.append(x)

                        lca = x
                        # cycle: v->...->lca plus reversed (w->...->lca), without double LCA
                        cyc = path_u[: path_u.index(lca) + 1] + list(reversed(path_w[:-1]))

                        # de-dup any accidental repeats while preserving order
                        seen = set()
                        cyc_unique = []
                        for a in cyc:
                            if a not in seen:
                                cyc_unique.append(a)
                                seen.add(a)

                        if 3 <= len(cyc_unique) <= max_ring_size:
                            key = self._canonicalize_cycle(cyc_unique)
                            rings_set.add(key)

        # store as lists (or sets if you prefer)
        self._rings = [list(r) for r in rings_set]
        return self._rings
    
    def get_atom_in_ring(self, max_ring_size=8):
        """
        Get boolean array indicating which atoms are in rings.
        
        Parameters:
        -----------
        max_ring_size : int
            Maximum ring size to detect (default 8)
        
        Returns:
        --------
        ndarray : Boolean array [n_atoms] where True = atom is in a ring
        """
        if self._atom_in_ring is not None:
            return self._atom_in_ring
        
        rings = self.detect_rings(max_ring_size)
        in_ring = np.zeros(len(self.atoms), dtype=bool)
        
        for ring in rings:
            for atom_idx in ring:
                in_ring[atom_idx] = True
        
        self._atom_in_ring = in_ring
        return in_ring
    
    def compute_rbf(self, distances=None):
        """
        Compute RBF encoding for bond distances.
        
        Parameters:
        -----------
        distances : array-like, optional
            If None, uses all bond distances
        
        Returns:
        --------
        ndarray : RBF features [n_distances, n_centers]
        """
        if distances is None:
            distances = list(self.bond_lengths.values())
        
        return gaussian_rbf(
            distances, 
            self.rbf_centers, 
            self.rbf_params['width']
        )
    
    def get_degrees(self, weighted=False):
        """
        Get node degrees (number of neighbors).
        
        Parameters:
        -----------
        weighted : bool
            If True, weight by inverse distance
        
        Returns:
        --------
        ndarray : Degrees [n_atoms]
        """
        degrees = np.zeros(len(self.atoms))
        
        if not weighted:
            # Simple degree count
            for i in range(len(self.atoms)):
                degrees[i] = len(self.connectivity[i])
        else:
            # Weighted by inverse distance
            for i in range(len(self.atoms)):
                for j in self.connectivity[i]:
                    bond = tuple(sorted([i, j]))
                    dist = self.bond_lengths[bond]
                    degrees[i] += 1.0 / dist
        
        return degrees
    
    def get_unit_vectors(self):
        """
        Get unit vectors for all bonds.
        
        Returns:
        --------
        dict : {(i, j): unit_vector} for each bond
        """
        unit_vectors = {}
        
        for i, j in self.bonds:
            pos_i = self.atoms[i].position
            pos_j = self.atoms[j].position
            vec = pos_j - pos_i
            norm = np.linalg.norm(vec)
            unit_vec = vec / norm if norm > 0 else vec
            unit_vectors[(i, j)] = unit_vec
        
        return unit_vectors
    
    def is_aromatic_carbon(self, atom_idx):
        """Detect aromatic carbon (3 bonds, 2+ carbon neighbors)."""
        if self.atoms[atom_idx].symbol != 'C':
            return False
        if len(self.connectivity[atom_idx]) != 3:
            return False
        
        carbon_neighbors = sum(1 for n in self.connectivity[atom_idx] 
                              if self.atoms[n].symbol == 'C')
        return carbon_neighbors >= 2
    
    def validate_atom(self, atom_idx, use_aromatic_check=True):
        """
        Check if atom satisfies valence requirements.
        
        Parameters:
        -----------
        atom_idx : int
            Atom index
        use_aromatic_check : bool
            Allow aromatic carbons to have valence 3 (default True)
        
        Returns:
        --------
        bool : True if valid
        """
        symbol = self.atoms[atom_idx].symbol
        current_valence = len(self.connectivity[atom_idx])
        expected_valence = STANDARD_VALENCE.get(symbol, 4)
        
        # Check standard valence
        if current_valence == expected_valence:
            return True
        
        # Check aromatic carbon
        if use_aromatic_check and self.is_aromatic_carbon(atom_idx):
            return True
        
        # Check alternative valences
        if symbol in ALTERNATIVE_VALENCE:
            return current_valence in ALTERNATIVE_VALENCE[symbol]
        
        return False
    
    def validate_all(self, use_aromatic_check=True):
        """
        Validate all atoms. Returns list of violations.
        
        Returns:
        --------
        list : List of dicts with violation info
        """
        violations = []
        for i in range(len(self.atoms)):
            if not self.validate_atom(i, use_aromatic_check):
                symbol = self.atoms[i].symbol
                current = len(self.connectivity[i])
                expected = STANDARD_VALENCE.get(symbol, 4)
                
                violations.append({
                    'atom_idx': i,
                    'symbol': symbol,
                    'current_valence': current,
                    'expected_valence': expected,
                    'excess': current - expected,
                    'is_aromatic': self.is_aromatic_carbon(i)
                })
        return violations
    
    def remove_longest_bonds(self, atom_idx, n_remove):
        """Remove n longest bonds from atom."""
        atom_bonds = [(bond, self.bond_lengths[bond]) 
                      for bond in self.bonds if atom_idx in bond]
        atom_bonds.sort(key=lambda x: x[1], reverse=True)
        
        removed = 0
        for bond, length in atom_bonds:
            if removed >= n_remove:
                break
            if bond in self.bonds:
                self.bonds.remove(bond)
                del self.bond_lengths[bond]
                removed += 1
                print(f"  Removed: {bond} ({length:.3f} Å)")
        
        self._update_connectivity()
        # Invalidate ring cache
        self._rings = None
        self._atom_in_ring = None
        return removed
    
    def add_shortest_bonds(self, atom_idx, n_add):
        """Add n shortest missing bonds to atom."""
        candidates = []
        for j in range(len(self.atoms)):
            if j == atom_idx or j in self.connectivity[atom_idx]:
                continue
            dist = self.atoms.get_distance(atom_idx, j, mic=True)
            if dist < 3.0:  # Reasonable bond distance
                candidates.append((j, dist))
        
        candidates.sort(key=lambda x: x[1])
        
        added = 0
        for j, dist in candidates[:n_add]:
            bond = tuple(sorted([atom_idx, j]))
            self.bonds.append(bond)
            self.bond_lengths[bond] = dist
            added += 1
            print(f"  Added: {bond} ({dist:.3f} Å)")
        
        self._update_connectivity()
        # Invalidate ring cache
        self._rings = None
        self._atom_in_ring = None
        return added
    
    def apply_correction(self, max_iterations=10, use_aromatic_check=True, verbose=True):
        """
        Apply valence correction iteratively.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum correction iterations (default 10)
        use_aromatic_check : bool
            Use aromatic carbon detection (default True)
        verbose : bool
            Print progress (default True)
        
        Returns:
        --------
        bool : True if all violations resolved
        """
        if verbose:
            print("\n" + "="*60)
            print("VALENCE CORRECTION")
            print("="*60)
        
        for iteration in range(max_iterations):
            violations = self.validate_all(use_aromatic_check)
            
            if not violations:
                if verbose:
                    print(f"\n✓ Converged at iteration {iteration}")
                return True
            
            if verbose:
                print(f"\nIteration {iteration + 1}: {len(violations)} violations")
            
            # Fix over-coordinated atoms
            for v in [v for v in violations if v['excess'] > 0]:
                if verbose:
                    print(f"Atom {v['atom_idx']} ({v['symbol']}): "
                          f"{v['current_valence']} bonds → {v['expected_valence']}")
                self.remove_longest_bonds(v['atom_idx'], v['excess'])
            
            # Fix under-coordinated atoms
            for v in [v for v in violations if v['excess'] < 0]:
                if verbose:
                    print(f"Atom {v['atom_idx']} ({v['symbol']}): "
                          f"{v['current_valence']} bonds → {v['expected_valence']}")
                self.add_shortest_bonds(v['atom_idx'], abs(v['excess']))
        
        # Final check
        violations = self.validate_all(use_aromatic_check)
        if violations and verbose:
            print("\n" + "="*60)
            print("REMAINING VIOLATIONS")
            print("="*60)
            for v in violations:
                symbol = v['symbol']
                current = v['current_valence']
                if v['is_aromatic']:
                    print(f"✓ Atom {v['atom_idx']} ({symbol}): "
                          f"Valence {current} accepted (aromatic)")
                elif symbol in ALTERNATIVE_VALENCE:
                    alts = ALTERNATIVE_VALENCE[symbol]
                    if current in alts:
                        print(f"✓ Atom {v['atom_idx']} ({symbol}): "
                              f"Valence {current} accepted (alternative)")
                    else:
                        print(f"⚠ Atom {v['atom_idx']} ({symbol}): "
                              f"Valence {current} cannot be resolved")
        
        return len(violations) == 0
    
    def build_graph(self, apply_correction=False, use_aromatic_check=True, 
                   detect_rings=True, max_ring_size=8, verbose=True):
        """
        One-step graph building with structural features.
        
        Parameters:
        -----------
        apply_correction : bool
            Apply valence correction (default False)
        use_aromatic_check : bool
            Use aromatic carbon detection (default True)
        detect_rings : bool
            Detect ring structures (default True)
        max_ring_size : int
            Maximum ring size to detect (default 8)
        verbose : bool
            Print progress (default True)
        
        Returns:
        --------
        dict : Graph representation with structural features
        """
        # Build connectivity
        self.build_connectivity()
        
        # Validate
        if verbose:
            violations = self.validate_all(use_aromatic_check)
            if violations:
                print(f"\nInitial validation: {len(violations)} violations")
                if apply_correction:
                    self.apply_correction(use_aromatic_check=use_aromatic_check, 
                                        verbose=verbose)
            else:
                print("\n All atoms satisfy valence requirements")
        elif apply_correction:
            self.apply_correction(use_aromatic_check=use_aromatic_check, 
                                verbose=False)
        
        # Detect rings if requested
        if detect_rings:
            if verbose:
                print(f"\nDetecting rings (max size {max_ring_size})...")
            rings = self.detect_rings(max_ring_size)
            in_ring = self.get_atom_in_ring(max_ring_size)
            if verbose:
                n_rings = len(rings)
                n_atoms_in_rings = np.sum(in_ring)
                print(f"Found {n_rings} rings containing {n_atoms_in_rings} atoms")
        
        # Return graph with structural features
        return self.get_graph()
    
    def get_graph(self):
        """
        Get graph representation with structural features.
        
        Returns:
        --------
        dict with keys:
            - edge_list: [(i, j, distance), ...]
            - adjacency: {i: [neighbors], ...}
            - n_atoms: int
            - n_bonds: int
            - distances: [distance, ...]
            - rbf_features: ndarray [n_bonds, n_rbf_centers]
            - unit_vectors: {(i, j): unit_vec, ...}
            - degrees: ndarray [n_atoms]
            - degrees_weighted: ndarray [n_atoms]
            - in_ring: ndarray [n_atoms] boolean
            - rings: list of sets
        """
        # Get all structural features
        distances = list(self.bond_lengths.values())
        rbf_features = self.compute_rbf(distances)
        unit_vectors = self.get_unit_vectors()
        degrees = self.get_degrees(weighted=False)
        degrees_weighted = self.get_degrees(weighted=True)
        in_ring = self.get_atom_in_ring()
        rings = self.detect_rings() if self._rings else []
        
        return {
            'edge_list': [(i, j, self.bond_lengths[(i, j)]) 
                         for i, j in self.bonds],
            'adjacency': dict(self.connectivity),
            'n_atoms': len(self.atoms),
            'n_bonds': len(self.bonds),
            'distances': distances,
            'rbf_features': rbf_features,
            'unit_vectors': unit_vectors,
            'degrees': degrees,
            'degrees_weighted': degrees_weighted,
            'in_ring': in_ring,
            'rings': rings
        }
    
    def print_summary(self):
        """Print molecular graph summary with structural features."""
        print("\n" + "="*60)
        print("MOLECULAR GRAPH SUMMARY (Structure-Focused)")
        print("="*60)
        print(f"Formula: {self.atoms.get_chemical_formula()}")
        print(f"Atoms: {len(self.atoms)}")
        print(f"Bonds: {len(self.bonds)}")
        
        # Degree statistics
        degrees = self.get_degrees(weighted=False)
        print(f"\nDegree statistics:")
        print(f"  Min: {int(np.min(degrees))}, Max: {int(np.max(degrees))}, "
              f"Mean: {np.mean(degrees):.2f}")
        
        # Ring statistics
        in_ring = self.get_atom_in_ring()
        rings = self.detect_rings() if self._rings else []
        n_atoms_in_rings = np.sum(in_ring)
        print(f"\nRing statistics:")
        print(f"  Rings: {len(rings)}")
        print(f"  Atoms in rings: {n_atoms_in_rings}/{len(self.atoms)}")
        if rings:
            ring_sizes = [len(r) for r in rings]
            print(f"  Ring sizes: {min(ring_sizes)}-{max(ring_sizes)}")
        
        # Bond length statistics
        if self.bond_lengths:
            lengths = list(self.bond_lengths.values())
            print(f"\nBond lengths: {min(lengths):.3f} - {max(lengths):.3f} Å "
                  f"(mean: {np.mean(lengths):.3f})")
        
        # RBF parameters
        print(f"\nRBF parameters:")
        print(f"  Centers: {self.rbf_params['num_centers']}")
        print(f"  Range: {self.rbf_params['start']:.1f} - {self.rbf_params['stop']:.1f} Å")
        print(f"  Width: {self.rbf_params['width']:.2f}")