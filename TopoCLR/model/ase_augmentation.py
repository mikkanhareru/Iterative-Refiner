"""Agumentation strategies for molecular SSL pretraining
"""
import torch
import dgl
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

class MolecularAugmentation:
    """
    Augmentation for contrastive learning
    """
    def __init__(self,
                 node_drop_rate: float=0.1,
                 edge_drop_rate: float=0.1,
                 feature_mask_rate: float=0.1,
                 subgraph_sample_rate:float=0.8,
                 apply_noise_coords:bool=False,
                 noise_scale:float=0.1, # Gaussian noise for RBF features
                 seed:Optional[int]=None
                 ):
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate
        self.subgraph_sample_rate = subgraph_sample_rate
        self.apply_noise_coords = apply_noise_coords
        self.noise_scale = noise_scale

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def node_dropping(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Randomly drop nodes from the graph

        Args:
            graph: Input DGL graph

        Returns:
            dgl.DGLGraph: Augmented dgl graph with nodes dropped
        """
        n_nodes = graph.num_nodes()

        # Don't drop too many nodes
        if n_nodes <= 3:
            return graph
        
        # Randomly select nodes to keep
        keep_prob = 1 - self.node_drop_rate
        keep_nodes = torch.rand(n_nodes) < keep_prob

        # Ensure at least 3 nodes remain (to maintain edges)
        if keep_nodes.sum() < 3:
            keep_nodes[:3] = True
        
        keep_indices = torch.where(keep_nodes)[0]

        # Create subgraph without storing node IDs
        # store_ids=False prevents DGL from adding '_ID' to ndata
        aug_graph = dgl.node_subgraph(graph, keep_indices, store_ids=False)
        
        # Validate schema was preserved
        if not validate_graph_schema(aug_graph):
            return graph
        
        # Safety check: if no edges, return original
        if aug_graph.num_edges() == 0:
            return graph

        return aug_graph
    
    def edge_dropping(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        randomly drop edges from the graph

        Args:
            graph (dgl.DGLGraph): Input DGL graph

        Returns:
            dgl.DGLGraph: Augmented graph with edges dropped
        """
        src, dst = graph.edges()
        n_edges = len(src)

        # Do not drop too many edges
        # Conservative: need buffer above num_nodes for safe dropping
        if n_edges <= graph.num_nodes() + 2:
            return graph
        
        # Randomly select edges to keep
        keep_prob = 1 - self.edge_drop_rate
        keep_edges = torch.rand(n_edges) < keep_prob

        # Ensure at least num_nodes edges for better connectivity
        min_edges = graph.num_nodes()
        if keep_edges.sum() < min_edges:
            # Keep the first min_edges to ensure connectivity
            keep_edges[:min_edges] = True

        keep_indices = torch.where(keep_edges)[0]
        
        # Safety check
        if len(keep_indices) == 0:
            return graph

        # Create subgraph with selected edges
        # CRITICAL: preserve_nodes=True keeps all nodes and their features
        aug_graph = dgl.edge_subgraph(graph, keep_indices, relabel_nodes=False, store_ids=False)
        
        # CRITICAL: Manually preserve edge features that might be lost
        if len(aug_graph.edata) == 0 and len(graph.edata) > 0:
            # Edge data was lost - copy from original edges
            for key in graph.edata.keys():
                if key != '_ID':  # Skip DGL internal keys
                    aug_graph.edata[key] = graph.edata[key][keep_indices]
        
        # Validate schema was preserved
        if not validate_graph_schema(aug_graph):
            return graph
        
        # Safety check: if no edges resulted, return original
        if aug_graph.num_edges() == 0:
            return graph

        return aug_graph
    
    def feature_masking(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        randomly mask mode feautres

        Args:
            graph: Input DGL graph

        Returns:
            dgl.DGLGraph: Agumented graph with masked features
        """
        aug_graph = graph.clone()
        n_nodes = aug_graph.num_nodes()

        # Mask atomic numbers
        if 'atomic_num' in aug_graph.ndata:
            mask = torch.rand(n_nodes) < self.feature_mask_rate
            # use 0 as mask token for atomic numbers
            aug_graph.ndata['atomic_num'] = aug_graph.ndata['atomic_num'].clone()
            aug_graph.ndata['atomic_num'][mask] = 0
        
        return aug_graph
        
    def subgraph_sampling(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """Sampling a connected subgraph via random walk
        """
        n_nodes = graph.num_nodes()
        target_nodes = max(3, int(n_nodes * self.subgraph_sample_rate))

        # return as it when the graph is too small
        if n_nodes <= 5:
            return graph
        
        # random walk to sample connected subgraph
        start_node = torch.randint(0, n_nodes, (1,)).item()
        sampled_nodes = {start_node}

        current_nodes = [start_node]

        while len(sampled_nodes) < target_nodes and current_nodes:
            node = random.choice(current_nodes)

            # Get neighbors
            successors = graph.successors(node).tolist()
            predecessors = graph.predecessors(node).tolist()
            neighbors = list(set(successors + predecessors))

            if neighbors: 
                # Add a random neighbor
                neighbor = random.choice(neighbors)
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    current_nodes.append(neighbor)
            
            current_nodes.remove(node)
        
        # Ensure we have at least 3 nodes
        if len(sampled_nodes) < 3:
            return graph
        
        sampled_indices = torch.tensor(list(sampled_nodes), dtype=torch.long)
        # Create subgraph without storing node IDs
        # store_ids=False prevents DGL from adding '_ID' to ndata
        aug_graph = dgl.node_subgraph(graph, sampled_indices, store_ids=False)
        
        # Validate schema was preserved
        if not validate_graph_schema(aug_graph):
            return graph
        
        # Safety check: if no edges, return original
        if aug_graph.num_edges() == 0:
            return graph

        return aug_graph
    
    def add_noise_to_rbf(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Add Gaussian noise to RBF edge features
        
        Args:
            graph: Input DGL graph with 'rbf' edge features
            
        Returns:
            Augmented graph with noisy RBF features
        """
        aug_graph = graph.clone()
        
        if 'rbf' in aug_graph.edata and self.apply_noise_coords:
            rbf_features = aug_graph.edata['rbf'].clone()
            noise = torch.randn_like(rbf_features) * self.noise_scale
            aug_graph.edata['rbf'] = rbf_features + noise
        
        return aug_graph
    
    def augment(
        self, 
        graph: dgl.DGLGraph, 
        strategy: str = 'random'
    ) -> dgl.DGLGraph:
        """
        Apply augmentation to a graph
        
        Args:
            graph: Input DGL graph
            strategy: Augmentation strategy
                - 'node_drop': Drop nodes
                - 'edge_drop': Drop edges
                - 'feature_mask': Mask features
                - 'subgraph': Sample subgraph
                - 'noise': Add noise to RBF
                - 'random': Randomly select one strategy
                - 'compose': Apply multiple strategies
                
        Returns:
            Augmented graph
        """
        if strategy == 'random':
            strategy = random.choice([
                'node_drop', 'edge_drop', 'feature_mask', 
                'subgraph', 'noise'
            ])
        
        aug_graph = None
        if strategy == 'node_drop':
            aug_graph = self.node_dropping(graph)
        elif strategy == 'edge_drop':
            aug_graph = self.edge_dropping(graph)
        elif strategy == 'feature_mask':
            aug_graph = self.feature_masking(graph)
        elif strategy == 'subgraph':
            aug_graph = self.subgraph_sampling(graph)
        elif strategy == 'noise':
            aug_graph = self.add_noise_to_rbf(graph)
        elif strategy == 'compose':
            # Apply multiple augmentations
            aug_graph = graph
            # Randomly select 2-4 augmentations from ALL available types
            n_augs = random.randint(2, 4)
            
            # Include ALL augmentation types for maximum diversity
            all_strategies = ['edge_drop', 'feature_mask', 'node_drop', 'subgraph']
            
            # Add noise if enabled
            if self.apply_noise_coords:
                all_strategies.append('noise')
            
            # Pick random subset (without replacement)
            n_to_select = min(n_augs, len(all_strategies))
            selected = random.sample(all_strategies, n_to_select)
            
            for strat in selected:
                aug_graph = self.augment(aug_graph, strategy=strat)
            
            # Final safety check
            if not validate_graph_schema(aug_graph):
                return graph
            
            return aug_graph
        else:
            aug_graph = graph
        
        # ULTIMATE SAFETY CHECK: Validate schema after any augmentation
        if not validate_graph_schema(aug_graph):
            return graph
        
        return aug_graph


def create_multi_view(
    molecule_data: Dict,
    augmentor: MolecularAugmentation,
    n_views: int = 2,
    use_different_graphs: bool = True
) -> List[dgl.DGLGraph]:
    """
    Create multiple augmented views of a molecule
    
    Args:
        molecule_data: Dictionary containing 'g_homo_outline' and 'g_homo_geometry'
        augmentor: MolecularAugmentation instance
        n_views: Number of views to create
        use_different_graphs: Whether to use both outline and geometry graphs
        
    Returns:
        List of augmented graphs (views)
    """
    views = []
    
    if use_different_graphs and n_views == 2:
        # Use the two different graph types as two views
        # CRITICAL: Apply DIFFERENT augmentations to create truly different views
        # Outline graphs need edge features added (they're topology-only by default)
        
        # View 1: Augmented outline graph
        # Use structural augmentations (edge drop, node drop)
        outline = molecule_data['g_homo_outline']
        outline = ensure_edge_features(outline)  # Add edge features if missing
        outline = ensure_node_features(outline)  # Add node features if missing
        
        # Apply structural changes to outline
        view1 = outline
        # First pass: edge dropping
        view1 = augmentor.edge_dropping(view1)
        # Second pass: node dropping (optional)
        if random.random() < 0.5:
            view1 = augmentor.node_dropping(view1)
        # Third pass: feature masking
        view1 = augmentor.feature_masking(view1)
        
        # Validate view1 schema
        if not validate_graph_schema(view1):
            view1 = outline  # Fallback to prepared outline
        
        # Ensure features after augmentation
        view1 = ensure_edge_features(view1)
        view1 = ensure_node_features(view1)
        views.append(view1)
        
        # View 2: Augmented geometry graph  
        # Use feature-based augmentations (feature mask, noise)
        geometry = molecule_data['g_homo_geometry']
        geometry = ensure_edge_features(geometry)  # Ensure 'e' exists
        geometry = ensure_node_features(geometry)  # Ensure 'h' exists
        
        # Apply feature changes to geometry
        view2 = geometry
        # First pass: feature masking
        view2 = augmentor.feature_masking(view2)
        # Second pass: add noise to RBF features
        if augmentor.apply_noise_coords:
            view2 = augmentor.add_noise_to_rbf(view2)
        # Third pass: edge dropping (with lower rate)
        if random.random() < 0.3:  # 30% chance
            view2 = augmentor.edge_dropping(view2)
        
        # Validate view2 schema
        if not validate_graph_schema(view2):
            view2 = geometry  # Fallback to prepared geometry
        
        # Ensure features after augmentation
        view2 = ensure_edge_features(view2)
        view2 = ensure_node_features(view2)
        views.append(view2)
    else:
        # Create n_views by augmenting the geometry graph
        # Use COMPOSE strategy for stronger augmentation
        base_graph = molecule_data['g_homo_geometry']
        base_graph = ensure_edge_features(base_graph)
        base_graph = ensure_node_features(base_graph)
        
        for i in range(n_views):
            # Apply MULTIPLE augmentations with compose
            # This creates more diverse views than single 'random' augmentation
            aug_view = augmentor.augment(base_graph, strategy='compose')
            
            # Validate augmented view schema
            if not validate_graph_schema(aug_view):
                aug_view = base_graph  # Fallback to original
            
            # Ensure features after augmentation
            aug_view = ensure_edge_features(aug_view)
            aug_view = ensure_node_features(aug_view)
            views.append(aug_view)
    
    return views


def ensure_node_features(graph: dgl.DGLGraph, num_features: int = 53) -> dgl.DGLGraph:
    """
    Ensure graph has 'h' node feature for model input
    Creates 'h' from existing features if it doesn't exist
    Also removes any DGL internal keys that cause batching issues
    
    Args:
        graph: DGL graph
        num_features: Expected feature dimension
        
    Returns:
        Graph with 'h' feature added
    """
    # CRITICAL: Remove DGL internal keys that cause batching issues
    # These are added by node_subgraph/edge_subgraph operations
    internal_keys = ['_ID', '_FEAT', '_SRC', '_DST']
    for key in internal_keys:
        if key in graph.ndata:
            del graph.ndata[key]
        if key in graph.edata:
            del graph.edata[key]
    
    # If 'h' already exists, return as is
    if 'h' in graph.ndata:
        return graph
    
    # Create 'h' from existing features
    num_nodes = graph.num_nodes()
    
    # Start with one-hot encoding of atomic numbers
    if 'atomic_num' in graph.ndata:
        atomic_nums = graph.ndata['atomic_num']
        # Create feature vector (simplified - expand as needed)
        h = torch.zeros(num_nodes, num_features)
        
        # First feature: atomic number normalized
        h[:, 0] = atomic_nums.float() / 100.0
        
        # Add other features if available
        if 'degree' in graph.ndata:
            h[:, 1] = graph.ndata['degree'].float()
        
        if 'in_ring' in graph.ndata:
            h[:, 2] = graph.ndata['in_ring'].float()
        
        # Store as 'h'
        graph.ndata['h'] = h
    else:
        # Fallback: create zeros
        graph.ndata['h'] = torch.zeros(num_nodes, num_features)
    
    return graph


def ensure_edge_features(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Ensure graph has edge features for model input
    Creates dummy RBF features if none exist (e.g., for outline graphs)
    Also removes any DGL internal keys that cause batching issues
    
    Args:
        graph: DGL graph
        
    Returns:
        Graph with 'e' feature
    """
    # CRITICAL: Remove DGL internal keys that cause batching issues
    internal_keys = ['_ID', '_FEAT', '_SRC', '_DST']
    for key in internal_keys:
        if key in graph.ndata:
            del graph.ndata[key]
        if key in graph.edata:
            del graph.edata[key]
    
    # If 'e' already exists, return as is
    if 'e' in graph.edata:
        return graph
    
    # If 'rbf' exists, copy to 'e'
    if 'rbf' in graph.edata:
        graph.edata['e'] = graph.edata['rbf']
        return graph
    
    # CRITICAL: If NO edge features exist, create dummy features
    # This is normal for outline graphs (topology only, no geometry)
    num_edges = graph.num_edges()
    if num_edges > 0 and len(graph.edata) == 0:
        # Create dummy RBF features (50-dim) for outline graphs
        # These won't have real geometric information, but allow processing
        graph.edata['rbf'] = torch.randn(num_edges, 50) * 0.01  # Small random values
        graph.edata['e'] = graph.edata['rbf']
    
    return graph


def validate_graph_schema(graph: dgl.DGLGraph) -> bool:
    """
    Validate that graph has proper schema for batching
    Note: Outline graphs may not have edge features (topology only)
    
    Returns:
        True if graph is valid, False otherwise
    """
    # Check 1: Must have at least 1 edge
    if graph.num_edges() == 0:
        return False
    
    # Check 2: Must have atomic_num at minimum for node features
    if 'atomic_num' not in graph.ndata:
        return False
    
    # Check 3: Edge features are optional
    # Outline graphs have no edge features by design (topology only)
    # Geometry graphs have 'rbf' features
    # Both are valid!
    
    return True


def ssl_collate_fn(
    batch: List[Dict],
    augmentor: MolecularAugmentation,
    n_views: int = 2,
    use_different_graphs: bool = True
) -> Tuple[List[dgl.DGLGraph], List[dgl.DGLGraph], torch.Tensor]:
    """
    Custom collate function for SSL training with augmentation
    
    Args:
        batch: List of molecule dictionaries
        augmentor: MolecularAugmentation instance
        n_views: Number of views per molecule
        use_different_graphs: Whether to use outline + geometry as two views
        
    Returns:
        Tuple of (view1_graphs, view2_graphs, batch_indices)
    """
    view1_list = []
    view2_list = []
    batch_indices = []
    
    for idx, mol_data in enumerate(batch):
        # Create multiple views
        # Features are added in create_multi_view before and after augmentation
        views = create_multi_view(
            mol_data,
            augmentor,
            n_views=n_views,
            use_different_graphs=use_different_graphs
        )
        
        # Views already have all required features from create_multi_view
        view1_list.append(views[0])
        view2_list.append(views[1])
        batch_indices.append(idx)
    
    # Batch graphs
    view1_batch = dgl.batch(view1_list)
    view2_batch = dgl.batch(view2_list)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    
    return view1_batch, view2_batch, batch_indices


def finetuning_collate_fn(
    batch: List[Dict],
    augmentor: Optional[MolecularAugmentation] = None,
    use_augmentation: bool = False,
    graph_type: str = 'geometry'
) -> Tuple[dgl.DGLGraph, torch.Tensor, List[str]]:
    """
    Collate function for fine-tuning on structure prediction
    
    Args:
        batch: List of molecule dictionaries
        augmentor: Optional augmentor for training augmentation
        use_augmentation: Whether to apply augmentation
        graph_type: Which graph to use ('outline' or 'geometry')
        
    Returns:
        Tuple of (batched_graphs, ground_truth_coords, mol_ids)
    """
    graphs = []
    coords = []
    mol_ids = []
    
    for mol_data in batch:
        # Select graph type
        if graph_type == 'geometry':
            graph = mol_data['g_homo_geometry']
        else:
            graph = mol_data['g_homo_outline']
        
        # Apply augmentation if needed (for training)
        if use_augmentation and augmentor is not None:
            graph = augmentor.augment(graph, strategy='compose')
        
        graphs.append(graph)
        coords.append(mol_data['ground_truth_coords'])
        mol_ids.append(mol_data['mol_id'])
    
    # Batch graphs
    batched_graph = dgl.batch(graphs)
    
    # Stack coordinates (will have variable sizes, need padding or special handling)
    batched_coords = torch.cat(coords, dim=0)  # [total_atoms, 3]
    
    return batched_graph, batched_coords, mol_ids