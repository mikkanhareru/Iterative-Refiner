"""
Molecular Topology SSL Model, inpired by MolCLR and InfoGraph 
with new topology discriminative tasks for strucuture prediction

1. Distance aware augemntation with RBF features
Multi-task topology prediction
Loss fnction: contrastive + local-global MI + topology discirminator

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

class GINEncoder(nn.Module):
    """
    Multi-scale GIN encode, get representations at all scales
    """
    def __init__(self, num_features, hidden_dim=256, num_layers=5, pool='mean'):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pool_type = pool

        # GIN convolutions
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else: 
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            self.convs.append(GINConv(mlp, aggregator_type='sum'))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Graph pooling
        if pool == 'mean':
            self.pool = AvgPooling()
        elif pool == 'add':
            self.pool = SumPooling()
        else:
            raise ValueError(f"unknown pooling method: {pool}")
        
    def forward(self,g, return_all_layers=True):
        """
        Args:
            g: Batched DGLGraph with node features in g.ndata['h]
            return_all_laers: Return multi-scale features
        Returns:
            h_graph: graph-level representation
            h_node: Node-level multi-scale representation
        """
        # Get node features
        h = g.ndata['h']

        xs = []

        for i in range(self.num_layers):
            # GIN 
            h = self.convs[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            xs.append(h)

        # Node-level: concatenate all layers
        if return_all_layers:
            h_node = torch.cat(xs, dim=1) #[num_nodes, hidden_dim*num_layers]
        else: 
            h_node = xs[-1] #[num_nodes, hidden_dim]

        # Graph-level: pool multi-sclae features
        h_graphs = []
        for x in xs:
            # Temporarily store features in graph
            h_g = self.pool(g, x) #[batch_size, hidden_dim]
            h_graphs.append(h_g)
        
        h_graph = torch.cat(h_graphs, dim=1) #[batch_size, hidden_dim*num_layers]

        return h_graph, h_node
    
class FF(nn.Module):
    """
    feed-forward network for discriminator (same as infograph)
    """
    def __init__(self, input_dim):
        super(FF, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
class LocalGlobalDiscriminator(nn.Module):
    """Discriminator for local-global mutual information 
    Distinguishes between h_graphs, h_node pairs from sam vs dfferent graphs
    """
    def __init__(self, hidden_dim):
        super(LocalGlobalDiscriminator, self).__init__()
        self.graph_encoder = FF(hidden_dim)
        self.node_encoder = FF(hidden_dim)

        # Bilinear layer for scoring
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, h_graph, h_node):
        """
        Args:
            h_graph (_type_): [batch_size, hidden_dim]
            h_node (_type_): [num_nodes, hidden_dim]
        
        Returns:
            h_graph_encoded: [batch_size, hidden_dim]
            h_node_encoded:[num_nodes, hidden_dim]
        """
        # encode
        h_graph_encoded = self.graph_encoder(h_graph)
        h_node_encoded = self.node_encoder(h_node)

        return h_graph_encoded, h_node_encoded
    
class TopologyDiscriminateHead(nn.Module):
    """
    Multi-task head for topology prediction
    Predicts strucutural features from leared representations
    """
    def __init__(self, hidden_dim):
        super(TopologyDiscriminateHead, self).__init__()

        # Task 1 ring features (node-level)
        self.ring_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Task 2: Node degree
        self.degree_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Task 3: Distnace base prediction
        self.distance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 50) # predit 50dim RBF 
        )

    def forward(self, h_node, g):
        """
        Args:
            h_node (_type_): Node representations [num_nodes, hidden_dim]
            g (_type_): _DGL graph with edges
        
        Returns:
            Dictionary of predictions
        """
        outputs = {}

        # Ring prediction 
        outputs['ring'] = torch.sigmoid(self.ring_predictor(h_node)) #[num_nodes, 1]

        # Degree prediction 
        outputs['degree'] = self.degree_predictor(h_node) #[num_nodes, 1]

        # Distnace prediction (For existing edges)
        src, dst = g.edges()
        edge_repr = torch.cat([h_node[src], h_node[dst]], dim=1)
        outputs['distance'] = self.distance_predictor(edge_repr) #[num_edges, 50]

        return outputs
    
class TopoSSL(nn.Module):
    """
    Complete SSL model
    1. Graph-level contrastive learning 
    2. Local-global MI maximization
    3. Topology discriminative prediction
    """
    def __init__(self, num_features=53, hidden_dim=256, num_layers=5,
                 proj_dim=128, pool='mean'):
        super(TopoSSL, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared encoder 
        self.encoder = GINEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pool=pool
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        # Local-global discriminator
        self.discriminator = LocalGlobalDiscriminator(hidden_dim * num_layers)

        # Topology discriminative head 
        self.topo_head = TopologyDiscriminateHead(hidden_dim * num_layers)

    def forward(self, g):
        """
        Forward pass for batched graph

        Args:
            g (_type_): Batched DGLGraph with node featres in g.ndata['h']

        Returns:
            h_graph: Graph-level representation 
            z_graph: Projected representation fr contrastive learning
            h_node: Node-level multi-scale representation
            topo_pred: Topology predictions
        """
        # Encode
        h_graph, h_node = self.encoder(g)

        # Project for contrastive learning
        z_graph = self.projection(h_graph)

        # Topology predictions
        topo_pred = self.topo_head(h_node, g)

        return h_graph, z_graph, h_node, topo_pred
    
    def forward_contrast(self, g1, g2):
        """
        Forward pass for two augmented views (contrastive learning)

        Args:
            g1 (_type_): batched view 1
            g2 (_type_): bathced view 2
        
        Returns: 
            Dictionary with outputs from both views
        """
        # Encode both views
        h_g1, z_g1, h_n1, topo_pred1 = self.forward(g1)
        h_g2, z_g2, h_n2, topo_pred2 = self.forward(g2)

        return {
            'view1': {'h_graph': h_g1, 'z_graph': z_g1, 'h_node': h_n1, 'topo': topo_pred1, 'g': g1},
            'view2': {'h_graph': h_g2, 'z_graph': z_g2, 'h_node': h_n2, 'topo': topo_pred2, 'g': g2}
        }
    
    def compute_node_degrees_dgl(g):
        return g.in_degrees().float()
    
if __name__ == '__main__':
    import numpy as np

    print("test model...")

    # Create dummy DGL graphs
    def create_dummy_graph(num_nodes=20, num_features=53):
        # Random edges
        src = np.random.randint(0, num_nodes, size=40)
        dst = np.random.randint(0, num_nodes, size=40)
        
        g = dgl.graph((src, dst))
        
        # Add node features
        g.ndata['h'] = torch.randn(num_nodes, num_features)
        
        # Add edge features (RBF)
        g.edata['e'] = torch.randn(g.num_edges(), 50)
        
        # Add ring mask
        g.ndata['ring'] = torch.randint(0, 2, (num_nodes,)).float()
        
        return g
    
    # Create batch of graphs
    graphs = [create_dummy_graph() for _ in range(4)]
    batched_g = dgl.batch(graphs)
    
    print(f"Created batched graph:")
    print(f"  Num graphs: {batched_g.batch_size}")
    print(f"  Total nodes: {batched_g.num_nodes()}")
    print(f"  Total edges: {batched_g.num_edges()}")

    # Initialize model 
    model = TopoSSL(num_features=53, hidden_dim=256, num_layers=5)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Forward pass
    print("\nTesting forward pass...")
    h_graph, z_graph, h_node, topo_pred = model(batched_g)
    
    print(f"✓ Forward pass successful!")
    print(f"  Graph representation: {h_graph.shape}")
    print(f"  Projected representation: {z_graph.shape}")
    print(f"  Node representation: {h_node.shape}")
    print(f"  Topology predictions:")
    print(f"    - Ring: {topo_pred['ring'].shape}")
    print(f"    - Degree: {topo_pred['degree'].shape}")
    print(f"    - Distance: {topo_pred['distance'].shape}")
    
    # Test contrastive forward
    print("\nTesting contrastive forward...")
    graphs2 = [create_dummy_graph() for _ in range(4)]
    batched_g2 = dgl.batch(graphs2)
    
    outputs = model.forward_contrast(batched_g, batched_g2)
    
    print(f"✓ Contrastive forward successful!")
    print(f"  View 1 z_graph: {outputs['view1']['z_graph'].shape}")
    print(f"  View 2 z_graph: {outputs['view2']['z_graph'].shape}")
    
    print("\n" + "="*60)
    print("✓ DGL Model test passed! Model is working correctly.")
    print("="*60)






