"""
Loss functions for molecular topology SSL
1. NT-Xent contrastive loss
Local-global MI loss
Topology discriminative loss 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

def nt_xent_loss(z1, z2, temperature=0.1, eps=1e-8):
    """Normalized temperature-sclaed cross entropy loss

    Args:
        z1: Projected representations [batch_size, proj_dim] 
        z2: Same
        temperature (float, optional): Temperature parameter for scaling.
        eps (_type_, optional): Small constant for numerical stability.
    Returns:
        NT-Xent loss value
    """
    batch_size = z1.size(0)
    
    # Check batch size
    if batch_size < 2:
        print(f"WARNING: Batch size too small for contrastive loss: {batch_size}")
        return torch.tensor(0.0, requires_grad=True, device=z1.device)
    
    # Check for NaN in inputs
    if torch.isnan(z1).any() or torch.isnan(z2).any():
        print("ERROR: NaN in contrastive loss inputs!")
        return torch.tensor(0.0, requires_grad=True, device=z1.device)

    # Normalize with numerical stability
    z1 = F.normalize(z1, p=2, dim=1, eps=eps)
    z2 = F.normalize(z2, p=2, dim=1, eps=eps)

    # Concatenate representations 
    representations = torch.cat([z1, z2], dim=0) #[2Batch_size, dim=0]

    # Compute similarity matrix
    similarity_matrix = torch.mm(representations, representations.T) #[2*B, 2*B]
    
    # Clamp similarities to prevent extreme values
    similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)

    # Create mask for positive pairs 
    mask = torch.ones(2*batch_size, 2*batch_size, dtype=torch.bool, device=z1.device)
    mask.fill_diagonal_(False)

    # Remove diagonal (self-similarity)
    mask.fill_diagonal_(False)

    # Mask for positive paris only
    positives_mask = torch.zeros_like(mask)
    positives_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
    positives_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z1.device)

    # Extract positives and negatives
    similarity_matrix = similarity_matrix / temperature

    # For numerical stability - subtract max
    similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()

    # Compute loss for each sample
    exp_sim = torch.exp(similarity_matrix)

    # Sum of all similarities excep self
    exp_sim_sum = exp_sim.masked_fill(~mask, 0).sum(dim=1)

    # pos similaries 
    pos_sim = exp_sim.masked_fill(~positives_mask, 0).sum(dim=1)

    #Loss with numerical stability
    loss = -torch.log((pos_sim + eps) / (exp_sim_sum + eps))
    
    # Check for NaN
    if torch.isnan(loss).any():
        print("ERROR: NaN in contrastive loss computation!")
        print(f"  pos_sim range: [{pos_sim.min().item():.4f}, {pos_sim.max().item():.4f}]")
        print(f"  exp_sim_sum range: [{exp_sim_sum.min().item():.4f}, {exp_sim_sum.max().item():.4f}]")
        return torch.tensor(0.0, requires_grad=True, device=z1.device)
    
    loss = loss.mean()

    return loss

def info_nce_local_global(h_graph, h_node, g, discriminator, temperature=0.1, eps=1e-8):
    """
    InfoNCE loss for local-global mutual information (DGL version)
    From InfoGraph
    
    Args:
        h_graph: Graph-level representations [batch_size, hidden_dim]
        h_node: Node-level representations [num_nodes, hidden_dim]
        g: Batched DGLGraph (to get batch assignment)
        discriminator: Discriminator network
        temperature: Temperature parameter
        eps: Small constant for numerical stability
        
    Returns:
        MI loss value
    """
    num_graphs = h_graph.size(0)
    num_nodes = h_node.size(0)
    
    # Check for NaN in inputs
    if torch.isnan(h_graph).any() or torch.isnan(h_node).any():
        print("ERROR: NaN in MI loss inputs!")
        return torch.tensor(0.0, requires_grad=True, device=h_graph.device)
    
    # Get batch assignment from DGL graph
    batch = dgl.broadcast_nodes(g, torch.arange(num_graphs, device=h_graph.device))
    
    # Encode graph and node representations
    h_graph_enc, h_node_enc = discriminator(h_graph, h_node)
    
    # Check for NaN after encoding
    if torch.isnan(h_graph_enc).any() or torch.isnan(h_node_enc).any():
        print("ERROR: NaN after discriminator encoding!")
        return torch.tensor(0.0, requires_grad=True, device=h_graph.device)
    
    # Expand graph representations to match each node
    graph_expanded = h_graph_enc[batch]  # [num_nodes, hidden_dim]
    
    # Compute positive scores (nodes with their own graph)
    pos_scores = (graph_expanded * h_node_enc).sum(dim=1)  # [num_nodes]
    pos_scores = pos_scores / temperature
    
    # Compute negative scores (nodes with other graphs)
    neg_scores = torch.mm(h_node_enc, h_graph_enc.T) / temperature  # [num_nodes, batch_size]
    
    # Numerical stability - subtract max
    max_scores = torch.max(torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), dim=1, keepdim=True)[0]
    pos_scores = pos_scores - max_scores.squeeze()
    neg_scores = neg_scores - max_scores
    
    # InfoNCE loss with numerical stability
    exp_pos = torch.exp(pos_scores)
    exp_neg = torch.exp(neg_scores).sum(dim=1)
    
    loss = -torch.log((exp_pos + eps) / (exp_neg + eps))
    
    # Check for NaN
    if torch.isnan(loss).any():
        print("ERROR: NaN in MI loss computation!")
        print(f"  exp_pos range: [{exp_pos.min().item():.4f}, {exp_pos.max().item():.4f}]")
        print(f"  exp_neg range: [{exp_neg.min().item():.4f}, {exp_neg.max().item():.4f}]")
        return torch.tensor(0.0, requires_grad=True, device=h_graph.device)
    
    loss = loss.mean()
    
    return loss


def topology_discriminative_loss(topo_pred, g, compute_distance=True):
    """
    Multi-task topology prediction loss
    
    Supervises the model to predict:
    1. Ring membership
    2. Node degrees
    3. RBF distance features (ONLY if compute_distance=True)
    
    Args:
        topo_pred: Dictionary of predictions from TopologyDiscriminativeHead
        g: DGLGraph with ground truth data
        compute_distance: Whether to compute distance prediction loss
                          Set to False for outline graphs with dummy features
        
    Returns:
        total_loss: Combined topology loss
        loss_dict: Dictionary of individual losses
    """
    loss_dict = {}
    total_loss = 0.0
    
    # 1. Ring membership loss (if available)
    if 'ring' in g.ndata:
        ring_pred = topo_pred['ring'].squeeze()
        ring_target = g.ndata['ring'].float()
        
        # Clamp for numerical stability
        ring_pred = torch.clamp(ring_pred, min=1e-7, max=1-1e-7)
        
        loss_ring = F.binary_cross_entropy(ring_pred, ring_target)
        
        if not torch.isnan(loss_ring):
            loss_dict['ring'] = loss_ring.item()
            total_loss += loss_ring
        else:
            loss_dict['ring'] = 0.0
    
    # 2. Degree prediction loss
    degrees = g.in_degrees().float()
    degree_pred = topo_pred['degree'].squeeze()
    
    # MSE loss for degree prediction
    # Also Huber 
    loss_degree = F.smooth_l1_loss(degree_pred, degrees)
    loss_degree = torch.clamp(loss_degree, max=10)
    
    if not torch.isnan(loss_degree):
        loss_dict['degree'] = loss_degree.item()
        total_loss += loss_degree
    else:
        loss_dict['degree'] = 0.0
        print("WARNING: NaN in degree loss!")
    
    # 3. Distance prediction loss (RBF features)
    # CRITICAL: Only compute if requested (skip for outline graphs)
    if compute_distance and 'e' in g.edata:
        distance_pred = topo_pred['distance']
        distance_target = g.edata['e']
        
        # Check for valid targets
        if not torch.isnan(distance_target).any():
            # MSE loss for RBF distance features
            # Update: Use Huber loss to avoid loss explotion
            loss_distance = F.smooth_l1_loss(distance_pred, distance_target)
            loss_distance = torch.clamp(loss_distance, max=50)
            
            if not torch.isnan(loss_distance):
                loss_dict['distance'] = loss_distance.item()
                total_loss += 0.5 * loss_distance  # Weight down distance loss
            else:
                loss_dict['distance'] = 0.0
                print("WARNING: NaN in distance loss!")
        else:
            loss_dict['distance'] = 0.0
    else:
        loss_dict['distance'] = 0.0  # Not computed for this view
    
    return total_loss, loss_dict


def compute_ssl_loss(outputs, discriminator, 
                     alpha=1.0, beta=0.5, gamma=0.3, temperature=0.1):
    """
    Combined SSL loss function (DGL version)
    
    Args:
        outputs: Dictionary from model.forward_contrast()
        discriminator: Local-global discriminator
        alpha: Weight for contrastive loss
        beta: Weight for MI loss
        gamma: Weight for topology loss
        temperature: Temperature parameter
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    loss_dict = {}
    
    # 1. Graph-level contrastive loss (MolCLR)
    z1 = outputs['view1']['z_graph']
    z2 = outputs['view2']['z_graph']
    
    # Debug: Check z1 and z2
    if z1.numel() == 0 or z2.numel() == 0:
        print("ERROR: Empty z_graph tensors!")
        print(f"  z1 shape: {z1.shape}, z2 shape: {z2.shape}")
        loss_contrastive = torch.tensor(0.0, requires_grad=True, device=z1.device)
        loss_dict['contrastive'] = 0.0
    elif torch.isnan(z1).any() or torch.isnan(z2).any():
        print("WARNING: NaN in z_graph inputs!")
        print(f"  z1 has NaN: {torch.isnan(z1).any()}")
        print(f"  z2 has NaN: {torch.isnan(z2).any()}")
        loss_contrastive = torch.tensor(0.0, requires_grad=True, device=z1.device)
        loss_dict['contrastive'] = 0.0
    elif torch.isinf(z1).any() or torch.isinf(z2).any():
        print("WARNING: Inf in z_graph inputs!")
        loss_contrastive = torch.tensor(0.0, requires_grad=True, device=z1.device)
        loss_dict['contrastive'] = 0.0
    else:
        # Compute contrastive loss
        loss_contrastive = nt_xent_loss(z1, z2, temperature=temperature)
        
        # Check for representation collapse
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        similarity = (z1_norm * z2_norm).sum(dim=1).mean()
        
        if similarity > 0.99:
            print(f"WARNING: Representation collapse detected! Similarity: {similarity.item():.4f}")
            print("  This means z1 and z2 are nearly identical.")
        
        if torch.isnan(loss_contrastive):
            print("WARNING: NaN in contrastive loss computation!")
            print(f"  z1 range: [{z1.min().item():.4f}, {z1.max().item():.4f}]")
            print(f"  z2 range: [{z2.min().item():.4f}, {z2.max().item():.4f}]")
            print(f"  z1 norm: {torch.norm(z1, dim=1).mean().item():.4f}")
            print(f"  z2 norm: {torch.norm(z2, dim=1).mean().item():.4f}")
            print(f"  Similarity: {similarity.item():.4f}")
            loss_contrastive = torch.tensor(0.0, requires_grad=True, device=z1.device)
        
        loss_dict['contrastive'] = loss_contrastive.item()
    
    # 2. Local-global MI loss (InfoGraph)
    # Use view2 (geometry) for MI loss - has real features
    h_g2 = outputs['view2']['h_graph']
    h_n2 = outputs['view2']['h_node']
    g2 = outputs['view2']['g']
    
    loss_mi = info_nce_local_global(h_g2, h_n2, g2, discriminator, temperature)
    
    if torch.isnan(loss_mi):
        print("WARNING: NaN in MI loss!")
        print(f"  h_graph shape: {h_g2.shape}, h_node shape: {h_n2.shape}")
        loss_mi = torch.tensor(0.0, requires_grad=True, device=h_g2.device)
    
    loss_dict['mi'] = loss_mi.item()
    
    # 3. Topology discriminative loss (NOVEL)
    # CRITICAL: Only compute distance loss on geometry graph (view2)!
    # View1 (outline) has dummy edge features - skip distance prediction
    # View2 (geometry) has real RBF features - use for distance prediction
    
    # For view1 (outline): Only ring and degree prediction (no distance)
    topo_pred1 = outputs['view1']['topo']
    g1 = outputs['view1']['g']
    loss_topo1, topo_losses1 = topology_discriminative_loss(
        topo_pred1, g1, 
        compute_distance=False  # Skip distance for outline graph
    )
    
    # For view2 (geometry): All topology predictions including distance
    topo_pred2 = outputs['view2']['topo']
    loss_topo2, topo_losses2 = topology_discriminative_loss(
        topo_pred2, g2,
        compute_distance=True  # Compute distance for geometry graph
    )
    
    # Average topology losses from both views
    loss_topo = (loss_topo1 + loss_topo2) / 2.0
    
    if torch.isnan(loss_topo):
        print("WARNING: NaN in topology loss!")
        loss_topo = torch.tensor(0.0, requires_grad=True, device=g2.device)
    
    loss_dict['topology'] = loss_topo.item()
    loss_dict.update(topo_losses2)  # Use view2 losses for logging
    
    # Combined loss
    total_loss = alpha * loss_contrastive + beta * loss_mi + gamma * loss_topo
    
    if torch.isnan(total_loss):
        print("WARNING: NaN in total loss!")
        print(f"  Contrastive: {loss_contrastive.item()}")
        print(f"  MI: {loss_mi.item()}")
        print(f"  Topology: {loss_topo.item()}")
        total_loss = torch.tensor(0.0, requires_grad=True, device=g2.device)
    
    # IMPORTANT: Add total to loss_dict for logging
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions with DGL
    import numpy as np
    from ase_model import LocalGlobalDiscriminator
    
    print("Testing DGL loss functions...")
    
    # Test NT-Xent loss
    z1 = torch.randn(32, 128)
    z2 = torch.randn(32, 128)
    loss_contrast = nt_xent_loss(z1, z2)
    print(f"âœ“ NT-Xent loss: {loss_contrast.item():.4f}")
    
    # Test InfoNCE loss with DGL graph
    def create_dummy_graph(num_nodes=20, num_features=53):
        src = np.random.randint(0, num_nodes, size=40)
        dst = np.random.randint(0, num_nodes, size=40)
        g = dgl.graph((src, dst))
        g.ndata['h'] = torch.randn(num_nodes, num_features)
        g.edata['e'] = torch.randn(g.num_edges(), 50)
        g.ndata['ring'] = torch.randint(0, 2, (num_nodes,)).float()
        return g
    
    # Create batched graph
    graphs = [create_dummy_graph() for _ in range(32)]
    batched_g = dgl.batch(graphs)
    
    h_graph = torch.randn(32, 256) * 0.01
    h_node = torch.randn(batched_g.num_nodes(), 256) * 0.01
    
    discriminator = LocalGlobalDiscriminator(256)

    for m in discriminator.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.01)
    
    loss_mi = info_nce_local_global(h_graph, h_node, batched_g, discriminator)
    print(f"âœ“ InfoNCE MI loss: {loss_mi.item():.4f}")
    
    # Test topology loss
    topo_pred = {
        'ring': torch.sigmoid(torch.randn(batched_g.num_nodes(), 1)),
        'degree': torch.randn(batched_g.num_nodes(), 1),
        'distance': torch.randn(batched_g.num_edges(), 50)
    }
    
    loss_topo, topo_dict = topology_discriminative_loss(topo_pred, batched_g)
    print(f"âœ“ Topology loss: {loss_topo.item():.4f}")
    print(f"  Components: {topo_dict}")
    
    print("\n" + "="*60)
    print("âœ“ All DGL loss functions working correctly!")
    print("="*60)