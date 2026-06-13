#import torch
#from torch.distributions.normal import Normal
#from torch.linalg import vector_norm

#def CEM(model, H, N, K, T, action_dim, start, goal, device):
#    mean = torch.ones(H, action_dim, device=device) * 256.0
#    std = torch.ones(H, action_dim, device=device) * 256.0
#    best_action_sequence, min_cost = None, float('inf')
#    goal_features = goal.view(1, -1) 
#    for t in range(T):
#        distribution = Normal(mean, std)
#        candidates = distribution.sample((N,)).to(device)
#        destinations = model.rollout(start, candidates, H)
#        costs = vector_norm((goal_features.expand(N, -1) - destinations), dim=1)
#        elite_costs, elite = torch.topk(costs, k=K, largest=False)
#        mean = candidates[elite].mean(dim=0)
#        std = candidates[elite].std(dim=0)
#        if elite_costs[0] < min_cost:
#            best_action_sequence = candidates[elite[0]]
#            min_cost = elite_costs[0]
#            
#    return best_action_sequence

import torch
from torch.distributions.normal import Normal
from torch.linalg import vector_norm

def CEM(model, H, N, K, T, action_dim, start, goal, device,
        a_low, a_high, init_std_frac=0.5, min_std=1e-2, alpha=0.1):
    """
    a_low, a_high: tensors broadcastable to (H, action_dim) giving the valid action range.
    alpha: smoothing for mean/std refit (momentum) — 0 = no smoothing.
    """
    a_low  = torch.as_tensor(a_low,  device=device, dtype=torch.float32)
    a_high = torch.as_tensor(a_high, device=device, dtype=torch.float32)

    mean = ((a_low + a_high) / 2).expand(H, action_dim).clone()
    std  = ((a_high - a_low) * init_std_frac).expand(H, action_dim).clone()

    goal_features = goal.view(1, -1)
    best_action_sequence, min_cost = None, float('inf')

    for t in range(T):
        dist = Normal(mean, std)
        candidates = dist.sample((N,))                          # (N, H, action_dim)
        candidates = torch.clamp(candidates, a_low, a_high)     # keep in physical range

        with torch.no_grad():
            destinations = model.rollout(start, candidates, H)  # (N, D)
        costs = vector_norm(goal_features.expand(N, -1) - destinations, dim=1)

        elite_costs, elite = torch.topk(costs, k=K, largest=False)
        elite_cand = candidates[elite]                          # (K, H, action_dim)

        new_mean = elite_cand.mean(dim=0)
        new_std  = elite_cand.std(dim=0).clamp_min(min_std)     # variance floor

        mean = (1 - alpha) * new_mean + alpha * mean
        std  = (1 - alpha) * new_std  + alpha * std

        if elite_costs[0] < min_cost:
            best_action_sequence = elite_cand[0].clone()        # (H, action_dim)
            min_cost = elite_costs[0].item()

    return best_action_sequence