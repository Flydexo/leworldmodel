import torch
from torch.distributions.normal import Normal
from torch.linalg import vector_norm

def CEM(model, H, N, K, T, action_dim, start, goal, device, a_low, a_high, init_std_frac=0.5, min_std=1e-2, alpha=0.1):
    mean = torch.ones(H, action_dim, device=device)
    std = torch.ones(H, action_dim, device=device)
    
    goal_features = goal.reshape(1, -1)
    best_action_sequence, min_cost = None, float('inf')

    for t in range(T):
        dist = Normal(mean, std)
        candidates = dist.sample((N,))                          # (N, H, action_dim)
        candidates = candidates.clamp(a_low, a_high)   # keep in physical range

        with torch.no_grad():
            height,W,C = start.shape
            start_shaped = start.reshape(1, 1, 1, height, W, C)
            goal_shaped = goal.reshape(1, 1, 1, height, W, C)
            costs = model.get_cost({'pixels': start_shaped, 'goal': goal_shaped}, candidates.unsqueeze(0)).view(-1)

        #with torch.no_grad():
        #    destinations = model.rollout(model.encode_frames(start.reshape(1,1,H,W,C).transpose(-1, -3).transpose(-2, -1)), candidates, H)  # (N, D)
        #costs_2 = vector_norm(goal_features.expand(N, -1) - destinations, dim=1)

        #print(costs, costs_2)
        #assert(costs == costs_2)

        elite_costs, elite = torch.topk(costs, k=K, largest=False)
        elite_cand = candidates[elite]   # (K, H, action_dim)

        new_mean = elite_cand.mean(dim=0)
        new_std  = elite_cand.std(dim=0).clamp_min(min_std)     # variance floor

        mean = (1 - alpha) * new_mean + alpha * mean
        std  = (1 - alpha) * new_std  + alpha * std
        
        if elite_costs[0] < min_cost:
            best_action_sequence = elite_cand[0].clone()        # (H, action_dim)
            min_cost = elite_costs[0].item()

    return best_action_sequence