import torch
from torch.distributions.normal import Normal
from torch.linalg import vector_norm

def CEM(model, H, N, K, T, action_dim, start, goal, device):
    mean = torch.ones(H, action_dim, device=device) * 256.0
    std = torch.ones(H, action_dim, device=device) * 256.0
    best_action_sequence, min_cost = None, float('inf')
    goal_features = goal.view(1, -1) 
    for t in range(T):
        distribution = Normal(mean, std)
        candidates = distribution.sample((N,)).to(device)
        destinations = model.rollout(start, candidates, H)
        costs = vector_norm((goal_features.expand(N, -1) - destinations), dim=1)
        elite_costs, elite = torch.topk(costs, k=K, largest=False)
        mean = candidates[elite].mean(dim=0)
        std = candidates[elite].std(dim=0)
        if elite_costs[0] < min_cost:
            best_action_sequence = candidates[elite[0]]
            min_cost = elite_costs[0]
            
    return best_action_sequence