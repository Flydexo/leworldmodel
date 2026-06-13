from model import WorldModel
from cem import CEM
import torch
import torchvision.transforms as T
import gymnasium as gym
import gym_pusht
import stable_worldmodel as swm
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
import numpy as np
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = WorldModel(
    hidden_dim=192, 
    patch_size=12,
    channels=3, 
    enc_nb_heads=3, 
    enc_nb_layers=12, 
    height=96, 
    width=96, 
    pred_nb_layers=6, 
    pred_p_dropout=0.1, 
    pred_nb_heads=16, 
    action_dim=10
).to(device)
model.load_state_dict(torch.load('./model-10.pt'))

# ==========================================
# 1. Image Preprocessing Pipeline
# ==========================================
# Converts the live Gym PyGame screen (H, W, C) into the (1, 1, C, H, W) tensor the ViT expects
transform = T.Compose([
    T.ToTensor(), 
    T.Resize((96, 96), antialias=True)
])

def process_image(img_array):
    # img_array: HxWxC uint8 from env.render(), values 0-255
    t = torch.from_numpy(np.ascontiguousarray(img_array)).to(device, dtype=torch.float32)
    t = t.permute(2, 0, 1)                      # C,H,W  (NO /255 — matches training)
    t = F.interpolate(t.unsqueeze(0), size=(96, 96),
                      mode="bilinear", antialias=True, align_corners=False)
    return t.unsqueeze(0)       

# ==========================================
# 2. Acquire the Goal State
# ==========================================
print("Loading expert dataset to extract goal state...")
dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=6,
    frameskip=5,
    keys_to_load=['pixels', 'action']
)
# Grab one trajectory and extract the final expert frame
sample_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=True)))
goal_image_tensor = sample_batch['pixels'][0, -1].to(device, dtype=torch.float32)

with torch.no_grad():
    model.eval()
    # goal_latent shape: (1, 1, 192)
    goal_latent = model.encode_frames(goal_image_tensor.unsqueeze(0).unsqueeze(0))
print("Goal Latent State Acquired!")

# Added to manually render the window

# ==========================================
# 3. Setup the Live Environment
# ==========================================
# FIX: Use rgb_array so the env correctly returns the image matrix
# We can drop obs_type="pixels" and just use env.render() to get the image
env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

obs, info = env.reset()
done = False
step_count = 0

print("Starting Live CEM Planning Loop...")

# ==========================================
# 4. The Model Predictive Control (MPC) Loop
# ==========================================
# Setup an OpenCV window to replace the broken PyGame one
cv2.namedWindow("LeWorldModel - PushT", cv2.WINDOW_NORMAL)

while not done:
    # A. Get the image from the render engine (works perfectly in rgb_array mode)
    current_image = env.render() 
    
    # Render it live to your screen! (Gym returns RGB, OpenCV expects BGR)
    cv2.imshow("LeWorldModel - PushT", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1) 
    
    # Encode the live frame
    start_tensor = process_image(current_image) 
    
    with torch.no_grad():
        start_latent = model.encode_frames(start_tensor)

    b = next(iter(DataLoader(dataset, batch_size=1)))
    print("train pixel range:", b['pixels'].min().item(), b['pixels'].max().item())

    with torch.no_grad():
        cand = torch.empty(300, 5, 10, device=device).uniform_(0, 512)
        dest = model.rollout(start_latent.view(1,1,-1), cand, 5)   # (300, D)
        goalf = goal_latent.view(1, -1)
        costs = torch.linalg.vector_norm(goalf - dest, dim=1)
        print("cost min/mean/max/std:", costs.min().item(), costs.mean().item(),
              costs.max().item(), costs.std().item())
        # also: how much does the LATENT move as a function of action?
        print("dest spread across candidates:", dest.std(0).mean().item())
        print("action range in data:", b['action'].min().item(), b['action'].max().item())
        print("adaLN cond weight magnitude:", model.predictor.condition.weight.abs().mean().item())
    
    # B. Plan the Future using CEM
    best_macro_sequence = CEM(
        model=model, H=10, N=300, K=30, T=30, action_dim=10,
        start=start_latent.view(1,1,-1), goal=goal_latent.view(1,1,-1),
        device=device,
        a_low=-1.0, a_high=1.0,
    )
    
    # C. Execute the chunk of actions
    action_chunk_norm = best_macro_sequence[0].view(5, 2).cpu().numpy()   # in [-1,1]
    action_chunk = (action_chunk_norm + 1) / 2 * 512   
    
    for action in action_chunk:
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # Keep the visualizer updated during the action chunk execution
        chunk_image = env.render()
        cv2.imshow("LeWorldModel - PushT", cv2.cvtColor(chunk_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        if terminated or truncated:
            done = True
            break

    print(f"Executed chunk. Total physics steps: {step_count}")

print("Episode Finished!")
env.close()
cv2.destroyAllWindows()