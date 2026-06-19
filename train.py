import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import stable_worldmodel as swm
from sigreg import SIGReg 
from torch.distributions.normal import Normal
from torch.linalg import vector_norm
from model import WorldModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from stable_worldmodel.wm.utils import save_pretrained
from omegaconf import OmegaConf
import trackio

def data_loader(cfg: DictConfig, name):
    dataset = swm.data.load_dataset(
        name,
        num_steps=cfg.num_steps+1,
        frameskip=1,
        keys_to_load=['pixels', 'action', 'state'],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )
    return dataset, loader

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    _, loader = data_loader(cfg, cfg.train_dataset)
    _, test_loader = data_loader(cfg, cfg.test_dataset)
    model = WorldModel(
        hidden_dim=cfg.model.hidden_dim, 
        patch_size=cfg.model.patch_size,
        channels=cfg.model.channels, 
        enc_nb_heads=cfg.model.enc_nb_heads, 
        enc_nb_layers=cfg.model.enc_nb_layers, 
        height=cfg.model.height, 
        width=cfg.model.width, 
        pred_nb_layers=cfg.model.pred_nb_layers, 
        pred_p_dropout=cfg.model.pred_p_dropout, 
        pred_nb_heads=cfg.model.pred_nb_heads, 
        action_dim=cfg.model.action_dim
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    trackio.init(
        project=cfg.trackio.project,
        name=cfg.trackio.name,
        config={"epochs": cfg.epochs, "learning_rate": cfg.learning_rate, "batch_size": cfg.batch_size, "lambda": cfg.lmbd},
        server_url=cfg.trackio.server_url
    )
    for epoch in range(cfg.epochs):
        val_loss = torch.ones(1)
        val_sigregloss = torch.ones(1)
        val_mseloss = torch.ones(1)
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            frames = batch['pixels'].to(cfg.device).type(torch.float32)
            actions = torch.nan_to_num(batch['action'], 0.0).to(cfg.device).type(torch.float32)
            pred, embeds = model(frames, actions)
            t_sigregloss = SIGReg(embeds.reshape(-1, embeds.shape[-1]))
            t_mseloss = F.mse_loss(pred[:,:-1,:], embeds[:,1:,:])
            t_loss = t_mseloss + cfg.lmbd*t_sigregloss
            t_loss.backward()
            optimizer.step()
            trackio.log({
                "epoch": epoch,
                "train_batch": i,
                "train_mse_loss": t_mseloss.item(),
                "train_sigreg_loss": t_sigregloss.item(),
                "train_loss": t_loss.item(),
                "val_mse_loss": val_mseloss.item(),
                "val_sigreg_loss": val_sigregloss.item(),
                "val_loss": val_loss.item(),
            })
        with torch.no_grad():    
            model.eval()
            for i, batch in enumerate(test_loader):
                frames = batch['pixels'].to(cfg.device).type(torch.float32)
                actions = torch.nan_to_num(batch['action'], 0.0).to(cfg.device).type(torch.float32)
                pred, embeds = model(frames, actions)
                v_sigregloss = SIGReg(embeds.reshape(-1, embeds.shape[-1]))
                v_mseloss = F.mse_loss(pred[:,:-1,:], embeds[:,1:,:])
                val_sigregloss = v_sigregloss
                val_mseloss = v_mseloss
                val_loss = v_mseloss + cfg.lmbd*v_sigregloss
                trackio.log({
                    "epoch": epoch,
                    "val_batch": i,
                    "val_mse_loss": val_mseloss.item(),
                    "val_sigreg_loss": val_sigregloss.item(),
                    "val_loss": val_loss.item(),
                    "train_mse_loss": t_mseloss.item(),
                    "train_sigreg_loss": t_sigregloss.item(),
                    "train_loss": t_loss.item(),
                })
            model.train()
        scheduler.step()
    trackio.finish()
    torch.save(model.state_dict(), f'{cfg.trackio.project}-{cfg.trackio.name}.pt')
    cfg.model.target = f'{cfg.trackio.project}.models.LeWorldModel'
    save_pretrained(
        model=model,
        run_name=cfg.trackio.project,
        config=cfg.model,
        filename=f'{cfg.trackio.name}.pt',
    )


if __name__ == "__main__":
    main()