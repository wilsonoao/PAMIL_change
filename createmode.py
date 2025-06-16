 


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



from PAMIL_multi_reward_model.models.basemodel import BasedMILTransformer as BaseMILT
from PAMIL_multi_reward_model.models.DPSF import PPO,Memory
from PAMIL_multi_reward_model.models.classmodel import ClassMultiMILTransformer as ClassMMILT
from PAMIL_multi_reward_model.models.SFFR import FusionHistoryFeatures
from PAMIL_multi_reward_model.utilmodule.utils import make_parse
import torch



def create_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    basedmodel = BaseMILT(args).to(device)
    ppo = PPO(args.feature_dim,args.state_dim, args.policy_hidden_dim, args.policy_conv,
                        device=device,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.action_size
                        )
    FusionHisF = FusionHistoryFeatures(args.feature_dim,args.state_dim).to(device) #feature_dim, state_dim

    classifymodel = ClassMMILT(args).to(device)
    memory = Memory()
    
    assert basedmodel is not None, "creating model failed. "
    print(f"basedmodel Total params: {sum(p.numel() for p in basedmodel.parameters()) / 1e6:.2f}M")
    print(f"classifymodel Total params: {sum(p.numel() for p in classifymodel.parameters()) / 1e6:.2f}M")
    
    return basedmodel,ppo,classifymodel,memory ,FusionHisF

if __name__ == "__mian__":
    
    args = make_parse()
    create_model(args)
