import os
import random
import argparse
import torch
import torch.nn.functional as F
from utils import create_save_folder
    
def main():
    parser = argparse.ArgumentParser(description="PyTorch noisy labels PLS-LSA")
    parser.add_argument('--net', type=str, default='preresnet18',
                        choices=['resnet50', 'preresnet18', "inception", "clip-ViT-B-32", "clip-RN50"],
                        help='net name (default: preresnet18)')
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'web-bird', 'web-car', 'web-aircraft', 'webvision'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=2, help='Consistency regularization temperature for semi-supervised imputation')
    parser.add_argument('--mu', type=float, default=0.2, help='Contrastive temperature')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--proj-size', type=int, default=128)
    parser.add_argument('--conv', default="3", type=int)
    
    #CNWL
    parser.add_argument('--noise-ratio', default="0.0", type=str)
    
    #Abla
    parser.add_argument('--thresh', default=.95, type=float)
    parser.add_argument('--cont', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--pretrained2', default=None, type=str)
    parser.add_argument('--l', default=1, type=int)#Depth
    parser.add_argument('--alternate-ep', default=["linear", "loss"], nargs='+', choices=["linear", "loss", "rrl", "nocorr", "oracle"], help="Order in which to perform the alternating")
    parser.add_argument('--pls', default=False, action='store_true')
    parser.add_argument('--pls-penalty', default=False, action='store_true')
    
    #Co-training options    
    parser.add_argument('--co-train', default=False, action='store_true') #Ours
    parser.add_argument('--real-co-train', default=False, action='store_true') #DivideMix
    parser.add_argument('--indep-co-train', default=False, action='store_true')
    parser.add_argument('--vote-co-train', default=False, action='store_true')

    
    parser.add_argument('--trusted', default='loss', choices=['loss', 'rrl', 'trusted1k', 'trusted100', 'trusted10k'], help='Metric refined by the linear separation')
    
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if (args.pls or args.co_train or args.real_co_train or args.indep_co_train or args.vote_co_train):
        #Enforcing the contrastive learning objective
        args.cont = True
    
    seeds = {'1': round(torch.exp(torch.ones(1)).item()*1e3), '2': round(torch.acos(torch.zeros(1)).item() * 2), '3':round(torch.sqrt(torch.tensor(2.)).item()*1e6)}
    if args.seed in seeds.keys():        
        torch.manual_seed(seeds[str(args.seed)])
        torch.cuda.manual_seed_all(seeds[str(args.seed)])  # GPU seed
        random.seed(seeds[str(args.seed)])  # python seed for image transformation                                                    
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
        random.seed(args.seed)
               
    create_save_folder(args)
    args.checkname = args.net + '_' + args.dataset
    args.save_dir = os.path.join(args.save_dir, args.checkname, args.exp_name, str(args.seed))
    torch.save(args, os.path.join(args.save_dir, 'args'))

    if args.pls:
        from trainers import Trainer_PLS_improved as Trainer
    elif args.real_co_train:
        from trainers import Trainer_cotrain_DM as Trainer
    elif args.vote_co_train:
        from trainers import Trainer_cotrain_vote as Trainer
    elif args.indep_co_train:
        from trainers import Trainer_cotrain_indep as Trainer
    elif args.co_train:
        from trainers import Trainer_cotrain_ours as Trainer
    else:
        #Base trainer
        from trainers import Trainer

    print(Trainer)
    
    _trainer = Trainer(args)         
    #One hot labels for all
    relabel = torch.tensor(_trainer.train_loader.dataset.targets)
    relabel = F.one_hot(relabel, num_classes=args.num_class).float()
    
    _trainer.train_loader.dataset.targets = relabel
    _trainer.track_loader.dataset.targets = relabel
    _trainer.guessed_labels_soft = relabel.clone()
    
    _trainer.run(args.epochs)
            
if __name__ == "__main__":
    main()
   

    
