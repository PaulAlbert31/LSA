import os
import copy
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from utils import make_data_loader, multi_class_loss, mixup_data, min_max, cosine_lr, adjust_lr, get_dataset_hyper
from wrappers import CLIP_wrapper, ResNet_wrapper, Inception_wrapper

from lightning_fabric import Fabric

#Import Faiss for the RRL metric
#import faiss

class Trainer(nn.Module):
    def __init__(self, args, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        
        self.args = args
        self.epoch = 0        
               
        self.kwargs = {'num_workers': 12, 'pin_memory': False} #False to avoid RuntimeError: Pin memory thread exited unexpectedly ?
        
        self.train_loader, self.val_loader, self.track_loader, self.clean_noisy = make_data_loader(args, **self.kwargs)
        self.trainset = self.train_loader.dataset
        self.args.num_class = self.train_loader.dataset.num_class

        self.accelerator = Fabric(strategy='ddp', precision='16-mixed') #Change here to train on multiple GPUs
        self.accelerator.launch()
        self.broadcast = lambda x: self.accelerator.broadcast(x)
        self.gather = lambda x: self.accelerator.all_gather(x)

        self.model = self.get_model()
        if args.wd is not None:
            wd = self.args.wd
        else:
            wd = 5e-4
            if self.args.net == 'resnet50':
                wd = 1e-3
                
        self.wd = wd

        if "clip" in args.net:
            self.optimizer = torch.optim.AdamW([{'params': self.model.model.parameters()},  {'params': [self.model.linear.weight, self.model.linear.bias, self.model.contrastive.weight, self.model.contrastive.bias], 'lr': 0.1}], lr=args.lr, weight_decay=0.1)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.wd)
        
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        
        self.cosine_lr_per_epoch = cosine_lr(self.args.lr, 0, self.args.epochs)
   
        self.best = 0
        self.best_epoch = 0

        #Init
        self.is_clean = torch.ones(len(self.train_loader.dataset), dtype=torch.bool)
        self.is_noisy = torch.zeros(len(self.train_loader.dataset), dtype=torch.bool)
        
        self.weights = torch.ones(len(self.train_loader.dataset))
        self.guessed_labels_soft = torch.zeros((len(self.train_loader.dataset), self.args.num_class))
        
        self.subset_in, self.subset_out = None, None

        self.preds_ = torch.zeros((self.args.epochs, len(self.train_loader.dataset), 5)).bool()
        self.probas_ = torch.zeros((self.args.epochs, len(self.train_loader.dataset), 6))

        mean, std, _, _ = get_dataset_hyper(args.dataset)

        #Used to speed up evaluation
        self.transform_test = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std)
        )        

    def run(self, epochs):
        self.model, self.optimizer = self.accelerator.setup(self.model, self.optimizer)
        self.train_loader, self.track_loader, self.val_loader = self.accelerator.setup_dataloaders(self.train_loader, self.track_loader, self.val_loader)
                    
        start_ep = 0
        if self.args.resume:
            start_ep = self.resume()
            self.model.eval()
            self.val(start_ep)
            
        for eps in range(start_ep, epochs):
            self.model.train()
            self.train(eps)
            self.model.eval()
            if self.args.alternate_ep is not None:
                which = self.args.alternate_ep[eps%len(self.args.alternate_ep)]
            else:
                which = "loss"                    
                
            self.val(eps)

    def resume(self):
        load_dict = torch.load(self.args.resume, map_location='cpu')
        self.model.module.load_state_dict(load_dict['state_dict'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        
        start_ep = load_dict['epoch']
        self.epoch = start_ep
        self.guessed_labels_soft = load_dict['guessed_labels_soft']
        self.weights = load_dict['weights_1']
        self.is_clean = load_dict['is_clean']
        self.is_noisy = load_dict['is_noisy']
        self.best = load_dict['best']
        self.best_epoch = load_dict['best_epoch']
        
        del load_dict        
        return start_ep

    def get_model(self):        
        if self.args.net == 'preresnet18':
            from nets.preresnet import PreActResNet18
            model = PreActResNet18(self.args.num_class)
            if self.args.pretrained is not None:
                dic = torch.load(self.args.pretrained)["state_dict"]
                #Converting from solo-learn to our code base
                dic = {k.replace('backbone.','').replace('downsample','shortcut'): v for k, v in dic.items()}
                model.load_state_dict(dic, strict=False)
            model = ResNet_wrapper(model, self.args.num_class, self.args.proj_size, dim=512)
        elif self.args.net == 'resnet50':
            if self.args.pretrained is None:
                model = torchvision.models.resnet50()
            elif self.args.pretrained == "imagenet":
                model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
            elif self.args.pretrained != 'imagenet':
                model = torchvision.models.resnet50()
                dic = torch.load(self.args.pretrained, map_location=torch.device('cpu'))["state_dict"]                    
                dic = {k.replace('backbone.','').replace('downsample','shortcut').replace('projector', 'contrastive'): v for k,v in dic.items()}
                model.load_state_dict(dic, strict=False)         
                
            model = ResNet_wrapper(model, self.args.num_class, self.args.proj_size, dim=2048)
        elif self.args.net == "inception":
            from nets.inceptionresnetv2 import InceptionResNetV2
            model = InceptionResNetV2(self.args.num_class)
            if self.args.pretrained is not None:
                dic = torch.load(self.args.pretrained, map_location=torch.device('cpu'))["state_dict"]
                dic = {k.replace('backbone.','').replace('downsample','shortcut').replace('projector', 'contrastive'): v for k, v in dict.items()}
                model.load_state_dict(dic, strict=False)
            model = Inception_wrapper(model, self.args.num_class, self.args.proj_size, dim=1536)
            
        elif self.args.net == "clip-RN50":
            import open_clip
            pretrained = "openai"
            (
                model,
                _,
                _,
            ) = open_clip.create_model_and_transforms(
                "RN50", pretrained=pretrained, cache_dir="./clip_weights"
            )
            
            model = CLIP_wrapper(model, self.args.num_class, self.args.proj_size, dim=1024)
            
        elif self.args.net == "clip-ViT-B-32":
            import open_clip
            pretrained = "openai"
            (
                model,
                _,
                _,
            ) = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=pretrained, cache_dir="./clip_weights"
            )
            
            model = CLIP_wrapper(model, self.args.num_class, self.args.proj_size, dim=512)
        else:
            raise NotImplementedError("Network {} is not implemented".format(self.args.net))
        
        print('Number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model
        
    def get_track_outputs(self, image, **kwargs):
        return self.model(image, **kwargs)
    
    def get_val_outputs(self, image, **kwargs):
        return self.model(image, **kwargs)

    def parse_features(self, features, features_interm):
        features_interm = F.normalize(features_interm, p=2)
        features = F.normalize(features, p=2)        
        return features, features_interm

    def get_train_image(self, sample):
        image = sample['image']
        is_noise = self.is_noisy[sample['index'].cpu()]
        image[is_noise] = sample['image1'][is_noise] #Hard augs for noisy images
        return image
        
    def compute_losses(self, sample, epoch):
        im, im_, target = sample['image'], sample['image_'], sample['target']
        ids = sample['index'].to(self.is_clean.device)
        weights = torch.ones(len(im))
        if epoch > self.args.warmup:
            weights[self.is_noisy[ids]] = 0
        
        im = self.get_train_image(sample)

        if self.args.mixup:
            image, la, lb, lam, o = mixup_data(im, target)
        else:
            image = im
        weights = weights.to(im)
        outputs = self.model(image)
        
        if epoch <= self.args.warmup:
            if self.args.mixup:
                loss_c = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
            else:
                loss_c = multi_class_loss(outputs, target)                        
        else:
            if self.args.mixup:
                loss_c = lam * (multi_class_loss(outputs, la) * weights).sum() / weights.sum() + (1-lam) * (multi_class_loss(outputs, lb) * weights[o]).sum() / weights[o].sum()                
            else:
                loss_c = (multi_class_loss(outputs, target) * weights).sum() / weights.sum()
        
        return loss_c

    
    def train(self, epoch):
        adjust_lr(self.optimizer, self.cosine_lr_per_epoch[epoch])
            
        acc = 0
        tbar = tqdm(self.train_loader)        
        self.epoch = epoch
        total_sum = 0        
        
        for i, sample in enumerate(tbar):
            loss = self.compute_losses(sample, epoch)
            loss = loss.mean()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if i % 20 == 0:
                tbar.set_description('Training loss {0:.2f}, LR {1:.6f}'.format(loss, self.optimizer.param_groups[0]['lr']))

        print('[Epoch: {}, numImages: {}, numClasses: {}, LR: {:.6f}]'.format(epoch, total_sum, self.args.num_class, self.optimizer.param_groups[0]['lr']))
        return
                    
    def val(self, epoch):
        self.model.eval()
        acc, acc_ens = 0, 0
        vbar = tqdm(self.val_loader)
        vbar.set_description("Validation")
        total = 0
        losses = torch.tensor([])
        trainLabels = torch.LongTensor(torch.argmax(self.track_loader.dataset.targets, dim=1))
        acc_m = 0
        #Computing linear val acc & kNN acc
        top1, top5 = 0, 0
        
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                sample['image'] = self.transform_test(sample['image'])
                
                image, target, ids = sample['image'], sample['target'], sample['index']
                    
                outputs, feat_cont = self.get_val_outputs(image, return_features=True)
                                
                preds = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)                                    

                acc += torch.sum(preds == target.data)
                total += preds.size(0)
                                    
        acc, total = self.gather((acc, total))
        if self.accelerator.global_rank == 0:
            acc, total = acc.sum(), total.sum()
            final_acc = float(acc)/total
            final_acc = round(final_acc.item()*100, 2)
                    
            if final_acc > self.best:
                self.best = final_acc
                self.best_epoch = epoch
                self.save_model(epoch, best=True)                             

            print(f'Validation Accuracy: {final_acc:.2f}, kNN {float(top1)/total:.2f}, best accuracy {self.best:.2f} at epoch {self.best_epoch}')        
        return 

    def save_model(self, epoch, t=False, best=False):
        if t:
            checkname = os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch))
        elif best:
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                f.write(str(self.best))
        else:
            checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'guessed_labels_soft': self.guessed_labels_soft,
            'weights_1': self.weights,
            'is_clean': self.is_clean,
            'is_noisy': self.is_noisy,
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)
        
    @torch.no_grad()
    def track_loss(self):
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description('Tracking loss')
               
            features_interm = []
            features_cont = []
            indexes = []
            losses = torch.zeros(len(self.track_loader.dataset))
                            
            for i, sample in enumerate(tbar):
                sample['image'] = self.transform_test(sample['image'])
                
                image, target, ids = sample['image'], sample['target'], sample['index']
                target, image = target, image
                
                outputs, feats_cont, feats_interm = self.get_track_outputs(image, return_features=True, return_interm=self.args.l)
                
                outputs = outputs.float()
                self.guessed_labels_soft[ids] = F.softmax(outputs, dim=1).cpu()
                #Track features
                features_cont.append(feats_cont.detach().cpu().float())
                features_interm.append(feats_interm.detach().cpu().float())
                    
                losses[ids] = self.criterion_nored(outputs, torch.argmax(target, dim=-1)).detach().cpu().float()
                indexes.append(ids.cpu())

            features_cont, features_interm, indexes = torch.cat(features_cont), torch.cat(features_interm), torch.cat(indexes)
            features_cont, features_interm = self.parse_features(features_cont, features_interm)
            asg = torch.argsort(indexes)
            features_cont, features_interm = features_cont[asg], features_interm[asg]
            
            scores = torch.zeros(len(features_cont))
            
            if self.which=='rrl':#RRL detect, quite compute intensive
                # initalize knn search
                if self.epoch == self.args.warmup:
                    self.soft_labels = self.guessed_labels_soft.clone()
                    
                labels = torch.argmax(self.train_loader.dataset.targets, dim=-1)
                features = features_cont.numpy()
                N = features.shape[0]      
                k = 200
                temperature = 0.3
                faiss.omp_set_num_threads(24)
                index = faiss.IndexFlatIP(features.shape[1])
                index.parallel_mode = 1
                
                index.add(features)  
                D,I = index.search(features,k+1)  
                neighbors = torch.LongTensor(I) #find k nearest neighbors excluding itself
                
                score = torch.zeros(N,self.args.num_class) #holds the score from weighted-knn
                weights = torch.exp(torch.Tensor(D[:,1:])/temperature)  #weight is calculated by embeddings' similarity
                for n in range(N):           
                    neighbor_labels = self.soft_labels[neighbors[n,1:]]
                    score[n] = (neighbor_labels*weights[n].unsqueeze(-1)).sum(0)  #aggregate soft labels from neighbors
                self.soft_labels = (score/score.sum(1).unsqueeze(-1) + self.guessed_labels_soft)/2  #combine with model's prediction as the new soft labels
                scores = 1-self.soft_labels[labels>=0, labels].float() #(self.soft_labels[labels>=0, labels] < 0.02).float()
                            
        return losses, features_interm, scores
        
    def get_trusted(self, met, thresh=.5):
        met = min_max(met).cpu()
        self.subset_out = torch.arange(len(met))[met > thresh]
        self.subset_in = torch.arange(len(met))[met <= thresh]       
            
    def detect_noise(self, which, correct=True, losses=None, features_interm=None, scores=None):
        if losses is None:
            losses, features_interm, scores = self.track_loss()

        if not correct:
            return losses, features_interm, scores
        
        losses, features_interm, scores = self.gather((losses, features_interm, scores))
        preds = torch.ones(len(losses))
        if self.accelerator.global_rank == 0:
            losses, features_interm, scores = losses.sum(dim=0), features_interm.sum(dim=0), scores.sum(dim=0)
            losses = losses.cpu()                        
            if self.args.trusted == 'loss':
                interest = min_max(losses)
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                gmm = gmm.fit(interest.reshape(-1,1))
                
                proba = gmm.predict_proba(interest.reshape(-1,1))[:, 0] #Probability to belong to the high loss mode (noisy)
                if gmm.means_[0] < gmm.means_[1]:
                    proba = 1-proba
                preds = torch.from_numpy(proba > self.args.thresh).float()
                self.get_trusted(preds)                
            elif self.args.trusted == 'rrl':
                self.get_trusted((scores>0.98).float()) #Threshold from the original paper
            elif self.args.trusted == 'trusted100' and self.subset_in is None:
                random_select = torch.randperm(len(losses))[:100]
                self.subset_in = random_select[~self.clean_noisy[random_select]]
                self.subset_out = random_select[self.clean_noisy[random_select]]
            elif self.args.trusted == 'trusted1k' and self.subset_in is None:
                random_select = torch.randperm(len(losses))[:1000]
                self.subset_in = random_select[~self.clean_noisy[random_select]]
                self.subset_out = random_select[self.clean_noisy[random_select]]
            elif self.args.trusted == 'trusted10k' and self.subset_in is None:
                random_select = torch.randperm(len(losses))[:10000]
                self.subset_in = random_select[~self.clean_noisy[random_select]]
                self.subset_out = random_select[self.clean_noisy[random_select]]
                
            if which == "loss":
                interest = min_max(losses).cpu()
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                gmm = gmm.fit(interest.reshape(-1,1))
                
                proba = gmm.predict_proba(interest.reshape(-1,1))[:, 0] #Probability to belong to the high loss mode (noisy)
                if gmm.means_[0] < gmm.means_[1]:
                    proba = 1-proba
                preds = torch.from_numpy(proba > self.args.thresh).bool()
            
            elif which == "linear": #Linear separation
                scaler = StandardScaler()
                f = features_interm.cpu() 
                f = scaler.fit_transform(f)
                tot = len(self.subset_in) + len(self.subset_out)                
                ret = LogisticRegression(penalty=None, max_iter=1000, class_weight="balanced").fit(f[torch.cat((self.subset_in, self.subset_out)).view(-1)], torch.cat((torch.zeros(len(self.subset_in)), torch.ones(len(self.subset_out)))).view(-1))
                proba = ret.predict_proba(f)[:, 1]                
                s = ret.predict(f)
                preds = torch.from_numpy(s).bool()
                
            elif which == "rrl":
                proba = scores.cpu().numpy()
                preds = (scores > 0.98).cpu()
                
            elif which == "nocorr":
                proba = torch.zeros(len(losses))
                preds = torch.zeros(len(losses)).bool()
                                
            if self.train_loader.dataset.clean_noisy is not None: #Displaying AUC if clean/noisy labels are available
                fpr, tpr, thresholds = metrics.roc_curve(self.train_loader.dataset.clean_noisy, preds)
                auc = metrics.auc(fpr, tpr)
                print(f'Retreival OOD {which}', auc)                
               
            self.is_clean = ~preds
            self.is_noisy = preds
            
        self.is_clean = self.broadcast(self.is_clean)
        self.is_noisy = self.broadcast(self.is_noisy)
        return self.is_noisy
       
class Trainer_PLS(Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer_PLS, self).__init__(*args, **kwargs)
        
    def get_guessed_targets(self, im, im_):
        out1 = F.softmax(self.model(im), dim=1)
        out2 = F.softmax(self.model(im_), dim=1)
        
        #Label guessing for ID noisy samples
        guessed_targets = (out1 + out2) / 2
        
        return guessed_targets

    def get_aug_feats(self, image1, image2=None, **kwargs):
        if image2 is None:
            return self.model(image1, return_features=True, **kwargs)
        return self.model(image1, image2, return_features=True, **kwargs)

    def detect_noise(self, which, correct=True):
        losses, features_interm, pseudo_loss, scores = self.track_loss()

        preds = super().detect_noise(which, correct, losses, features_interm, scores)
                
        pseudo_loss = self.gather(pseudo_loss)
        if self.accelerator.global_rank == 0:            
            pseudo_loss = pseudo_loss.sum(dim=0)
            
            self.weights = torch.ones(len(self.weights))
            preds = self.is_noisy #Detected noisy samples
            
            if pseudo_loss.max() == float('Inf'):
                u = torch.unique(pseudo_loss)
                second_max = torch.topk(u, 2)[0][1] #Second top value
                pseudo_loss[pseudo_loss == float('Inf')] = second_max
      
            if preds.sum() > 2 and len(pseudo_loss[pseudo_loss != pseudo_loss]) == 0:
                interest = min_max(pseudo_loss[preds]).cpu() #Pseudo loss of noisy samples
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                gmm = gmm.fit(interest.reshape(-1,1))
                #Compute confidence in the pseudo-label
                proba_ = gmm.predict_proba(interest.reshape(-1,1))[:, 0]
                if gmm.means_[1] < gmm.means_[0]:
                    proba_ = 1-proba_
                w = torch.from_numpy(proba_).float() #w=1 means the pseudo-label can be 100% trusted                
        
                self.weights[preds] = w
                
        self.weights = self.broadcast(self.weights)       
            
        return preds
    
    def compute_losses(self, sample, epoch):        
        im, im_, target = sample['image'], sample['image_'], sample['target']
        ids = sample['index'].to(self.is_clean.device)
        if epoch >= self.args.warmup:
            batch_clean = self.is_clean[ids]
            batch_idn = self.is_noisy[ids]
            #SSL
            with torch.no_grad():
                guessed_targets_ = self.get_guessed_targets(im, im_)
                
            guessed_targets = guessed_targets_ ** (self.args.gamma) #temp sharp
            
            guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
            guessed_targets = guessed_targets.detach()

            self.guessed_labels_soft[ids] = guessed_targets.cpu() #track for the pseudo-loss

            if epoch > self.args.warmup:
                target[batch_idn] = guessed_targets[batch_idn]
                
            weights = self.weights[ids]

        target_cont = copy.deepcopy(target)
        im = self.get_train_image(sample)
        kwargs = {}
        if self.args.mixup:
            image, la, lb, lam, o = mixup_data(im, target)
        else:
            image = im

        penalty, loss_u = 0, 0
        if epoch <= self.args.warmup:
            outputs, feats1_cont = self.get_aug_feats(image, **kwargs)
            if self.args.mixup:
                loss_c = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
            else:
                loss_c = multi_class_loss(outputs, target)
                
            loss_f = multi_class_loss(feats1_cont, torch.randn(feats1_cont.shape).to(feats1_cont)) #0. * loss hack for DDP unused params
            loss_c = loss_c + loss_f * 0.
        else:
            image2 = sample["image2"]            
            if self.args.cont:
                (outputs, feats1_cont), (_, feats2_cont) = self.get_aug_feats(image, image2, **kwargs)                    
            else:
                outputs, feats1_cont = self.get_aug_feats(image, **kwargs)
                
            weights = weights.to(outputs.device)
            if self.args.mixup:
                loss_c = lam * (multi_class_loss(outputs, la) * weights).sum() / weights.sum() + (1-lam) * (multi_class_loss(outputs, lb) * weights[o]).sum() / weights[o].sum()
            else:
                loss_c = (weights * multi_class_loss(outputs, target)).sum()/weights.sum()
                
            #PLS class balanced regularization
            prior = torch.ones(self.args.num_class)/self.args.num_class
            prior = prior.to(outputs.device)        
            pred_mean = torch.softmax(outputs, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            if self.args.cont:
                feats1_cont_n, feats2_cont_n = F.normalize(feats1_cont, p=2), F.normalize(feats2_cont, p=2)                    
                logits = torch.matmul(feats1_cont_n, feats2_cont_n.t())
                
                labels = torch.argmax(target_cont, dim=1)
                labels = F.one_hot(labels, num_classes=self.args.num_class)
                labels1 = labels
                labels1 = labels1.mm(labels1.t())                                            
                    
                if self.args.mixup:
                    labels2 = labels1[o]
                
                logits = torch.div(logits, self.args.mu) #Contrastive temperature

                if self.args.mixup:
                    labels1, labels2 = labels1.to(logits.device), labels2.to(logits.device)
                    loss_u = (lam * multi_class_loss(logits, labels1) / labels1.sum(dim=-1) + (1-lam) * multi_class_loss(logits, labels2) / labels2.sum(dim=-1))                
                else:
                    loss_u = multi_class_loss(logits, labels) / labels.sum(dim=-1)
                loss_u = loss_u.mean()
            else:
                loss_f = multi_class_loss(feats1_cont, torch.randn(feats1_cont.shape).to(feats1_cont)) #0. * loss hack for DDP unused params
                loss_u = loss_f * 0.
                
        return loss_c + penalty + loss_u

    @torch.no_grad()
    def track_loss(self, which='loss'):
        self.model.eval()
        with torch.no_grad():
            tbar = tqdm(self.track_loader)
            tbar.set_description('Tracking loss')
                
            losses = torch.zeros(len(self.track_loader.dataset))
            features_interm = []
            features_cont = []
            indexes = []
            self.guessed_labels_soft = self.gather(self.guessed_labels_soft)
            self.guessed_labels_soft = self.guessed_labels_soft.sum(dim=0).cpu()
            pseudo_loss = torch.zeros(len(self.track_loader.dataset))
            
            for i, sample in enumerate(tbar):
                sample['image'] = self.transform_test(sample['image'])

                image, target, ids = sample['image'], sample['target'], sample['index']   
                
                outputs, feats_cont, feats_interm = self.get_track_outputs(image, return_features=True, return_interm=self.args.l)
                
                outputs = outputs.float()                       
                                   
                pseudo_guess = self.guessed_labels_soft[ids.to(self.guessed_labels_soft.device)]
                pseudo_loss[ids] = multi_class_loss(outputs, pseudo_guess.to(outputs.device)).cpu()
                features_cont.append(feats_cont.detach().cpu().float())
                features_interm.append(feats_interm.detach().cpu().float())
                    
                losses[ids] = self.criterion_nored(outputs, torch.argmax(target, dim=-1)).detach().cpu().float()
                indexes.append(ids.cpu())

            features_cont, features_interm, indexes = torch.cat(features_cont), torch.cat(features_interm), torch.cat(indexes)
            features_cont, features_interm = self.parse_features(features_cont, features_interm)
            asg = torch.argsort(indexes)
            features_cont, features_interm = features_cont[asg], features_interm[asg]
            
            scores = torch.zeros(len(features_cont))
            if which=='rrl':#RRL detect, quite compute intensive
                # initalize knn search
                if self.epoch == self.args.warmup:
                    self.soft_labels = self.guessed_labels_soft.clone()
                    
                labels = torch.argmax(self.train_loader.dataset.targets, dim=-1)
                features = features_cont.numpy()
                N = features.shape[0]      
                k = 200
                temperature = 0.3
                faiss.omp_set_num_threads(24)
                index = faiss.IndexFlatIP(features.shape[1])
                index.parallel_mode = 1
                
                index.add(features)  
                D,I = index.search(features,k+1)  
                neighbors = torch.LongTensor(I) #find k nearest neighbors excluding itself
                
                score = torch.zeros(N,self.args.num_class) #holds the score from weighted-knn
                weights = torch.exp(torch.Tensor(D[:,1:])/temperature)  #weight is calculated by embeddings' similarity
                for n in range(N):           
                    neighbor_labels = self.soft_labels[neighbors[n,1:]]
                    score[n] = (neighbor_labels*weights[n].unsqueeze(-1)).sum(0)  #aggregate soft labels from neighbors
                self.soft_labels = (score/score.sum(1).unsqueeze(-1) + self.guessed_labels_soft)/2  #combine with model's prediction as the new soft labels
                scores = 1-self.soft_labels[labels>=0, labels].float() #(self.soft_labels[labels>=0, labels] < 0.02).float()
            
        return losses, features_interm, pseudo_loss, scores

class Trainer_PLS_improved(Trainer_PLS):
    def __init__(self, *args, **kwargs):
        super(Trainer_PLS_improved, self).__init__(*args, **kwargs)

    def run(self, epochs):
        self.model, self.optimizer = self.accelerator.setup(self.model, self.optimizer)
        self.train_loader, self.track_loader, self.val_loader = self.accelerator.setup_dataloaders(self.train_loader, self.track_loader, self.val_loader)           
        self.which = ""
        start_ep = 0
        if self.args.resume:
            start_ep = self.resume()
            self.model.eval()
            self.val(start_ep)
            preds = self.detect_noise(which)
            
        for eps in range(start_ep, epochs):

            self.model.train()
            self.train(eps)
            self.model.eval()
            
            if eps >= self.args.warmup:
                if self.args.alternate_ep is not None:
                    which = self.args.alternate_ep[eps%len(self.args.alternate_ep)]
                else:
                    which = "loss"                    
                preds = self.detect_noise(which)                    
                self.which = which                                            
            self.val(eps)
        
    def get_guessed_targets(self, im, im_):        
        out2 = F.softmax(self.model(im_), dim=1)
        return out2

    def compute_losses(self, sample, epoch):        
        im, im_, target = sample['image'], sample['image_'], sample['target']
        ids = sample['index'].to(self.is_clean.device)

        if epoch >= self.args.warmup:
            batch_clean = self.is_clean[ids]
            batch_idn = self.is_noisy[ids]
            #SSL
            with torch.no_grad():
                guessed_targets_ = self.get_guessed_targets(im, im_)
                
            guessed_targets = guessed_targets_ ** (self.args.gamma) #temp sharp
            
            guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True) #normalization
            guessed_targets = guessed_targets.detach()

            self.guessed_labels_soft[ids] = guessed_targets.cpu() #track for the pseudo-loss
            
            if epoch > self.args.warmup:
                target[batch_idn] = guessed_targets[batch_idn]
                
            weights = self.weights[ids]

        target_cont = copy.deepcopy(target)

        im = self.get_train_image(sample)
        kwargs = {}
        if self.args.mixup:
            image, la, lb, lam, o = mixup_data(im, target)
        else:
            image = im

        penalty, loss_u = 0, 0
        if epoch <= self.args.warmup:
            outputs, feats1_cont = self.get_aug_feats(image, **kwargs)
                
            if self.args.mixup:
                loss_c = lam * multi_class_loss(outputs, la) + (1-lam) * multi_class_loss(outputs, lb)
            else:
                loss_c = multi_class_loss(outputs, target)
            loss_f = multi_class_loss(feats1_cont, torch.randn(feats1_cont.shape).to(feats1_cont)) #0. * loss hack for DDP unused params
            loss_c = loss_c + loss_f * 0.
        else:            
            image2 = sample["image2"] 
            if self.args.cont:
                (outputs, feats1_cont), (_, feats2_cont) = self.get_aug_feats(image, image2, **kwargs)
            else:
                outputs, feats1_cont = self.get_aug_feats(image, **kwargs)
                                
            weights = weights.to(outputs.device)
            if self.args.mixup:
                loss_c = lam * (multi_class_loss(outputs, la) * weights).sum() / weights.sum() + (1-lam) * (multi_class_loss(outputs, lb) * weights[o]).sum() / weights[o].sum()
            else:
                loss_c = (weights * multi_class_loss(outputs, target)).sum() / weights.sum()

            penalty = 0
            #PLS class balanced regularization
            if self.args.pls_penalty:
                prior = torch.ones(self.args.num_class)/self.args.num_class
                prior = prior.to(outputs.device)        
                pred_mean = torch.softmax(outputs, dim=1).mean(0)
                penalty += torch.sum(prior*torch.log(prior/pred_mean))                

            if self.args.cont:
                feats1_cont_n, feats2_cont_n = F.normalize(feats1_cont, p=2), F.normalize(feats2_cont, p=2)                    
                logits = torch.matmul(feats1_cont_n, feats2_cont_n.t())
                
                labels = torch.argmax(target_cont, dim=1)
                labels = F.one_hot(labels, num_classes=self.args.num_class)
                labels1 = labels.float()
                labels1 = labels1.mm(labels1.t())
                    
                if self.args.mixup:
                    labels2 = labels1[o]
                
                logits = torch.div(logits, self.args.mu) #Contrastive temperature

                if self.args.mixup:
                    labels1, labels2 = labels1.to(logits.device), labels2.to(logits.device)
                    loss_u = (lam * multi_class_loss(logits, labels1) / labels1.sum(dim=-1) + (1-lam) * multi_class_loss(logits, labels2) / labels2.sum(dim=-1))                
                else:
                    loss_u = multi_class_loss(logits, labels) / labels.sum(dim=-1)
                loss_u = loss_u.mean()
            else:
                loss_f = multi_class_loss(feats1_cont, torch.randn(feats1_cont.shape).to(feats1_cont)) #0. * loss hack for DDP unused params
                loss_u = loss_f * 0.
                
        return loss_c + penalty + loss_u
            
class Trainer_cotrain(Trainer_PLS_improved):
    def __init__(self, *args, **kwargs):
        super(Trainer_cotrain, self).__init__(*args, **kwargs)
        self.best = (0, 0, 0)
        self.best_indi = (0, 0)
        
        self.args.pretrained2 = self.args.pretrained
        
        self.model_2 = self.get_model()
        if "clip" in self.args.net:
            self.optimizer2 = torch.optim.AdamW([{'params': self.model_2.model.parameters()},  {'params': [self.model_2.linear.weight, self.model_2.linear.bias, self.model_2.contrastive.weight, self.model_2.contrastive.bias], 'lr': 0.1}], lr=self.args.lr, weight_decay=0.1)
        else:
            self.optimizer2 = torch.optim.SGD(self.model_2.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.wd)
            
        self.weights_other = torch.ones(len(self.train_loader.dataset))
        self.is_clean_other = torch.ones(len(self.train_loader.dataset), dtype=torch.bool)
        self.is_noisy_other = torch.zeros(len(self.train_loader.dataset), dtype=torch.bool)
        
    def resume(self):
        load_dict = torch.load(self.args.resume, map_location='cpu')
        
        self.model_1.module.load_state_dict(load_dict['state_dict'])        
        self.model_2.module.load_state_dict(load_dict['state_dict2'])
        
        self.optimizer1.load_state_dict(load_dict['optimizer1'])
        self.optimizer2.load_state_dict(load_dict['optimizer2'])
        
        start_ep = load_dict['epoch']
        self.epoch = start_ep
        self.guessed_labels_soft_1 = load_dict['guessed_labels_soft_1']
        self.guessed_labels_soft_2 = load_dict['guessed_labels_soft_2']
        self.weights = load_dict['weights_1']
        self.is_clean = load_dict['is_clean']
        self.is_noisy = load_dict['is_noisy']
        
        self.weights_other = load_dict['weights_2']
        self.is_clean_other = load_dict['is_clean_other']
        self.is_noisy_other = load_dict['is_noisy_other']

        self.best = load_dict['best']
        self.best_epoch = load_dict['best_epoch']
        
        del load_dict
                
        return start_ep
        
    def run(self, epochs):
        self.model, self.optimizer = self.accelerator.setup(self.model, self.optimizer)
        self.train_loader, self.track_loader, self.val_loader = self.accelerator.setup_dataloaders(self.train_loader, self.track_loader, self.val_loader)
        self.model_1 = self.model       
        self.optimizer1 = self.optimizer
        self.model_2, self.optimizer2 = self.accelerator.setup(self.model_2, self.optimizer2)
                            
        start_ep = 0
        if self.args.resume:
            start_ep = self.resume()
            self.model_1.eval()
            self.model_2.eval()
            self.val(start_ep)
            
        for eps in range(start_ep, epochs):
            self.model_1.train()
            self.model_2.train()
            
            self.train(eps)

            self.model_1.eval()
            self.model_2.eval()
            
            if eps >= self.args.warmup:                
                if self.args.alternate_ep is not None:
                    which = self.args.alternate_ep[eps%len(self.args.alternate_ep)]
                else:
                    which = "loss"
                    
                self.detect_noise(which)
                
            self.val(eps)            
            self.which = which
            
    @torch.no_grad
    def get_guessed_targets(self, im, im_):
        if self.epoch <= self.args.warmup:
            return F.softmax(self.model(im_), dim=1)
        
        out1 = F.softmax(self.model_1(im_), dim=1)
        out2 = F.softmax(self.model_2(im_), dim=1)            
        return (out1 + out2) / 2
    
    def val(self, epoch, f=None, dataset='val'):        
        acc1, acc2, acc_ens = 0, 0, 0
        vbar = tqdm(self.val_loader)
        vbar.set_description("Validation")
        total = 0
        losses = torch.tensor([])
        trainLabels = torch.LongTensor(torch.argmax(self.track_loader.dataset.targets, dim=1))
        acc_m = 0
        #Computing linear val acc & kNN acc                                                                                           
        top1, top5 = 0, 0
        with torch.no_grad():
            for i, sample in enumerate(vbar):
                sample['image'] = self.transform_test(sample['image'])                
                image, target, ids = sample['image'], sample['target'], sample['index']
                (outputs1, feat_cont1), (outputs2, feat_cont2) = self.get_val_outputs(image, return_features=True)

                preds1 = torch.argmax(F.log_softmax(outputs1, dim=1), dim=1)
                preds2 = torch.argmax(F.log_softmax(outputs2, dim=1), dim=1)
                preds_ens = torch.argmax(F.log_softmax((outputs1 + outputs2)/2, dim=1), dim=1)

                acc1 += torch.sum(preds1 == target.data)
                acc2 += torch.sum(preds2 == target.data)
                acc_ens += torch.sum(preds_ens == target.data)
                total += preds1.size(0)

        acc1, acc2, acc_ens, total = self.gather((acc1, acc2, acc_ens, total))

        if self.accelerator.global_rank == 0:
            acc1, acc2, acc_ens, total = acc1.sum(dim=0).item(), acc2.sum(dim=0).item(), acc_ens.sum(dim=0).item(), total.sum(dim=0).item()
            final_acc1 = float(acc1)/total * 100.
            final_acc2 = float(acc2)/total * 100.
            final_acc_ens = float(acc_ens)/total * 100.
            
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            
            if final_acc_ens > self.best[0]:
                self.best = (final_acc_ens, final_acc1, final_acc2)
                self.best_epoch = epoch
                self.save_model(epoch, best=True)
                with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                    f.write(str(self.best))
                    f.write(str(self.best_indi))
            if final_acc1 > self.best_indi[0]:
                self.best_indi = (final_acc1, self.best_indi[1])
                with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                    f.write(str(self.best))
                    f.write(str(self.best_indi))
            if final_acc2 > self.best_indi[1]:
                self.best_indi = (self.best_indi[0], final_acc2)
                with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                    f.write(str(self.best))
                    f.write(str(self.best_indi))
                                            
            #Checkpoint
            self.save_model(epoch)

            print(f'Validation Accuracy: {final_acc_ens:.2f}, net1 {final_acc1:.2f}, net2 {final_acc2:.2f}, best accuracy {self.best[0]:.2f} at epoch {self.best_epoch}')
        return
            
    def get_val_outputs(self, image, **kwargs):
        outs1, feats1 = self.model_1(image, **kwargs)
        outs2, feats2 = self.model_2(image, **kwargs)
        return (outs1, feats1), (outs2, feats2)

    def train(self, epoch):
       
        self.model = self.model_1
        self.optimizer = self.optimizer1
        self.model_other = self.model_2

        super().train(epoch)
        
        self.guessed_labels_soft_1 = self.guessed_labels_soft
                
        if epoch > 0:
            self.model_2.eval()
            self.model_1.eval()
            self.val(epoch)
            self.model_2.train()
            self.model_1.train()

        self.model = self.model_2
        self.optimizer = self.optimizer2
        self.model_other = self.model_1

        if epoch > self.args.warmup:
            self.is_clean, self.is_noisy = self.is_clean_other, self.is_noisy_other
            self.weights = self.weights_other

        super().train(epoch)
        
        self.guessed_labels_soft_2 = self.guessed_labels_soft
        
    def save_model(self, epoch, t=False, best=False):
        if t:
            checkname = os.path.join(self.args.save_dir, '{}_{}.pth.tar'.format(self.args.checkname, epoch))
        elif best:
            checkname = os.path.join(self.args.save_dir, '{}_best.pth.tar'.format(self.args.checkname, epoch))
            with open(os.path.join(self.args.save_dir, 'bestpred_{}.txt'.format(self.args.checkname)), 'w') as f:
                f.write(str(self.best))
        else:
            checkname = os.path.join(self.args.save_dir, '{}.pth.tar'.format(self.args.checkname, epoch))
            
        torch.save({
            'epoch': epoch+1,
            'state_dict': self.model_1.module.state_dict(),
            'optimizer': self.optimizer1.state_dict(),
            'state_dict2': self.model_2.module.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'guessed_labels_soft_1': self.guessed_labels_soft_1,
            'guessed_labels_soft_2': self.guessed_labels_soft_2,
            'weights_1': self.weights,
            'is_clean': self.is_clean,
            'is_noisy': self.is_noisy,
            'weights_2': self.weights_other,
            'is_clean_other': self.is_clean_other,
            'is_noisy_other': self.is_noisy_other,
            'best': self.best,
            'best_epoch':self.best_epoch
        }, checkname)



class Trainer_cotrain_indep(Trainer_cotrain):
    def __init__(self, *args, **kwargs):    
        super(Trainer_cotrain_indep, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def get_guessed_targets(self, im, im_):
        out = F.softmax(self.model(im_), dim=1)        
        return out

    def detect_noise(self, which, correct=True):
        self.model = self.model_2
        self.guessed_labels_soft = self.guessed_labels_soft_2
        preds2 = super(Trainer_cotrain, self).detect_noise(which, correct)
        self.is_clean_other, self.is_noisy_other = copy.deepcopy(self.is_clean), copy.deepcopy(self.is_noisy)
        self.weights_other = self.weights

        self.model = self.model_1
        self.guessed_labels_soft = self.guessed_labels_soft_1
        preds1 = super(Trainer_cotrain, self).detect_noise(which, correct)
        
        return (preds1 * preds2)

class Trainer_cotrain_DM(Trainer_cotrain):
    def __init__(self, *args, **kwargs):
        super(Trainer_cotrain_DM, self).__init__(*args, **kwargs)
        
    def detect_noise(self, which, correct=True):
        self.model = self.model_1
        self.guessed_labels_soft = self.guessed_labels_soft_2
        preds1 = super(Trainer_cotrain, self).detect_noise(which, correct)
        self.is_clean_other, self.is_noisy_other = copy.deepcopy(self.is_clean), copy.deepcopy(self.is_noisy)
        self.weights_other = self.weights
        
        self.model = self.model_2
        self.guessed_labels_soft = self.guessed_labels_soft_1
        preds2 = super(Trainer_cotrain, self).detect_noise(which, correct)       
               
        return (preds1 * preds2)

class Trainer_cotrain_vote(Trainer_cotrain):
    def __init__(self, *args, **kwargs):
        super(Trainer_cotrain_vote, self).__init__(*args, **kwargs)
    
    def detect_noise(self, which, correct=True):        
        self.model = self.model_2
        self.guessed_labels_soft = self.guessed_labels_soft_2
        preds2 = super(Trainer_cotrain, self).detect_noise(which, correct)
        self.is_clean_other, self.is_noisy_other = copy.deepcopy(self.is_clean), copy.deepcopy(self.is_noisy)
        self.weights_other = self.weights


        self.model = self.model_1
        self.guessed_labels_soft = self.guessed_labels_soft_1
        preds1 = super(Trainer_cotrain, self).detect_noise(which, correct)

        self.is_noisy = self.is_noisy * self.is_noisy_other
        self.is_clean = ~self.is_noisy

        self.is_clean_other, self.is_noisy_other = self.is_clean, self.is_noisy
        self.weights = (self.weights+self.weights_other)/2
        self.weights[~self.is_noisy] = 1
        
        return (preds1 * preds2)


class Trainer_cotrain_ours(Trainer_cotrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)            
    
    def detect_noise(self, which, correct=True):
        self.model = self.model_1
        self.guessed_labels_soft = self.guessed_labels_soft_2
        preds1 = super(Trainer_cotrain, self).detect_noise(which, correct)
        self.is_clean_other, self.is_noisy_other = copy.deepcopy(self.is_clean), copy.deepcopy(self.is_noisy)
        self.weights_other = self.weights
        
        self.model = self.model_2
        self.guessed_labels_soft = self.guessed_labels_soft_1
        preds2 = super(Trainer_cotrain, self).detect_noise(which, correct)

        self.is_noisy = self.is_noisy * self.is_noisy_other
        self.is_clean = ~self.is_noisy

        self.is_clean_other, self.is_noisy_other = self.is_clean, self.is_noisy
        self.weights = (self.weights+self.weights_other)/2
        self.weights[~self.is_noisy] = 1
        
        return (preds1 * preds2)
    
    
