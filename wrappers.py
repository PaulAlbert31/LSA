import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseWrapper(nn.Module):
    def __init__(self, model, num_classes, proj_size, dim):
        super().__init__()
        self.model = model

        self.linear = nn.Linear(dim, num_classes) 
        self.contrastive = nn.Linear(dim, proj_size)            

    def get_features(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):#Attention layers in ViTs
                output = output[0]
            self.features[name] = output
        return hook
        
    def forward_pass(self, x, return_features=False, return_interm=None):
        self.features = {}
        feat = self.model(x)        
        out = self.linear(feat)        
        if return_features:
            feat_cont = self.contrastive(feat)
            if return_interm is not None:
                to_return = [f for k, f in self.features.items()]
                f = to_return[return_interm]
                f = F.adaptive_avg_pool2d(f, (1,1)).view(f.shape[0], -1)
                        
                return out, feat_cont, f
            return out, feat_cont        
        return out
        
    def forward(self, x, x2=None, **kwargs):
        if x2 is not None:          
            r1 = self.forward_pass(x, **kwargs)
            r2 = self.forward_pass(x2, **kwargs)            
            return r1, r2
        
        return self.forward_pass(x, **kwargs)


class ResNet_wrapper(BaseWrapper):
    def __init__(self, model, num_classes, proj_size=128, dim=512):
        super().__init__(model, num_classes, proj_size, dim)
        self.model.fc = nn.Identity()        
        self.model.layer1.register_forward_hook(self.get_features('layer1'))
        self.model.layer2.register_forward_hook(self.get_features('layer2'))
        self.model.layer3.register_forward_hook(self.get_features('layer3'))
        self.model.layer4.register_forward_hook(self.get_features('layer4'))

class Inception_wrapper(BaseWrapper):
    def __init__(self, model, num_classes, proj_size=128, dim=512):
        super().__init__(model, num_classes, proj_size, dim)
        self.model.fc = nn.Identity()
        self.model.conv2d_2b.register_forward_hook(self.get_features('layer1'))
        self.model.conv2d_4a.register_forward_hook(self.get_features('layer2'))
        self.model.mixed_7a.register_forward_hook(self.get_features('layer3'))        

class CLIP_wrapper(BaseWrapper):
    def __init__(self, model, num_classes, proj_size=128, dim=768):
        super().__init__(model, num_classes, proj_size, dim)
        
        #Lang params
        delattr(self.model, "transformer")
        delattr(self.model, "token_embedding")
        delattr(self.model, "ln_final")
        delattr(self.model, "positional_embedding")
        delattr(self.model, "text_projection")                                
        delattr(self.model, "logit_scale")
        
        if hasattr(self.model.visual, "layer1"):
            self.type = 'conv'
            self.model.visual.layer1.register_forward_hook(self.get_features('layer1'))
            self.model.visual.layer2.register_forward_hook(self.get_features('layer2'))
            self.model.visual.layer3.register_forward_hook(self.get_features('layer3'))
            self.model.visual.layer4.register_forward_hook(self.get_features('layer4'))
        else:
            self.type = 'transformer'
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                block.register_forward_hook(self.get_features(f'layer{i}'))
            
    def forward_pass(self, x, return_features=False, return_interm=None):
        self.features = {}
        feat = self.model.encode_image(x)
        feat = F.normalize(feat, p=2)
        
        out = self.linear(feat)
        
        if return_features:
            feat_cont = self.contrastive(feat)
            if return_interm is not None:
                to_return = [f for k, f in self.features.items()]
                f = to_return[return_interm]
                if return_interm != len(to_return)-1:
                    if self.type == "transformer":
                        f = f.mean(dim=1)
                    else:
                        f = F.adaptive_avg_pool2d(f, (1,1)).view(f.shape[0], -1)
                        
                return out, feat_cont, f
            return out, feat_cont        
        return out

    
