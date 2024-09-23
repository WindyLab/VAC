import torch
from torch import nn
import ot
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

class EMDLoss(torch.nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pred, gt): #predicted->supplier gt->demander
        batch_size = pred.shape[0]
        
        batch_id = 0
        pred = pred[batch_id,0,:,:]
        gt = gt[batch_id,0,:,:]
        print("p_gt max:",torch.max(gt))
        print("p_gt min:",torch.min(gt))
        
        p_pred = torch.nonzero(pred>0.001)
        p_gt = torch.nonzero(gt>0.001)
       # print("p_pred:",p_pred)
      #  print("p_gt:",p_gt)
        n_pre = len(p_pred)
        n_gt = len(p_gt)
        print("n_pre:",n_pre)
        print("n_gt:",n_gt)
        
        xs = torch.zeros((n_pre,2))
        xt = torch.zeros((n_gt,2))
        for i in range(n_pre):
            xs[i,0] = p_pred[i][0]
            xs[i,1] = p_pred[i][1]
        
        for i in range(n_gt):
            xt[i,0] = p_gt[i][0]
            xt[i,1] = p_gt[i][1]
        a, b = torch.ones((n_pre,)) , torch.ones((n_gt,))

        for i in range(0,n_pre):
            a[i] *= pred[p_pred[i][0],p_pred[i][1]]
        for i in range(0,n_gt):
            b[i] *= gt[p_gt[i][0],p_gt[i][1]]
            
        a = torch.softmax(a,dim = -1)
        b = torch.softmax(b,dim = -1)
        
        M = ot.dist(xs, xt)
        #Gs = ot.sinkhorn(a, b, M,reg = 1e1)
        Gs = ot.emd(a, b, M)
        
        M_tensor = torch.tensor(M)
        Gs_tensor = torch.tensor(Gs)
        
        reduction_loss = torch.sum(Gs_tensor * M_tensor)
        reduction_loss = reduction_loss / n_gt / n_pre

        return reduction_loss

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize

class HeatmapLossWithScale(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, gt, mask, scale):
        assert pred.size() == gt.size()

        dis = gt.clone()
        dis = torch.where(torch.gt(gt, 0), gt, gt + 1)
        dis = torch.log(dis)

        scaled_gt = gt + gt*dis*scale + 0.5*gt*dis**2*scale**2
        weight = torch.abs(1-pred) * gt ** 0.01 + torch.abs(pred) * (1 - gt**0.01)

        loss = (pred - scaled_gt)**2 * weight * mask[:,None].expand_as(gt)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)

        return loss

class ScaleLoss(nn.Module): ##regulizer in paper
    def __init__(self):
        super().__init__()

    def forward(self, mask, scale):
        loss = scale ** 2 * mask[:, None, :, :].expand_as(scale)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss
    
class KLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        # ratio = torch.exp(pred.log() - gt.log())
        # clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # return loss_pi
        clip_ratio = 0.2
        loss = pred*(pred.log() - gt.log())
        #loss = torch.clamp(loss, 1-clip_ratio, 1+clip_ratio)
        return torch.sum(loss)
    
@torch.inference_mode()
def evaluate(net, dataloader, device,criterion,show_curve=False):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0
    roc_auc = 0
    corr = 0
    for idx,batch in enumerate(dataloader):
        image, mask_true = batch[0].to(device), batch[1].to(device)
        image = image.to(device=device, dtype=torch.float32)
        #mask_true = mask_true.to(device=device, dtype=torch.long)
        
        mask_pred = net(image)
        mask_pred_classify = torch.sigmoid(mask_pred)
        
        predicted = torch.log_softmax(mask_pred,dim=-1)
        target = torch.log_softmax(mask_true,dim=-1)
        loss = criterion(predicted, target)
        loss = torch.mean(loss)  #equivalent to reduction = mean, see offical description for loss
        
        if net.n_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            
            #temp = mask_true.cpu().numpy()
            #plt.matshow(temp[0,:,:])
            #plt.show()
            
            mask_true = (mask_true > 0.5).int().squeeze(1)
            y_true = torch.flatten(mask_true).cpu().numpy().tolist()
            y_score = torch.flatten(mask_pred_classify).cpu().numpy().tolist()

            y_ture_soft = torch.flatten(target).cpu().numpy().tolist()
            y_score_soft = torch.flatten(predicted).cpu().numpy().tolist()

            corr += np.corrcoef(y_ture_soft, y_score_soft)[0,1]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc += auc(fpr, tpr)
            if show_curve:
            # draw ROC curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.show()
            total_loss += loss
    net.train()
    return total_loss / max(num_val_batches, 1),roc_auc / max(num_val_batches, 1),corr / max(num_val_batches, 1)