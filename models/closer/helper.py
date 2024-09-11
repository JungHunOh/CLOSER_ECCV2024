# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    supcon_criterion = SupConLoss(temperature=1/args.supcon_temp)

    supcon = 0
    ce = 0
    cossim = 0
    std = 0
    inter = 0
    intra = 0

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_ for _ in batch]

        B,C,H,W = data[0].shape

        data = torch.cat(data,dim=0).cuda()
        num_aug = args.num_aug

        train_label = train_label.cuda()

        if args.closer:
            feats = model.module.encoder(data)
            
            feats = F.normalize(feats,dim=-1)
            logits = F.linear(feats, F.normalize(model.module.classifier.weight[:args.base_class],dim=-1)) * args.temp
            ce_loss = F.cross_entropy(logits, train_label.repeat(logits.shape[0]//B))
            
            supcon_loss = supcon_criterion(feats.reshape(1+num_aug,B,-1).permute(1,0,2))

            labels = train_label.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().cuda()
            mask = mask.repeat(1+num_aug,1+num_aug) *2 -1

            cossim_mat = F.linear(feats,feats)
            mask = torch.ones(cossim_mat.shape).triu(diagonal=1).cuda() * mask
            intra_loss = (cossim_mat * (mask==1)).sum() / (mask==1).sum() # intra class
            inter_loss = (cossim_mat * (mask==-1)).sum() / (mask==-1).sum() # inter class

            loss = ce_loss + supcon_loss * args.ssc_lamb - inter_loss*args.inter_lamb
            acc = count_acc(logits, train_label.repeat(logits.shape[0]//B))

            supcon += supcon_loss.item()
            inter += inter_loss.item()
            intra += intra_loss.item()
            ce += ce_loss.item()
        else:
            feats = model.module.encoder(data)
            
            feats = F.normalize(feats,dim=-1)
            logits = F.linear(feats, F.normalize(model.module.classifier.weight[:args.base_class],dim=-1)) * args.temp
            
            ce_loss = F.cross_entropy(logits, train_label.repeat(cosine.shape[0]//B))
            
            loss = ce_loss
            acc = count_acc(cosine, train_label.repeat(cosine.shape[0]//B))

            ce += ce_loss.item()
        
        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},ce loss={:.4f}, supcon loss={:.4f}, inter loss={:.4f}, intra loss={:.4f}, acc={:.4f}'.format(epoch, lrc, ce /i,supcon/i, inter/i, intra/i, acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl = [ce/i,inter/i, intra/i,supcon/i]
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model.module.encoder(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)
    
    model.module.classifier.weight.data[:args.base_class] = proto_list

    return model

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va_total = 0
    va_correct = 0
    if session > 0:
        va_base_total = 0
        va_base_correct = 0
        va_new_total = 0
        va_new_correct = 0

    model.session = session
    model.test = True

    labels = []
    pred = []        
    accs = np.zeros([100])
    feats = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]

            logits = model(data)
            logits = logits[:, :test_class]

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            pred = logits.argmax(dim=-1)
            correct = pred == test_label

            vl.add(loss.item())
            va_total += pred.shape[0]
            va_correct += correct.sum()

            if session > 0:
                pred = logits.argmax(dim=-1)
                correct = pred == test_label

                base_mask = test_label < args.base_class
                new_mask = test_label >= args.base_class

                va_base_total += base_mask.sum()
                va_new_total += new_mask.sum()

                va_base_correct += correct[base_mask].sum()
                va_new_correct += correct[new_mask].sum()
            

        vl = vl.item()
        va = va_correct / va_total
        if session >0:
            va_base = va_base_correct / va_base_total
            va_new = va_new_correct / va_new_total

            assert va_total == va_base_total + va_new_total
            assert va_correct == va_base_correct + va_new_correct
        
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    if session > 0:
        return vl, va, va_base, va_new
    else:
        return vl, va

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif (labels is None and mask is None):
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss