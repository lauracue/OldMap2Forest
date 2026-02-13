# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        
        self.out_layer = nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.activ = nn.ReLU()

    
    def forward(self, x):
        x = self.up(x)
        x = self.activ(self.bn(self.conv(x)))
        out = self.out_layer(x)
        
        return out



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SoftBootstrappingLoss(nn.Module):
    def __init__(self, beta=0.95, epsilon=1e-8):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        soft_labels = torch.sigmoid(inputs)

        loss = - (self.beta * targets * torch.log(soft_labels + self.epsilon) +
                  (1 - self.beta) * (1 - targets) * torch.log(1 - soft_labels + self.epsilon))

        return loss


class HardBootstrappingLoss(nn.Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, beta=0.8, reduce=False):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)



class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, channel, classes=1, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        
        # self.batch1 = nn.BatchNorm2d(channel)

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        # self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.upsample_1 = Upsample_(scale=2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        # self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.upsample_2 = Upsample_(scale=2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        # self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.upsample_3 = Upsample_(scale=2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        
        self.drop = nn.Dropout2d(p=0.5)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], classes, 1, 1),

        )
        
        self.sigma = nn.Parameter(torch.ones(1))


    def forward(self, x):
        # Encode
        # x = self.batch1(x)
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = nn.functional.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        # x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = nn.functional.interpolate(x6, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = nn.functional.interpolate(x8, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)
        x10 = self.drop(x10)
        output = self.output_layer(x10)

        return output
    

    
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, weight_mask):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_BCE_loss = BCE_loss * weight_mask  # Apply the weights mask here
        return weighted_BCE_loss.mean()  # Return the mean loss over all elements


    
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weight_mask):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss *= weight_mask  # Apply the weights mask here
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()  # Return the mean loss over all elements
    
class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, alpha = 0.1):
        super(WeightedAsymmetricLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, weight_mask):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_loss *= weight_mask 
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) * bce_loss + (1 - self.alpha) * pt * bce_loss
        return loss
    
#PyTorch

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, weight_mask):
        assert y_pred.size() == y_true.size() == weight_mask.size()
        
        y_pred = torch.sigmoid(y_pred)
        
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        weight_mask = weight_mask.contiguous().view(-1)
        
        weighted_intersection = (y_pred * y_true * weight_mask).sum()
        weighted_union = (y_pred * weight_mask).sum() + (y_true * weight_mask).sum()
        
        weighted_dice_coeff = (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        
        return 1 - weighted_dice_coeff

    
class WeightedJaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true, weight_mask):
        assert y_pred.size() == y_true.size() == weight_mask.size()
        
        y_pred = torch.sigmoid(y_pred)
        
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        weight_mask = weight_mask.contiguous().view(-1)
        
        weighted_intersection = (y_pred * y_true * weight_mask).sum()
        weighted_sum = (y_pred * weight_mask).sum() + (y_true * weight_mask).sum()
        weighted_union = weighted_sum - weighted_intersection
        
        weighted_jaccard_coeff = (weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        
        return 1 - weighted_jaccard_coeff
    
    
class FBetaLoss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-7):
        """
        Initialize the F-beta loss module.
        
        Args:
            beta (float): The beta value in F-beta score. Determines the weight of recall in the combined score.
                          beta < 1 lends more weight to precision, while beta > 1 favors recall.
            epsilon (float): Small value to avoid division by zero.
        """
        super(FBetaLoss, self).__init__()
        self.beta = beta
        self.beta_squared = beta ** 2
        self.epsilon = epsilon

    def forward(self, y_true, y_pred, weight_mask):
        """
        Forward pass for F-beta loss, with optional weighting, for 3D inputs.
        
        Args:
            y_true (Tensor): True binary labels, shape [batch_size, rows, columns].
            y_pred (Tensor): Predicted probabilities, shape [batch_size, rows, columns].
            weight_mask (Tensor): Weights for each element, same shape as y_true and y_pred.
            
        Returns:
            Tensor: The computed weighted F-beta loss.
        """
        # Flatten y_true, y_pred, and weight_mask to treat each pixel independently
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        weight_mask_flat = weight_mask.view(-1)

        
        # Compute true positives, false positives, and false negatives with weighting
        tp = (y_true_flat * y_pred_flat * weight_mask_flat).sum()
        fp = ((1 - y_true_flat) * y_pred_flat * weight_mask_flat).sum()
        fn = (y_true_flat * (1 - y_pred_flat) * weight_mask_flat).sum()

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        fbeta = (1 + self.beta_squared) * precision * recall / (self.beta_squared * precision + recall + self.epsilon)
        
        return 1 - fbeta
