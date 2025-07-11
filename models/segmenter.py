# models/segmenter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UNetWithProjection(nn.Module):
    def __init__(self, encoder_name='efficientnet-b4', encoder_weights='imagenet',
                 in_channels=3, classes=5, emb_dim=64):
        super().__init__()

        # 1) 构造一个 SMP 的 Unet
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
        with torch.no_grad():
            dummy_x = torch.randn(1, in_channels, 256, 256)
            # 先做 encoder
            features = self.unet.encoder(dummy_x)              # 一个 list/tuple
            # 然后做 decoder
            decoder_output = self.unet.decoder(*features)      # [1, dec_out_ch, H, W]
            decoder_out_channels = decoder_output.shape[1]

        # 3) 建立投影层 (1x1 conv) - 用于对比学习的 embedding
        self.proj = nn.Conv2d(decoder_out_channels, emb_dim, kernel_size=1)

    def forward(self, x):
        """
        同时返回:
          logits (用于分割损失: CE, Dice)
          emb    (用于对比学习: Contrastive Loss)
        """
        # a) 编码 & 解码
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)  # [N, dec_out_ch, H, W]

        # b) 最终分割输出
        logits = self.unet.segmentation_head(decoder_output)  # [N, classes, H, W]

        # c) 对比学习特征
        emb = self.proj(decoder_output)    # [N, emb_dim, H, W]
        emb = F.normalize(emb, dim=1)      # L2 归一化
        return logits, emb


def get_unet_model(encoder_name='efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=5):
    """
    创建并返回一个基于 U-Net + Projection Head 的模型，
    返回值是一个 forward 同时输出 (logits, embedding) 的 nn.Module。
    """
    model = UNetWithProjection(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        emb_dim=16  # 可根据需求调整 embedding 维度
    )
    return model


if __name__ == "__main__":
    model = get_unet_model(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        classes=5
    )
    x = torch.randn(2, 3, 256, 256)
    logits, emb = model(x)
    print("logits shape:", logits.shape)  # (2, 5, 256, 256)
    print("embedding shape:", emb.shape)  # (2, 64, 256, 256)

