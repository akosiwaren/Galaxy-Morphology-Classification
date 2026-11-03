
import torch
import torch.nn as nn
import torchvision.models as models


# ==========================================
# 1. CHANNEL ATTENTION MODULE
# ==========================================

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):

        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(in_channels // reduction, 1)

        self.shared_mlp = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                in_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


# ==========================================
# 2. SPATIAL ATTENTION MODULE
# ==========================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ==========================================
# 3. CONVOLUTIONAL BLOCK ATTENTION MODULE
# ==========================================

class CBAM(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction=16,
        kernel_size=7
    ):

        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(
            in_channels,
            reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size
        )


    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# ==========================================
# 4. CBAM-INTEGRATED RESNET18
# ==========================================

class CBAMResNet18(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):

        super(CBAMResNet18, self).__init__()

        # Load ResNet18 backbone
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(
            weights=weights
        )

        # Insert CBAM after each residual stage
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Replace original classification layer
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(
            num_features,
            num_classes
        )


    def forward(self, x):

        # Initial ResNet18 layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Residual stage 1 + CBAM
        x = self.backbone.layer1(x)
        x = self.cbam1(x)

        # Residual stage 2 + CBAM
        x = self.backbone.layer2(x)
        x = self.cbam2(x)

        # Residual stage 3 + CBAM
        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        # Residual stage 4 + CBAM
        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        # Global average pooling
        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        # Final classification
        x = self.classifier(x)

        return x


# ==========================================
# 5. MODEL VERIFICATION
# ==========================================

if __name__ == "__main__":

    print("CBAM-INTEGRATED RESNET18")
    print("------------------------")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = CBAMResNet18(
        num_classes=10,
        pretrained=True
    )

    model = model.to(device)

    print(f"Device: {device}")

    # Test with a sample batch
    dummy_input = torch.randn(
        2, 3, 256, 256
    ).to(device)

    model.eval()

    with torch.no_grad():

        output = model(dummy_input)

    print("\nMODEL VERIFICATION")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\nMODEL PARAMETERS")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nCBAM MODULES")

    print("Channel Attention: Available")
    print("Spatial Attention: Available")
    print("CBAM after Layer 1: Available")
    print("CBAM after Layer 2: Available")
    print("CBAM after Layer 3: Available")
    print("CBAM after Layer 4: Available")

    print("\nModel architecture initialized successfully!")
