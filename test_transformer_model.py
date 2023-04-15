import torch
import argparse
from loguru import logger

from modules.linearTransformer import LinearAttention, LinearTransformerEncodeBlock, LinearTransformerEncoderLayers
from modules.transformer import ScaleDotProductAttention, TransformerEncoderLayers
from modules.SequenceReductionTransformer import SequenceReductionAttention, SequenceReductionTransformerEncoderLayers


def setup():
    logger.info('Setup init parameters')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='self', choices=['self', 'cross'])
    parser.add_argument('--modes', default=['self', 'cross', 'cross', 'self'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup()
    # Device
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # attention mode
    mode = args.mode
    modes = args.modes

    # Inputs
    feats0 = torch.randn((2, 1200, 256)).to(dev)
    feats1 = torch.randn((2, 1200, 256)).to(dev)
    feats = torch.cat((feats0, feats1), dim=0)
    ims = torch.randn((2, 256, 224, 224)).to(dev)

    # Linear Attention
    linearAttn = LinearAttention(256, mode, 8, False).to(dev)
    out = linearAttn(feats)
    print(f'Linear {mode}-attention output shape:{out.shape}')

    # LinearTransformerBlock
    model = LinearTransformerEncodeBlock(256, mode, 8).to(dev)
    out = model(feats)
    print(f'Linear {mode}-attention block output shape:{out.shape}')

    # LinearTransformerLayers
    model = LinearTransformerEncoderLayers(256, modes, 8).to(dev)

    out = model(feats)
    print(f'Linear Transformer Layers output shape:{out.shape}')

    # Scale Dot-Product Attention
    attn = ScaleDotProductAttention(256, mode, 8).to(dev)
    out = attn(feats)
    print(f'Scale Dot-Product {mode}-attention output shape:{out.shape}')

    # TransformerLayers
    model = TransformerEncoderLayers(256, modes, 8).to(dev)
    out = model(feats)
    print(f'Transformer Layers output shape:{out.shape}')

    # Sequence Reduction Attention
    sr_attn = SequenceReductionAttention(256, mode, 8, sr_ratio=2).to(dev)
    out = sr_attn(feats, 30, 40)
    print(f'Sequence Reduction {mode}-attention output shape:{out.shape}')

    # SequenceReductionTransformerLayers
    model = SequenceReductionTransformerEncoderLayers(256, modes, 8, sr_ratio=2).to(dev)
    out = model(feats, 30, 40)
    print(f'Sequence Reduction Transformer Layers output shape:{out.shape}')
