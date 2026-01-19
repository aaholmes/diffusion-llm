#!/usr/bin/env python3
"""
Visualize the neural network architecture for the diffusion captioning model.

Creates two types of visualizations:
1. Detailed computation graph (using torchview)
2. High-level architecture diagram (using graphviz)

Usage:
    python visualize_architecture.py
    python visualize_architecture.py --detailed  # Include all layer details
"""

import argparse
import torch
from graphviz import Digraph

from src.core.model import DiffusionTransformer, ModelConfig


def create_high_level_diagram(output_path: str = "architecture_captioning"):
    """
    Create a high-level architecture diagram for the image captioning model.

    Shows the flow: Image -> CLIP -> Cross-Attention Decoder -> Caption
    """
    dot = Digraph(comment='Image Captioning Architecture')
    dot.attr(rankdir='TB', size='12,16', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')

    # Define colors
    colors = {
        'input': '#E8F5E9',      # Light green
        'frozen': '#BBDEFB',     # Light blue
        'trainable': '#FFF3E0',  # Light orange
        'output': '#F3E5F5',     # Light purple
        'process': '#FFFDE7',    # Light yellow
    }

    # Title
    dot.attr(label='Image Captioning: CLIP Encoder + Diffusion Decoder\n\n',
             labelloc='t', fontsize='20', fontname='Helvetica-Bold')

    # Input cluster
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input', style='rounded', bgcolor='#FAFAFA')
        c.node('image', 'Image\n(224×224×3)', fillcolor=colors['input'])
        c.node('noisy_caption', 'Noisy Caption\n[MASK] tokens', fillcolor=colors['input'])
        c.node('timestep', 'Timestep t\n∈ [0, 1]', fillcolor=colors['input'])

    # CLIP Encoder (frozen)
    with dot.subgraph(name='cluster_clip') as c:
        c.attr(label='Vision Encoder (Frozen)', style='rounded', bgcolor='#E3F2FD')
        c.node('clip', 'CLIP ViT-B/32\n(pretrained)', fillcolor=colors['frozen'])
        c.node('clip_features', 'Image Features\n[50, 768]', fillcolor=colors['frozen'])

    # Decoder (trainable)
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Diffusion Decoder (Trainable)', style='rounded', bgcolor='#FFF8E1')

        # Embeddings
        c.node('token_emb', 'Token\nEmbedding', fillcolor=colors['trainable'])
        c.node('pos_emb', 'Position\nEmbedding', fillcolor=colors['trainable'])
        c.node('time_emb', 'Timestep\nEmbedding', fillcolor=colors['trainable'])
        c.node('add_emb', '+', shape='circle', fillcolor=colors['process'])

        # Transformer blocks
        c.node('blocks', 'Transformer Blocks ×6\n(Self-Attn + Cross-Attn + FFN)',
               fillcolor=colors['trainable'], shape='box3d')

        # Output
        c.node('norm', 'LayerNorm', fillcolor=colors['trainable'])
        c.node('proj', 'Output Projection\n→ vocab logits', fillcolor=colors['trainable'])

    # Output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='rounded', bgcolor='#FAFAFA')
        c.node('logits', 'Logits\n[seq_len, vocab_size]', fillcolor=colors['output'])
        c.node('caption', 'Generated Caption\n"A small red square"', fillcolor=colors['output'])

    # Edges - main flow
    dot.edge('image', 'clip')
    dot.edge('clip', 'clip_features')

    dot.edge('noisy_caption', 'token_emb')
    dot.edge('token_emb', 'add_emb')
    dot.edge('pos_emb', 'add_emb')
    dot.edge('timestep', 'time_emb')
    dot.edge('time_emb', 'add_emb')

    dot.edge('add_emb', 'blocks')
    dot.edge('clip_features', 'blocks', label='cross-attention', style='dashed', color='#1976D2')

    dot.edge('blocks', 'norm')
    dot.edge('norm', 'proj')
    dot.edge('proj', 'logits')
    dot.edge('logits', 'caption', label='iterative\ndenoising', style='dashed')

    # Save
    dot.render(output_path, format='png', cleanup=True)
    dot.render(output_path, format='svg', cleanup=True)
    print(f"Saved: {output_path}.png and {output_path}.svg")

    return dot


def create_transformer_block_diagram(output_path: str = "architecture_block"):
    """
    Create a detailed diagram of a single transformer block with cross-attention.
    """
    dot = Digraph(comment='Transformer Block')
    dot.attr(rankdir='TB', size='8,12', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')

    colors = {
        'norm': '#E1F5FE',
        'attention': '#FFF3E0',
        'ffn': '#F3E5F5',
        'residual': '#E8F5E9',
    }

    dot.attr(label='Transformer Block (Pre-Norm with Cross-Attention)\n\n',
             labelloc='t', fontsize='16', fontname='Helvetica-Bold')

    # Input
    dot.node('input', 'Input x\n[batch, seq, d_model]', fillcolor='#FAFAFA')

    # Self-attention path
    dot.node('norm1', 'LayerNorm', fillcolor=colors['norm'])
    dot.node('self_attn', 'Self-Attention\n(bidirectional)', fillcolor=colors['attention'])
    dot.node('add1', '+', shape='circle', fillcolor=colors['residual'])

    # Cross-attention path
    dot.node('norm_cross', 'LayerNorm', fillcolor=colors['norm'])
    dot.node('cross_attn', 'Cross-Attention\n(Q from decoder,\nK,V from encoder)',
             fillcolor=colors['attention'])
    dot.node('add2', '+', shape='circle', fillcolor=colors['residual'])
    dot.node('encoder', 'Encoder Output\n(CLIP features)', fillcolor='#BBDEFB', style='rounded,filled,dashed')

    # FFN path
    dot.node('norm2', 'LayerNorm', fillcolor=colors['norm'])
    dot.node('ffn', 'Feed-Forward\nLinear→GELU→Linear', fillcolor=colors['ffn'])
    dot.node('add3', '+', shape='circle', fillcolor=colors['residual'])

    # Output
    dot.node('output', 'Output\n[batch, seq, d_model]', fillcolor='#FAFAFA')

    # Main flow
    dot.edge('input', 'norm1')
    dot.edge('norm1', 'self_attn')
    dot.edge('self_attn', 'add1')
    dot.edge('input', 'add1', style='dashed', label='residual')

    dot.edge('add1', 'norm_cross')
    dot.edge('norm_cross', 'cross_attn')
    dot.edge('encoder', 'cross_attn', style='dashed', color='#1976D2')
    dot.edge('cross_attn', 'add2')
    dot.edge('add1', 'add2', style='dashed', label='residual')

    dot.edge('add2', 'norm2')
    dot.edge('norm2', 'ffn')
    dot.edge('ffn', 'add3')
    dot.edge('add2', 'add3', style='dashed', label='residual')

    dot.edge('add3', 'output')

    dot.render(output_path, format='png', cleanup=True)
    dot.render(output_path, format='svg', cleanup=True)
    print(f"Saved: {output_path}.png and {output_path}.svg")

    return dot


def create_diffusion_process_diagram(output_path: str = "architecture_diffusion"):
    """
    Create a diagram showing the diffusion process for text generation.
    """
    dot = Digraph(comment='Diffusion Process')
    dot.attr(rankdir='LR', size='14,6', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')

    dot.attr(label='Discrete Diffusion: Iterative Denoising\n\n',
             labelloc='t', fontsize='16', fontname='Helvetica-Bold')

    # Show the progression from noisy to clean
    steps = [
        ('t1', 't=1.0', '[MASK] [MASK] [MASK] [MASK] [MASK]', '#FFCDD2'),
        ('t08', 't=0.8', '[MASK] [MASK] red [MASK] [MASK]', '#FFECB3'),
        ('t06', 't=0.6', 'A [MASK] red [MASK] [MASK]', '#FFF9C4'),
        ('t04', 't=0.4', 'A small red [MASK] [MASK]', '#E8F5E9'),
        ('t02', 't=0.2', 'A small red square [MASK]', '#C8E6C9'),
        ('t0', 't=0.0', 'A small red square .', '#A5D6A7'),
    ]

    for i, (node_id, time_label, tokens, color) in enumerate(steps):
        with dot.subgraph(name=f'cluster_{node_id}') as c:
            c.attr(label=time_label, style='rounded', bgcolor=color)
            c.node(node_id, tokens, fillcolor='white')

    # Connect the steps
    for i in range(len(steps) - 1):
        dot.edge(steps[i][0], steps[i+1][0], label='denoise')

    # Add model box
    dot.node('model', 'Diffusion\nDecoder', shape='box3d', fillcolor='#FFF3E0')

    # Show model predicts at each step
    dot.edge('model', 't08', style='dashed', label='predict')

    dot.render(output_path, format='png', cleanup=True)
    dot.render(output_path, format='svg', cleanup=True)
    print(f"Saved: {output_path}.png and {output_path}.svg")

    return dot


def create_torchview_diagram(output_path: str = "architecture_detailed"):
    """
    Create a detailed computation graph using torchview.
    """
    try:
        from torchview import draw_graph
    except ImportError:
        print("torchview not installed. Run: pip install torchview")
        return None

    # Create model
    config = ModelConfig(
        d_model=768,
        n_heads=12,
        n_layers=6,
        d_ff=3072,
        vocab_size=8192,
        max_seq_len=128,
        has_cross_attention=True,
    )
    model = DiffusionTransformer(config)

    # Create sample inputs
    batch_size = 1
    seq_len = 32
    encoder_seq_len = 50

    x = torch.randint(0, 8192, (batch_size, seq_len))
    t = torch.rand(batch_size)
    encoder_output = torch.randn(batch_size, encoder_seq_len, 768)

    # Draw graph
    model_graph = draw_graph(
        model,
        input_data=(x, t, None, encoder_output),
        expand_nested=True,
        depth=3,
        device='cpu',
    )

    # Save
    model_graph.visual_graph.render(output_path, format='png', cleanup=True)
    model_graph.visual_graph.render(output_path, format='svg', cleanup=True)
    print(f"Saved: {output_path}.png and {output_path}.svg")

    return model_graph


def print_model_summary():
    """Print a text summary of the model architecture."""
    config = ModelConfig(
        d_model=768,
        n_heads=12,
        n_layers=6,
        d_ff=3072,
        vocab_size=8192,
        max_seq_len=128,
        has_cross_attention=True,
    )
    model = DiffusionTransformer(config)

    print("=" * 70)
    print("IMAGE CAPTIONING MODEL ARCHITECTURE")
    print("=" * 70)
    print()
    print("VISION ENCODER (Frozen)")
    print("-" * 40)
    print("  CLIP ViT-B/32 (pretrained)")
    print("  Input:  Image [224, 224, 3]")
    print("  Output: Features [50, 768]")
    print("  Params: ~86M (not trained)")
    print()
    print("DIFFUSION DECODER (Trainable)")
    print("-" * 40)
    print(f"  d_model:    {config.d_model}")
    print(f"  n_heads:    {config.n_heads}")
    print(f"  n_layers:   {config.n_layers}")
    print(f"  d_ff:       {config.d_ff}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq:    {config.max_seq_len}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("LAYER BREAKDOWN")
    print("-" * 40)

    # Token embedding
    tok_params = sum(p.numel() for p in model.token_embedding.parameters())
    print(f"  Token Embedding:     {tok_params:>12,} params")

    # Position embedding
    pos_params = sum(p.numel() for p in model.position_embedding.parameters())
    print(f"  Position Embedding:  {pos_params:>12,} params")

    # Timestep embedding
    time_params = sum(p.numel() for p in model.timestep_embedding.parameters())
    print(f"  Timestep Embedding:  {time_params:>12,} params")

    # Transformer blocks
    block_params = sum(p.numel() for p in model.blocks.parameters())
    print(f"  Transformer Blocks:  {block_params:>12,} params ({config.n_layers} blocks)")

    # Per-block breakdown
    block = model.blocks[0]
    self_attn_params = sum(p.numel() for p in block.self_attn.parameters())
    cross_attn_params = sum(p.numel() for p in block.cross_attn.parameters()) if hasattr(block, 'cross_attn') else 0
    ff_params = sum(p.numel() for p in block.ff.parameters())
    norm_params = sum(p.numel() for n, p in block.named_parameters() if 'norm' in n)

    print(f"    - Self-Attention:  {self_attn_params:>12,} params/block")
    print(f"    - Cross-Attention: {cross_attn_params:>12,} params/block")
    print(f"    - Feed-Forward:    {ff_params:>12,} params/block")
    print(f"    - LayerNorms:      {norm_params:>12,} params/block")

    # Output
    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    proj_params = sum(p.numel() for p in model.output_proj.parameters())
    print(f"  Final LayerNorm:     {norm_params:>12,} params")
    print(f"  Output Projection:   {proj_params:>12,} params")

    print()
    print(f"  TOTAL:               {total_params:>12,} params")
    print(f"  Trainable:           {trainable_params:>12,} params")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize model architecture")
    parser.add_argument("--detailed", action="store_true",
                        help="Create detailed computation graph with torchview")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for diagrams")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Print text summary
    print_model_summary()

    # Create high-level diagram
    print("\nCreating architecture diagrams...")
    create_high_level_diagram(f"{args.output_dir}/architecture_captioning")
    create_transformer_block_diagram(f"{args.output_dir}/architecture_block")
    create_diffusion_process_diagram(f"{args.output_dir}/architecture_diffusion")

    # Create detailed graph if requested
    if args.detailed:
        print("\nCreating detailed computation graph...")
        create_torchview_diagram(f"{args.output_dir}/architecture_detailed")

    print("\nDone! Open the .png or .svg files to view the diagrams.")


if __name__ == "__main__":
    main()
