import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


MODULES = {}


def register(name):
    def decorator(cls):
        MODULES[name] = cls
        return cls
    return decorator


def get_modules(cfg, **kwargs):
    model = MODULES[cfg.name](cfg, **kwargs)
    return model


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to
            compute a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input
            in the positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    if num_encoding_functions == 0:
        return tensor
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_norm_layer(layer_type='inst'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'batch3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif layer_type == 'inst':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif layer_type == 'inst3d':
        norm_layer = functools.partial(
            nn.InstanceNorm3d, affine=False, track_running_stats=False
        )
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            f'normalization layer {layer_type} is not found'
        )
    return norm_layer


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def deconvBlock(input_nc, output_nc, bias, norm_layer=None, nl='relu'):
    layers = [nn.ConvTranspose3d(input_nc, output_nc, 4, 2, 1, bias=bias)]

    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    if nl == 'relu':
        layers += [nn.ReLU(True)]
    elif nl == 'lrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    else:
        raise NotImplementedError('NL layer {} is not implemented' % nl)
    return nn.Sequential(*layers)


if __name__ == "__main__":
    import torch
    from easydict import EasyDict as edict
    nz = 64
    config = edict(bias=False, res=64, nz=nz, ngf=32, max_nf=8, norm="batch3d")
    model = Deconv3DDecoder(config)
    z = torch.zeros((1, 64, 1, 1, 1))
    layers_out, out = model(z)
