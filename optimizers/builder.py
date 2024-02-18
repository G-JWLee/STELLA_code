import torch.optim
import torch.distributed as dist

def build_optimizer(cfg, model, get_num_layer=None, get_layer_sacle=None):
    weight_decay = cfg.args.weight_decay
    avm_wd_multi = cfg.avm_wd_multi

    skip = {}
    if hasattr(model.backbone.transformer, 'no_weight_decay'):
        skip = model.backbone.transformer.no_weight_decay()

    freeze_list = []
    params = get_parameter_groups(model, weight_decay, avm_wd_multi, freeze_list, skip, get_num_layer, get_layer_sacle, lr=cfg.args.lr)
    cfg.args.weight_decay = 0

    if cfg.method in vars(torch.optim):
        opt = torch.optim.__dict__[cfg.method](
            params,
            **cfg.args
        )
    elif cfg.method.lower() == 'lars':
        import apex
        opt = torch.optim.SGD(
            params,
            **cfg.args
        )
        opt = apex.parallel.LARC.LARC(
            opt,
            trust_coefficient=0.001,
            clip=False
        )
    else:
        raise NotImplementedError(f'Optimizer {cfg.method} not found.')

    return opt


class LayerDecayValueAssigner(object):
    def __init__(self, values, mid_fusion_depth=0):
        self.values = values
        self.mid_fusion_depth = mid_fusion_depth

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_videoaudiomae(var_name, len(self.values), num_modality_depth=self.mid_fusion_depth)


def get_num_layer_for_videoaudiomae(var_name, num_max_layer, num_modality_depth=0):
    if var_name in {"mask_token_a", "mask_token_v", "modality_v", "modality_a", "decoder_modality_v", "decoder_modality_a",
                    "pos_embed_a", "pos_embed_v", "decoder_pos_embed_v", "decoder_pos_embed_a"}:
        return 0

    elif var_name.startswith("patch_embed"):
        return 0

    elif any(var_name.startswith(module_name) for module_name in ["blocks_v", "blocks_a"]):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1

    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + num_modality_depth + 1
    else:
        return num_max_layer - 1


def get_parameter_groups(model, weight_decay=1e-5, avm_wd_multi=1.0, freeze_list=(), skip_list=(), get_num_layer=None, get_layer_scale=None, lr=0.0):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.backbone.transformer.named_parameters():

        if any(mod in name for mod in freeze_list):

            if name.startswith('decoder'):
                pass
            else:
                param.requires_grad = False

        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    if hasattr(model, "avmatching_module"):
        for name, param in model.avmatching_module.named_parameters():

            if not param.requires_grad:
                continue  # frozen weights

            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay_avm"
                this_weight_decay = weight_decay * avm_wd_multi

            if group_name not in parameter_group_names:
                scale = 1.
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr * scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr * scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
