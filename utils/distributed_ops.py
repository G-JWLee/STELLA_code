import functools
import os
import pickle
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def init_distributed_environment(gpu, ngpus_per_node, args):
    if args.environment.dist_url == "env://" and args.environment.rank == -1:
        args.environment.rank = int(os.environ["RANK"])
    if args.environment.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        # args.environment.rank = args.environment.rank * ngpus_per_node + gpu
        args.environment.rank = gpu
    dist.init_process_group(backend=args.environment.dist_backend,
                            init_method=args.environment.dist_url,
                            world_size=args.environment.world_size,
                            rank=args.environment.rank)

    if args.environment.gpu is not None:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.environment.gpu)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.optim.batch_size = int(
            args.optim.batch_size / ngpus_per_node)
        args.environment.workers = int(
            (args.environment.workers + ngpus_per_node - 1) /
            ngpus_per_node)
    return args


def send_to_device(module, distributed, device=None, unused_parameters=False):
    if distributed:
        module.cuda(device)
        module = DistributedDataParallel(
            module, device_ids=[device] if device is not None else None,
            find_unused_parameters=unused_parameters,
        )
    elif device is not None:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        module.cuda()
    return module


def shuffle_batch(x1, x2):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    if not dist.is_initialized():
        batch_size_all = x1.shape[0]
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)
        return x1[idx_shuffle], x2[idx_shuffle]

    else:
        # gather from all gpus
        batch_size_this = x1.shape[0]
        x1_gather = concat_all_gather(x1)
        x2_gather = concat_all_gather(x2)
        batch_size_all = x1_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        num_gpus = batch_size_all // batch_size_this
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        # shuffle
        return x1_gather[idx_this], x2_gather[idx_this]


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
