import torch.distributed as dist

_mesh = None


# initializes mesh
def init_mesh(mesh):
    global _mesh
    _mesh = mesh


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _mesh.get_pp_num() - 1
    assert abs(src_stage - dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage):
    global _groups
    src_stage = _mesh.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _mesh.stage_to_global(stage_id=dest_stage)
    return dist.send(tensor, dest_rank)


def recv(tensor, src_stage):
    global _groups
    dest_stage = _mesh.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = _mesh.stage_to_global(stage_id=src_stage)

    return dist.recv(tensor, src_rank)
