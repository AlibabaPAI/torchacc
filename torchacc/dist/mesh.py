# This file is largely inspired by and mostly follows the structure of
# ``deepspeed.runtime.pipe.topology.ProcessTopology`` in
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/topology.py

from collections import namedtuple
from itertools import product as cartesian_product

import torch.distributed as dist
import torchacc as ta


class ProcessTopology:
    """Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """

    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Args:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis

        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        self.ProcessCoord = namedtuple('ProcessCoord', axes)

        self.mapping = {}
        ranges = [range(d) for d in dims]
        # example: 1, (0,0,1)
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            # for example, {ProcessCoord(row=0, col=1) : 1}
            self.mapping[key] = global_rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError(
                'get_rank() does not support slices. Use filter_match())')

        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {coord_kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data', 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """

        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coord] for coord in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [
            self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx
        ]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


class Mesh:
    """Implements a mesh object that stores the parallel ranks
    corresponding to each of the parallel strategies.
    """

    def __init__(self, dp_num=1, pp_num=1, tp_num=1, fsdp_num=1, sp_num=1, topology=None):
        self.global_rank = ta.dist.rank()
        self.world_size = ta.dist.world_size()

        # default_topo = ['dp', 'fsdp', 'pp', 'tp', 'sp']
        default_topo = ['dp', 'fsdp', 'pp', 'tp']
        default_dims = [dp_num, fsdp_num, pp_num, tp_num]
        if topology is not None:
            assert isinstance(topology, list)
            dims = []
            assert len(topology) == len(set(topology))
            for axis in topology:
                if axis == 'dp':
                    dims.append(dp_num)
                elif axis == 'pp':
                    dims.append(pp_num)
                elif axis == 'tp':
                    dims.append(tp_num)
                elif axis == 'fsdp':
                    dims.append(fsdp_num)
                # elif axis == 'sp':
                #     dims.append(sp_num)
                else:
                    raise ValueError(
                        f"Expect 'dp', 'pp', 'tp' 'fsdp' or 'sp' in topology, but got {axis}"
                    )
            for p in default_topo:
                if p not in topology:
                    dims.append(1)
                    topology.append(p)
            self._topo = ProcessTopology(axes=topology, dims=dims)
        else:
            self._topo = ProcessTopology(axes=default_topo, dims=default_dims)

        self.dp_num = dp_num
        self.pp_num = pp_num
        self.tp_num = tp_num
        self.fsdp_num = fsdp_num
        self.sp_num = sp_num
        

        # Create new ProcessGroup for data collectives - these are the data parallel groups
        self.dp_group = None
        self.dp_proc_group = None
        self.dp_groups = self._topo.get_axis_comm_lists('dp')
        if self.dp_num > 1:
            self._check_mesh_valid()
            for ranks in self.dp_groups:
                proc_group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.dp_group = ranks
                    self.dp_proc_group = proc_group

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = None
        self.pp_proc_group = None
        self.pp_groups = self._topo.get_axis_comm_lists('pp')
        if self.pp_num > 1:
            self._check_mesh_valid()
            for ranks in self.pp_groups:
                proc_group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.pp_group = ranks
                    self.pp_proc_group = proc_group

        # Create new ProcessGroup for tensor collectives - these are tensor parallel groups
        self.tp_group = None
        self.tp_proc_group = None
        self.tp_groups = self._topo.get_axis_comm_lists('tp')
        if self.tp_num > 1:
            self._check_mesh_valid()
            for ranks in self.tp_groups:
                proc_group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.tp_group = ranks
                    self.tp_proc_group = proc_group

        # Create new ProcessGroup for fsdp collectives - these are fsdp parallel groups
        self.fsdp_group = None
        self.fsdp_proc_group = None
        self.fsdp_groups = self._topo.get_axis_comm_lists('fsdp')
        if self.fsdp_num > 1:
            self._check_mesh_valid()
            for ranks in self.fsdp_groups:
                proc_group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.fsdp_group = ranks
                    self.fsdp_proc_group = proc_group
                    
        # Create new ProcessGroups for sp collectives - these are sequence parallel groups
        self.sp_group = None
        self.sp_proc_group = None
        self.sp_groups = self._topo.get_axis_comm_lists('sp')
        if self.sp_num > 1:
            self._check_mesh_valid()
            for ranks in self.sp_groups:
                proc_group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self.sp_group = ranks
                    self.sp_proc_group = proc_group

    def _check_mesh_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        assert ranks == self.world_size, f"The configured parallel strategy should utilize all GPU devices."

    # general
    def get_global_rank(self):
        return self.global_rank

    def get_world_size(self):
        return self.world_size

    # DP
    def get_dp_rank(self):
        return self._topo.get_coord(rank=self.global_rank).dp

    def get_dp_num(self):
        """ The number of dp. """
        return self.dp_num

    def get_dp_proc_group(self):
        """ The distributed group (return from torch.distributed.new_group) within the same dp. """
        return self.dp_proc_group

    def get_dp_rank_groups(self):
        """ A list of list, the groups of ranks within the same dp. """
        return self.dp_groups

    # PP
    def get_stage_id(self):
        return self._topo.get_coord(rank=self.global_rank).pp

    def is_first_stage(self):
        return self.get_stage_id() == 0

    def is_last_stage(self):
        return self.get_stage_id() == (self.pp_num - 1)

    # returns the global rank of the process with the provided stage id
    # which has the same dp_rank, tp_rank and fsdp_rank as caller process
    def stage_to_global(self, stage_id):
        me = self._topo.get_coord(self.global_rank)
        transform = me._replace(pp=stage_id)._asdict()
        return self._topo.get_rank(**transform)

    def get_pp_rank(self):
        """ The stage of the pipeline this rank resides in. """
        return self.get_stage_id()

    def get_pp_num(self):
        """ The number of stages in the pipeline. """
        return self.pp_num

    def get_pp_proc_group(self):
        """ The distributed group (return from torch.distributed.new_group) within the same pp. """
        return self.pp_proc_group

    def get_pp_rank_groups(self):
        """ A list of list, the groups of ranks within the same pp. """
        return self.pp_groups

    # TP
    def get_tp_rank(self):
        return self._topo.get_coord(rank=self.global_rank).tp

    def get_tp_num(self):
        """ The number of tp. """
        return self.tp_num

    def get_tp_proc_group(self):
        """ The distributed group (return from torch.distributed.new_group) within the same tp. """
        return self.tp_proc_group

    def get_tp_rank_groups(self):
        """ A list of list, the groups of ranks within the same tp. """
        return self.tp_groups

    # FSDP
    def get_fsdp_rank(self):
        return self._topo.get_coord(rank=self.global_rank).fsdp

    def get_fsdp_num(self):
        """ The number of fsdp. """
        return self.fsdp_num

    def get_fsdp_proc_group(self):
        """ The distributed group (return from torch.distributed.new_group) within the same fsdp. """
        return self.fsdp_proc_group

    def get_fsdp_rank_groups(self):
        """ A list of list, the groups of ranks within the same fsdp. """
        return self.fsdp_groups

    # SP
    def get_sp_num(self):
        """ The number of sp. """
        return self.sp_num
