"""ASE Calculator for MACE-H Hamiltonian prediction.

Provides a Calculator interface that loads a trained MACE-H model and
predicts Hamiltonians for arbitrary ASE Atoms objects with periodic
boundary conditions.

Example
-------
    from maceh.calculator import MACEHCalculator
    from ase.io import read

    calc = MACEHCalculator(model_dir='/path/to/trained_model')
    atoms = read('structure.cif')
    calc.calculate(atoms)
    H = calc.results['hamiltonian']
    # H is a dict: "[Rx, Ry, Rz, i, j]" -> torch.Tensor of shape [n_orb_i, n_orb_j]
"""

import os
import sys
import json
import itertools
import collections
from configparser import ConfigParser

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from ase.calculators.calculator import Calculator, all_changes

from .kernel import DatasetInfo, NetOutInfo
from .e3modules import e3TensorDecomp
from .utils import flt2cplx
from .from_pymatgen.lattice import find_neighbors, _compute_cube_index, _three_to_one


class MACEHCalculator(Calculator):
    """ASE Calculator that predicts Hamiltonians using a trained MACE-H model.

    The calculator loads a trained model from a checkpoint directory and
    constructs periodic neighbor lists to predict Hamiltonian matrix
    elements for any given structure.

    Parameters
    ----------
    model_dir : str
        Path to trained model directory (must contain ``best_model.pkl``
        and ``src/``).  Can also be a parent directory; the model will
        be located by recursive search.
    radius : float or dict or None
        Cutoff radius used for neighbor-list construction.

        - *None* (default): use the model's training cutoff for all pairs.
        - *float*: uniform cutoff override for every species pair.
        - *dict* ``{Z: r_Z, ...}``: per-species cutoffs keyed by atomic
          number.  The cutoff for a pair (Z_i, Z_j) is
          ``r_Zi + r_Zj``.  Species not listed fall back to the
          model's training cutoff.
    device : str
        Torch device, e.g. ``'cpu'`` or ``'cuda'``.
    dtype : str
        ``'float32'`` or ``'float64'``.
    debug : bool
        If True, fill unpredicted matrix elements with 0 instead of
        raising an error.
    num_threads : int or None
        If given, calls :func:`torch.set_num_threads` to clamp the
        number of CPU threads used by torch ops.  Unlike the
        ``-n``/``OMP_NUM_THREADS`` clamp in ``deephe3-eval.py``, this
        does **not** set the BLAS/OMP environment variables — those
        must be set before ``torch`` is imported to take effect, so
        clamp them in the user's launch script if BLAS-level threading
        also needs to be limited.
    uncertainty_method : str or None
        Method used to estimate Hamiltonian uncertainty.  When set, the
        chosen estimator is evaluated in :meth:`calculate` and stored
        under ``self.results['hamiltonian_uncertainty']``.

        - *None* (default): no uncertainty is computed.
        - ``'hermicity'``: element-wise Hermiticity error
          :math:`|H_{R,i,j} - H_{-R,j,i}^\\dagger|^p` per block
          (see :meth:`hermicity_error`).
    symmetrize : bool
        If True, the predicted Hamiltonian is Hermitised before being
        stored in ``self.results['hamiltonian']`` via
        :math:`H_{R,i,j} \\leftarrow \\tfrac12(H_{R,i,j}
        + H_{-R,j,i}^\\dagger)` (see :meth:`symmetrize_hamiltonian`).
        Can be overridden per call via the ``symmetrize`` argument of
        :meth:`calculate`.  Uncertainty estimators are always evaluated
        on the raw (non-symmetrised) Hamiltonian.
    load_model : bool
        If True, load the model and its metadata immediately in
        ``__init__``.  If False (default), loading is deferred until the
        first prediction (or metadata export) that actually needs it.
        Deferring the load is convenient in MPI setups where the
        calculator is instantiated on every rank but only a subset of
        ranks run predictions — those idle ranks then never pay the load
        cost.  The load happens exactly once and is cached thereafter.
    """

    implemented_properties = ['hamiltonian', 'hamiltonian_uncertainty']

    _UNCERTAINTY_METHODS = ('hermicity',)

    def __init__(self, model_dir, radius=None, device='cpu', dtype='float32',
                 debug=False, uncertainty_method=None, symmetrize=False,
                 num_threads=None, load_model=False, **kwargs):
        super().__init__(**kwargs)

        self.model_dir = model_dir
        self._radius = radius
        self.model_device = device
        self.debug = debug
        self.symmetrize = symmetrize
        self._model_loaded = False

        if num_threads is not None:
            torch.set_num_threads(int(num_threads))

        if uncertainty_method is not None \
                and uncertainty_method not in self._UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unknown uncertainty_method {uncertainty_method!r}. "
                f"Supported: {self._UNCERTAINTY_METHODS} or None."
            )
        self.uncertainty_method = uncertainty_method

        if dtype in ('float64', 'double'):
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
        else:
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32

        torch.set_default_dtype(self.torch_dtype)

        if load_model:
            self._ensure_model_loaded()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        """Load the model and per-pair cutoffs on first use.

        Idempotent: the network, its metadata and the cutoff matrix are
        loaded exactly once and reused on every subsequent call.  Loading
        is deferred until the first prediction / metadata export unless
        ``load_model=True`` was passed at construction time (see the class
        docstring).
        """
        if self._model_loaded:
            return
        self._load_model(self.model_dir)
        self._setup_cutoffs(self._radius)
        self._model_loaded = True

    @staticmethod
    def _find_model_path(model_dir):
        """Locate a model directory containing best_model.pkl and src/."""
        if (os.path.isfile(os.path.join(model_dir, 'best_model.pkl'))
                and os.path.isdir(os.path.join(model_dir, 'src'))):
            return model_dir
        for root, dirs, files in os.walk(model_dir):
            if 'best_model.pkl' in files and 'src' in dirs:
                return os.path.abspath(root)
        raise FileNotFoundError(
            f"Cannot find a trained model (best_model.pkl + src/) "
            f"under {model_dir}"
        )

    def _load_model(self, model_dir):
        """Load the trained MACE-H model and all required metadata."""
        model_path = self._find_model_path(model_dir)
        src_path = os.path.join(model_path, 'src')

        # --- dataset info (orbital types, element list, spinful flag) ---
        self.dataset_info = DatasetInfo.from_json(src_path)

        # --- target block definitions ---
        self.net_out_info = NetOutInfo.from_json(src_path)

        # --- config values from train.ini (without TrainConfig side effects) ---
        config = ConfigParser(inline_comment_prefixes=(';',))
        default_ini = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'default_configs/train_default.ini',
        )
        if os.path.isfile(default_ini):
            config.read(default_ini)
        config.read(os.path.join(src_path, 'train.ini'))

        self.cutoff_radius = config.getfloat('network', 'cutoff_radius')
        no_parity = config.getboolean('network', 'ignore_parity')
        convert_net_out = config.getboolean('target', 'convert_net_out')

        # --- e3TensorDecomp: converts raw edge output to H blocks ---
        self.construct_kernel = e3TensorDecomp(
            net_irreps_out=None,   # assertion skipped; irreps inferred from js
            out_js_list=self.net_out_info.js,
            default_dtype_torch=self.torch_dtype,
            spinful=self.dataset_info.spinful,
            no_parity=no_parity,
            if_sort=convert_net_out,
            device_torch=self.model_device,
        )

        # --- neural network ---
        sys.path.insert(0, src_path)
        try:
            from build_model import net  # auto-generated during training
        finally:
            sys.path.remove(src_path)
            sys.modules.pop('build_model', None)

        checkpoint = torch.load(
            os.path.join(model_path, 'best_model.pkl'),
            map_location='cpu',
            weights_only=False,
        )
        net.load_state_dict(checkpoint['state_dict'])
        net.to(self.model_device)
        net.eval()
        self.net = net

        # --- orbital bookkeeping ---
        self._atom_num_orbital = [
            sum(2 * l + 1 for l in orb_types)
            for orb_types in self.dataset_info.orbital_types
        ]

    def _setup_cutoffs(self, radius):
        """Build the per-species-pair cutoff matrix.

        Parameters
        ----------
        radius : float, dict, or None
            See ``__init__`` docstring.
        """
        n = len(self.dataset_info.index_to_Z)

        if radius is None:
            # uniform: model's training cutoff
            self._pair_cutoffs = torch.full(
                (n, n), self.cutoff_radius, dtype=self.torch_dtype
            )
        elif isinstance(radius, (int, float)):
            # uniform override
            self._pair_cutoffs = torch.full(
                (n, n), float(radius), dtype=self.torch_dtype
            )
        elif isinstance(radius, dict):
            # per-species; default to model cutoff for unlisted species
            species_r = torch.full(
                (n,), self.cutoff_radius, dtype=self.torch_dtype
            )
            for Z, r_Z in radius.items():
                idx = self.dataset_info.Z_to_index[int(Z)].item()
                if idx < 0:
                    raise ValueError(
                        f"Element Z={Z} not in model's training set "
                        f"{self.dataset_info.index_to_Z.tolist()}"
                    )
                species_r[idx] = float(r_Z)
            # pair cutoff = sum of the two species radii
            self._pair_cutoffs = (
                species_r.unsqueeze(0) + species_r.unsqueeze(1)
            )
        else:
            raise TypeError(
                f"radius must be None, float, or dict, got {type(radius)}"
            )

        self._r_max = self._pair_cutoffs.max().item()

    # ------------------------------------------------------------------
    # Graph construction from ASE Atoms
    # ------------------------------------------------------------------

    def _build_graph(self, atoms):
        """Build a PyG ``Data`` object with a periodic neighbor list.

        This replicates the ``create_from_DFT=False`` branch of
        :func:`maceh.graph.get_graph`, adding the extra fields
        (``atom_num_orbital``, ``spinful``, ``edge_key``) that the
        network and post-processing require.
        """
        dtype = self.torch_dtype
        r = self._r_max               # global max for bounding box / images
        numerical_tol = 1e-8

        cart_coords = torch.tensor(atoms.get_positions(), dtype=dtype)
        frac_coords = torch.tensor(atoms.get_scaled_positions(), dtype=dtype)
        lattice = torch.tensor(atoms.get_cell().array, dtype=dtype)
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

        # Map raw atomic numbers -> model type indices
        x = self.dataset_info.Z_to_index[numbers].clone()
        if torch.any(x < 0):
            unknown = numbers[x < 0].unique().tolist()
            known = self.dataset_info.index_to_Z.tolist()
            raise ValueError(
                f"Structure contains elements {unknown} not in the "
                f"model's training set {known}"
            )

        num_atom = cart_coords.shape[0]

        # --- determine bounding box for image filtering ---
        cart_np = cart_coords.numpy()
        frac_np = frac_coords.numpy()
        lat_np = lattice.numpy()

        global_min = np.min(cart_np, axis=0) - r - numerical_tol
        global_max = np.max(cart_np, axis=0) + r + numerical_tol
        global_min_t = torch.tensor(global_min)
        global_max_t = torch.tensor(global_max)

        # --- enumerate required lattice images ---
        recip = np.linalg.inv(lat_np).T * 2 * np.pi
        recp_len = np.sqrt(np.sum(recip ** 2, axis=1))
        maxr = np.ceil((r + 0.15) * recp_len / (2 * np.pi))
        nmin = np.floor(np.min(frac_np, axis=0)) - maxr
        nmax = np.ceil(np.max(frac_np, axis=0)) + maxr
        all_ranges = [np.arange(lo, hi, dtype='int64')
                      for lo, hi in zip(nmin, nmax)]
        images = torch.tensor(
            list(itertools.product(*all_ranges))
        ).type_as(lattice)

        # --- compute all image coordinates and filter ---
        coords = (images @ lattice)[:, None, :] + cart_coords[None, :, :]
        indices = torch.arange(num_atom).unsqueeze(0).expand(
            images.shape[0], num_atom
        )
        valid = (coords.gt(global_min_t) * coords.lt(global_max_t)).all(dim=-1)
        valid_coords = coords[valid]
        valid_indices = indices[valid]

        # --- spatial hashing into cubes ---
        valid_np = valid_coords.detach().numpy()
        all_cube = _compute_cube_index(valid_np, global_min, r)
        nx, ny, nz = _compute_cube_index(global_max, global_min, r) + 1
        all_cube_1d = _three_to_one(all_cube, ny, nz)
        site_cube = _three_to_one(
            _compute_cube_index(cart_np, global_min, r), ny, nz
        )

        cube_map = collections.defaultdict(list)
        for idx, cube_id in enumerate(all_cube_1d.ravel()):
            cube_map[cube_id].append(idx)

        site_nbrs = find_neighbors(site_cube, nx, ny, nz)

        # --- find edges within cutoff for each center atom ---
        inv_lattice = torch.inverse(lattice).type(dtype)
        edge_idx_src, edge_idx_dst = [], []
        edge_fea_list, edge_key_list = [], []

        for i_center, (cart_i, nbr_cubes) in enumerate(
            zip(cart_coords, site_nbrs)
        ):
            l1 = np.array(_three_to_one(nbr_cubes, ny, nz), dtype=int).ravel()
            ks = [k for k in l1 if k in cube_map]
            nn_idx = np.concatenate([cube_map[k] for k in ks], axis=0)
            nn_coords = valid_coords[nn_idx]
            nn_indices = valid_indices[nn_idx]
            dist = torch.norm(nn_coords - cart_i[None, :], dim=1)

            nn_coords = nn_coords.squeeze()
            nn_indices = nn_indices.squeeze()
            dist = dist.squeeze()

            # R-vector: integer lattice translation
            R = torch.round(
                (nn_coords - cart_coords[nn_indices]) @ inv_lattice
            ).int()
            ek = torch.cat([
                R,
                torch.full([R.shape[0], 1], i_center, dtype=torch.int) + 1,
                nn_indices.unsqueeze(1).int() + 1,
            ], dim=1)

            # per-pair cutoff: r_ij = max(r_Zi, r_Zj)
            type_j = x[nn_indices.long()]
            r_pair = self._pair_cutoffs[x[i_center], type_j]
            mask = dist.lt(r_pair + numerical_tol)
            edge_idx_src.extend([i_center] * int(mask.sum()))
            edge_idx_dst.extend(nn_indices[mask].tolist())
            edge_fea_list.append(torch.cat([
                dist[mask].view(-1, 1),
                nn_coords[mask] - cart_i,
            ], dim=-1))
            edge_key_list.append(ek[mask])

        edge_fea = torch.cat(edge_fea_list).type(dtype)
        edge_index = torch.stack([
            torch.LongTensor(edge_idx_src),
            torch.LongTensor(edge_idx_dst),
        ])
        edge_key = torch.cat(edge_key_list, dim=0)

        atom_num_orbital = torch.tensor(
            [self._atom_num_orbital[xi] for xi in x]
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_fea,
            pos=cart_coords.type(dtype),
            lattice=lattice.unsqueeze(0),
            edge_key=edge_key,
            Aij=None,
            Aij_mask=None,
            atom_num_orbital=atom_num_orbital,
            spinful=self.dataset_info.spinful,
            stru_id='ase_atoms',
        )

    # ------------------------------------------------------------------
    # Hamiltonian assembly from network output
    # ------------------------------------------------------------------

    def _update_hopping(self, H_pred, node_attr, edge_index, edge_key):
        """Convert the raw network prediction into a Hamiltonian dict.

        Adapted from :meth:`DeepHE3Kernel.update_hopping`.

        Returns
        -------
        dict
            Keys are ``"[Rx, Ry, Rz, i, j]"`` strings (1-based atom
            indices).  Values are ``torch.Tensor`` of shape
            ``[n_orb_i, n_orb_j]`` (or ``[2*n_orb_i, 2*n_orb_j]``
            when spinful).
        """
        dtype = self.torch_dtype
        spinful = self.dataset_info.spinful
        atom_num_orb = self._atom_num_orbital
        index_to_Z = self.dataset_info.index_to_Z

        H_dict = {}

        for ie in range(edge_index.shape[1]):
            key_str = str(edge_key[ie].tolist())
            i_type, j_type = node_attr[edge_index[:, ie]]

            # initialise matrix for this (R, i, j) pair
            if key_str not in H_dict:
                fill = 0 if self.debug else np.nan
                fill_c = complex(fill, fill) if np.isnan(fill) else 0 + 0j
                if spinful:
                    H_dict[key_str] = torch.full(
                        (atom_num_orb[i_type] * 2,
                         atom_num_orb[j_type] * 2),
                        fill_c, dtype=flt2cplx(dtype),
                    )
                else:
                    H_dict[key_str] = torch.full(
                        (atom_num_orb[i_type], atom_num_orb[j_type]),
                        fill, dtype=dtype,
                    )

            Z_pair = (f'{index_to_Z[i_type].item()} '
                      f'{index_to_Z[j_type].item()}')

            for it, block in enumerate(self.net_out_info.blocks):
                for nm_str, bslice in block.items():
                    if nm_str != Z_pair:
                        continue
                    sr = slice(bslice[0], bslice[1])
                    sc = slice(bslice[2], bslice[3])
                    lr = bslice[1] - bslice[0]
                    lc = bslice[3] - bslice[2]
                    so = slice(self.net_out_info.slices[it],
                               self.net_out_info.slices[it + 1])

                    if spinful:
                        sr_ds = slice(atom_num_orb[i_type] + bslice[0],
                                      atom_num_orb[i_type] + bslice[1])
                        sc_ds = slice(atom_num_orb[j_type] + bslice[2],
                                      atom_num_orb[j_type] + bslice[3])
                        H_dict[key_str][sr, sc] = \
                            H_pred[ie, 0, so].reshape(lr, lc)
                        H_dict[key_str][sr, sc_ds] = \
                            H_pred[ie, 1, so].reshape(lr, lc)
                        H_dict[key_str][sr_ds, sc] = \
                            H_pred[ie, 2, so].reshape(lr, lc)
                        H_dict[key_str][sr_ds, sc_ds] = \
                            H_pred[ie, 3, so].reshape(lr, lc)
                    else:
                        H_dict[key_str][sr, sc] = \
                            H_pred[ie, so].reshape(lr, lc)

        return H_dict

    # ------------------------------------------------------------------
    # Hermiticity analysis
    # ------------------------------------------------------------------

    @staticmethod
    def symmetrize_hamiltonian(H_dict):
        """Return a Hermitised copy of the Hamiltonian dictionary.

        For each key ``[Rx, Ry, Rz, i, j]``, replaces the block with

        .. math::

            H_{R,i,j} \\leftarrow \\tfrac12
            \\bigl(H_{R,i,j} + H_{-R,j,i}^\\dagger\\bigr).

        Parameters
        ----------
        H_dict : dict
            Hamiltonian dictionary as returned by :meth:`calculate`.

        Returns
        -------
        dict
            Same keys as *H_dict*; values are the symmetrised blocks.
        """
        sym_dict = {}
        for key_str, H_block in H_dict.items():
            Rx, Ry, Rz, i, j = json.loads(key_str)
            conj_key = str([-Rx, -Ry, -Rz, j, i])
            if conj_key not in H_dict:
                raise KeyError(
                    f"Conjugate key {conj_key} not found for {key_str}. "
                    "The neighbor list should be symmetric."
                )
            sym_dict[key_str] = 0.5 * (H_block + H_dict[conj_key].T.conj())
        return sym_dict

    @staticmethod
    def hermicity_error(H_dict, p=1):
        """Element-wise Hermiticity error for each block in the Hamiltonian.

        For each key ``[Rx, Ry, Rz, i, j]``, computes

        .. math::

            |H_{R,i,j} - H_{-R,j,i}^\\dagger|^p

        element-wise, where :math:`\\dagger` denotes conjugate transpose
        (reduces to transpose for real / non-spinful Hamiltonians).

        Parameters
        ----------
        H_dict : dict
            Hamiltonian dictionary as returned by :meth:`calculate`.
        p : int or float
            Exponent for the element-wise error.  Default 2.

        Returns
        -------
        dict
            Same keys as *H_dict*.  Values are real-valued tensors of the
            same shape containing the element-wise error.
        """
        error_dict = {}
        for key_str, H_block in H_dict.items():
            Rx, Ry, Rz, i, j = json.loads(key_str)
            conj_key = str([-Rx, -Ry, -Rz, j, i])
            if conj_key not in H_dict:
                raise KeyError(
                    f"Conjugate key {conj_key} not found for {key_str}. "
                    "The neighbor list should be symmetric."
                )
            H_conj = H_dict[conj_key]
            error_dict[key_str] = torch.abs(H_block - H_conj.T.conj()) ** p
        return error_dict

    @staticmethod
    def mean_hermicity_error(error_dict):
        """Mean Hermiticity error over all blocks.

        Every non-trivially-zero error value appears exactly twice in the
        error dictionary: off-diagonal key pairs ``(R, i, j)`` and
        ``(-R, j, i)`` carry duplicate blocks, and within on-site blocks
        ``(R=0, i=i)`` the off-diagonal elements satisfy
        ``err[a, b] = err[b, a]``.  The only non-duplicated entries are
        the diagonal elements of on-site blocks, which are zero by
        construction.  Since the duplication factor is the same in both
        the numerator (sum) and denominator (count), it cancels, giving

        .. math::

            \\frac{\\sum_{\\text{all keys, all elements}} \\mathrm{err}}
                  {N_{\\text{total}} - N_{\\text{on-site diag}}}

        Parameters
        ----------
        error_dict : dict
            Output of :meth:`hermicity_error`.

        Returns
        -------
        float
            Mean Hermiticity error.
        """
        total_sum = 0.0
        total_count = 0
        on_site_diag_count = 0

        for key_str, err in error_dict.items():
            Rx, Ry, Rz, i, j = json.loads(key_str)
            total_sum += err.sum().item()
            total_count += err.numel()
            if Rx == 0 and Ry == 0 and Rz == 0 and i == j:
                on_site_diag_count += err.shape[0]

        denominator = total_count - on_site_diag_count
        if denominator == 0:
            return 0.0
        return total_sum / denominator

    # ------------------------------------------------------------------
    # ASE Calculator interface
    # ------------------------------------------------------------------

    def calculate(self, atoms=None, properties=None, system_changes=all_changes,
                  symmetrize=None):
        """Predict the Hamiltonian for ``atoms`` and store it in ``results``.

        Parameters
        ----------
        atoms, properties, system_changes
            Standard ASE Calculator arguments.
        symmetrize : bool or None
            Per-call override of ``self.symmetrize``.  If *None*
            (default), uses the value set at construction time.  When
            True the stored Hamiltonian is Hermitised; uncertainty
            estimators are always evaluated on the raw prediction.
        """
        self._ensure_model_loaded()

        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        if symmetrize is None:
            symmetrize = self.symmetrize

        data = self._build_graph(self.atoms)
        batch = Batch.from_data_list([data])

        with torch.no_grad():
            _node_out, edge_out = self.net(batch.to(device=self.model_device))
            H_pred = self.construct_kernel.get_H(edge_out).cpu()

        H_dict = self._update_hopping(
            H_pred,
            batch.x.cpu(),
            batch.edge_index.cpu(),
            batch.edge_key.cpu(),
        )

        if not self.debug:
            for key, val in H_dict.items():
                has_nan = torch.any(torch.isnan(val))
                if has_nan:
                    raise RuntimeError(
                        f"Unpredicted orbital blocks remain for edge {key}. "
                        "Use debug=True to fill them with 0."
                    )

        # Uncertainty must be evaluated on the raw H — after symmetrisation
        # the hermicity error is exactly zero by construction.
        if self.uncertainty_method is not None:
            self.results['hamiltonian_uncertainty'] = \
                self._compute_uncertainty(H_dict)

        if symmetrize:
            H_dict = self.symmetrize_hamiltonian(H_dict)

        self.results['hamiltonian'] = H_dict

    def _compute_uncertainty(self, H_dict):
        """Dispatch to the configured uncertainty estimator."""
        if self.uncertainty_method == 'hermicity':
            return self.hermicity_error(H_dict)
        raise ValueError(
            f"Unknown uncertainty_method {self.uncertainty_method!r}."
        )

    def get_hamiltonian(self, atoms=None, symmetrize=None):
        """Convenience wrapper: predict and return the Hamiltonian dict.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Structure to predict.  If *None*, reuses the last-set atoms.
        symmetrize : bool or None
            Per-call override of ``self.symmetrize`` (see
            :meth:`calculate`).

        Returns
        -------
        dict
            Hamiltonian dictionary.  Keys are ``"[Rx, Ry, Rz, i, j]"``
            strings with **1-based** atom indices.  Values are tensors
            of shape ``[n_orb_i, n_orb_j]``.
        """
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No atoms provided and none previously set.")
            atoms = self.atoms
        self.calculate(atoms, symmetrize=symmetrize)
        return self.results['hamiltonian']

    # ------------------------------------------------------------------
    # DeepH-E3 metadata export
    # ------------------------------------------------------------------

    def write_metadata(self, output_dir, atoms=None):
        """Write DeepH-E3 / MACE-H metadata files for a structure.

        Produces the six metadata files that the preprocessing pipeline
        emits alongside ``hamiltonians_pred.h5`` and ``overlaps.h5``:

        - ``info.json``          — ``{"isspinful": bool}``
        - ``lat.dat``            — real-space lattice, 3x3
        - ``rlat.dat``           — reciprocal lattice (with the 2*pi
          factor), 3x3
        - ``element.dat``        — atomic numbers, one per atom
        - ``site_positions.dat`` — Cartesian positions, 3xN
        - ``orbital_types.dat``  — per-atom orbital angular momenta

        Additionally writes ``hamiltonians_pred.h5`` and
        ``hamiltonians_uncertainty.h5``, each containing one **empty**
        HDF5 dataset per ``"[Rx, Ry, Rz, i, j]"`` edge key, exposing the
        sparsity pattern of the predicted Hamiltonian / uncertainty
        without storing any matrix elements.  Each sparsity file is only
        written when no file of that name already exists in *output_dir*
        — an existing file (e.g. one written by :meth:`write_results`
        with actual values) is left untouched.

        Lattice vectors and atomic coordinates are written as **columns**
        (i.e. transposed relative to ASE's row-wise cell convention),
        matching :func:`maceh.data.AijData.process_worker` which loads
        them with ``np.loadtxt(...).T``.

        Parameters
        ----------
        output_dir : str
            Destination directory.  Created if it does not exist.
        atoms : ase.Atoms, optional
            Structure to export.  If *None*, uses the atoms currently
            attached to the calculator.
        """
        self._ensure_model_loaded()

        if atoms is None:
            if self.atoms is None:
                raise ValueError("No atoms provided and none previously set.")
            atoms = self.atoms

        os.makedirs(output_dir, exist_ok=True)

        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        indices = self.dataset_info.Z_to_index[torch.from_numpy(numbers)]
        if torch.any(indices < 0):
            unknown = sorted({int(z) for z, i in zip(numbers, indices.tolist())
                              if i < 0})
            known = self.dataset_info.index_to_Z.tolist()
            raise ValueError(
                f"Structure contains elements {unknown} not in the "
                f"model's training set {known}"
            )

        with open(os.path.join(output_dir, 'info.json'), 'w') as f:
            json.dump({'isspinful': bool(self.dataset_info.spinful)},
                      f, indent=4)

        L = np.asarray(atoms.get_cell().array, dtype=np.float64)
        np.savetxt(os.path.join(output_dir, 'lat.dat'), L.T, delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rlat.dat'),
                   2.0 * np.pi * np.linalg.inv(L), delimiter='\t')

        np.savetxt(os.path.join(output_dir, 'element.dat'), numbers, fmt='%d')

        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        np.savetxt(os.path.join(output_dir, 'site_positions.dat'),
                   positions.T, delimiter='\t')

        with open(os.path.join(output_dir, 'orbital_types.dat'), 'w') as f:
            for idx in indices.tolist():
                orbs = self.dataset_info.orbital_types[idx]
                f.write('\t'.join(str(int(l)) for l in orbs) + '\n')

        # Sparsity-only HDF5 files: one empty dataset per edge key, for both
        # the predicted Hamiltonian and its uncertainty.  Never clobber an
        # existing file — it may already hold real data.
        pred_path = os.path.join(output_dir, 'hamiltonians_pred.h5')
        unc_path = os.path.join(output_dir, 'hamiltonians_uncertainty.h5')
        need_pred = not os.path.exists(pred_path)
        need_unc = not os.path.exists(unc_path)
        if need_pred or need_unc:
            graph = self._build_graph(atoms)
            keys = {str(graph.edge_key[ie].tolist())
                    for ie in range(graph.edge_key.shape[0])}
            if self.dataset_info.spinful:
                h5_dtype = (np.complex64 if self.np_dtype == np.float32
                            else np.complex128)
            else:
                h5_dtype = self.np_dtype
            if need_pred:
                with h5py.File(pred_path, 'w') as f:
                    for key in keys:
                        f[key] = h5py.Empty(h5_dtype)
            if need_unc:
                # Uncertainty is real-valued even for spinful Hamiltonians.
                with h5py.File(unc_path, 'w') as f:
                    for key in keys:
                        f[key] = h5py.Empty(self.np_dtype)

    def write_results(self, output_dir, atoms=None, metadata=True, data=True,
                      uncertainty=False):
        """Write predicted Hamiltonian, uncertainty and/or metadata to disk.

        The destination folder mirrors the DeepH-E3 preprocessed-data
        layout: metadata files (``info.json``, ``lat.dat``, ``rlat.dat``,
        ``element.dat``, ``site_positions.dat``, ``orbital_types.dat``)
        together with ``hamiltonians_pred.h5`` holding the predicted matrix
        blocks and ``hamiltonians_uncertainty.h5`` holding the per-block
        uncertainty.  Each dataset in the HDF5 files is keyed by the
        ``"[Rx, Ry, Rz, i, j]"`` string used everywhere else in the
        pipeline (1-based atom indices).

        Parameters
        ----------
        output_dir : str
            Destination directory.  Created if missing.
        atoms : ase.Atoms, optional
            Structure to export.  If *None*, uses the atoms currently
            attached to the calculator.
        metadata : bool
            If True, write the six metadata files (see
            :meth:`write_metadata`).
        data : bool
            If True, compute (or reuse) the predicted Hamiltonian and
            write it to ``hamiltonians_pred.h5``.
        uncertainty : bool
            If True, compute (or reuse) the per-block uncertainty and
            write it to ``hamiltonians_uncertainty.h5``.  Requires the
            calculator to have been constructed with a non-None
            ``uncertainty_method``.

        Notes
        -----
        Set ``metadata=False`` to write only the data / uncertainty files
        with real values, or ``data=False`` / ``uncertainty=False`` to skip
        those.  When ``metadata=True`` but a value file is skipped, that
        file is still produced by :meth:`write_metadata` but contains empty
        datasets exposing just the sparsity pattern.  When both metadata and
        a value flag are True, :meth:`write_metadata` runs first (and leaves
        any existing files alone), then the value branch overwrites the
        relevant ``.h5`` file with real blocks.  All three flags False is an
        error.
        """
        if not (metadata or data or uncertainty):
            raise ValueError(
                "Nothing to write: set metadata, data and/or uncertainty "
                "to True."
            )
        if uncertainty and self.uncertainty_method is None:
            raise ValueError(
                "uncertainty=True requires the calculator to be constructed "
                "with a non-None uncertainty_method."
            )
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No atoms provided and none previously set.")
            atoms = self.atoms

        os.makedirs(output_dir, exist_ok=True)

        if metadata:
            self.write_metadata(output_dir, atoms=atoms)

        # A single calculate() populates both the Hamiltonian and (when an
        # uncertainty_method is set) the uncertainty in self.results.
        if data or uncertainty:
            self.calculate(atoms)

        if data:
            H_dict = self.results['hamiltonian']
            h5_path = os.path.join(output_dir, 'hamiltonians_pred.h5')
            with h5py.File(h5_path, 'w') as f:
                for key, block in H_dict.items():
                    if isinstance(block, torch.Tensor):
                        block = block.detach().cpu().numpy()
                    f[key] = np.asarray(block)

        if uncertainty:
            unc_dict = self.results['hamiltonian_uncertainty']
            h5_path = os.path.join(output_dir, 'hamiltonians_uncertainty.h5')
            with h5py.File(h5_path, 'w') as f:
                for key, block in unc_dict.items():
                    if isinstance(block, torch.Tensor):
                        block = block.detach().cpu().numpy()
                    f[key] = np.asarray(block)

    def get_hamiltonian_uncertainty(self, atoms=None):
        """Convenience wrapper: predict and return the uncertainty dict.

        Requires the calculator to have been constructed with a non-None
        ``uncertainty_method``.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Structure to predict.  If *None*, reuses the last-set atoms.

        Returns
        -------
        dict
            Same keys as the Hamiltonian dict; values are the
            uncertainty per block as defined by the configured method.
        """
        if self.uncertainty_method is None:
            raise ValueError(
                "No uncertainty_method was set on the calculator. "
            )
        if atoms is not None:
            self.calculate(atoms)
        elif self.atoms is not None:
            self.calculate(self.atoms)
        else:
            raise ValueError("No atoms provided and none previously set.")
        return self.results['hamiltonian_uncertainty']
