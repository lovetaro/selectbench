import os
import contextlib
import multiprocessing
import multiprocessing.pool
from multiprocessing import Process, Queue
from vina import Vina
from selectbench.data.ligand_prep import prepare_ligand


class _NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonContext(type(multiprocessing.get_context())):
    Process = _NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    """A Pool whose workers are non-daemonic, so they can spawn child processes."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = _NoDaemonContext()
        super().__init__(*args, **kwargs)

# Per-molecule docking timeout in seconds. 600 matches the paper protocol
# and scripts/run_eval.py's --timeout default, so a direct `selectbench
# evaluate-*` call gets the same budget as the batch layer.
DOCK_TIMEOUT = int(os.environ.get("SELECTBENCH_DOCK_TIMEOUT", 600))

# Vina's Monte Carlo search is stochastic, and ``Vina(seed=0)`` means "pick a
# random seed", so an unseeded run gives different scores every time. Pin it so
# results are reproducible. Override for seed sweeps:
# SELECTBENCH_DOCK_SEED=<int>. Note this only guarantees reproducibility at
# ``cpu=1`` (the default); with cpu>1 Vina's threads race and scores still drift.
DOCK_SEED = int(os.environ.get("SELECTBENCH_DOCK_SEED", 42))


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect fd 2 (stderr) to /dev/null to suppress C-level Vina warnings."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


def _prep_ligand_worker(args):
    """Multiprocessing worker: prepare a ligand from SMILES to PDBQT string."""
    mol_idx, smiles = args
    try:
        pdbqt = prepare_ligand(smiles)
        return {"mol_idx": mol_idx, "smiles": smiles, "pdbqt": pdbqt, "error": None}
    except Exception as e:
        return {"mol_idx": mol_idx, "smiles": smiles, "pdbqt": None, "error": str(e)}


def _run_vina_subprocess(queue, ligand_pdbqt, receptor_pdbqt, box_file,
                         output_pdbqt, exhaustiveness, n_poses, save_poses, cpu):
    """Subprocess target: run vina and put result on queue."""
    try:
        out = run_vina(ligand_pdbqt, receptor_pdbqt, box_file,
                       output_pdbqt=output_pdbqt, exhaustiveness=exhaustiveness,
                       n_poses=n_poses, save_poses=save_poses, timeout=None, cpu=cpu)
        queue.put(("ok", out))
    except Exception as e:
        queue.put(("error", str(e)))


def run_vina_safe(ligand_pdbqt, receptor_pdbqt, box_file,
                  output_pdbqt=None, exhaustiveness=20, n_poses=5,
                  save_poses=True, timeout=DOCK_TIMEOUT, cpu=1):
    """Run vina with a timeout. Spawns a subprocess to allow killing hung docking.

    See ``run_vina`` for ``cpu`` semantics.
    """
    if timeout is None or timeout <= 0:
        return run_vina(ligand_pdbqt, receptor_pdbqt, box_file,
                        output_pdbqt=output_pdbqt, exhaustiveness=exhaustiveness,
                        n_poses=n_poses, save_poses=save_poses, timeout=None, cpu=cpu)
    queue = Queue()
    p = Process(target=_run_vina_subprocess,
                args=(queue, ligand_pdbqt, receptor_pdbqt, box_file,
                      output_pdbqt, exhaustiveness, n_poses, save_poses, cpu))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        raise TimeoutError(f"Docking timed out after {timeout}s")
    if queue.empty():
        raise RuntimeError("Docking subprocess died unexpectedly")
    status, value = queue.get_nowait()
    if status == "ok":
        return value
    else:
        raise RuntimeError(value)


def _dock_pdbqt_worker(args):
    """Multiprocessing worker: dock a prepared PDBQT string against one target.

    Tuple layout:
      5-tuple: (mol_idx, pdbqt, receptor, box_file, target_label)
      6-tuple: ... + exhaustiveness
      7-tuple: ... + exhaustiveness + timeout      (used by fallback retry)
    """
    timeout = None
    if len(args) == 7:
        mol_idx, ligand_pdbqt, receptor, box_file, target_label, exhaustiveness, timeout = args
    elif len(args) == 6:
        mol_idx, ligand_pdbqt, receptor, box_file, target_label, exhaustiveness = args
    else:
        mol_idx, ligand_pdbqt, receptor, box_file, target_label = args
        exhaustiveness = 20
    if not os.path.exists(receptor):
        return {"mol_idx": mol_idx, "target_label": target_label, "score": None,
                "error": f"Missing receptor: {receptor}"}
    if not os.path.exists(box_file):
        return {"mol_idx": mol_idx, "target_label": target_label, "score": None,
                "error": f"Missing box file: {box_file}"}
    try:
        kwargs = {"exhaustiveness": exhaustiveness}
        if timeout is not None:
            kwargs["timeout"] = timeout
        out = run_vina_safe(ligand_pdbqt, receptor, box_file, **kwargs)
        score = float(out[0][0])
        return {"mol_idx": mol_idx, "target_label": target_label, "score": score, "error": None}
    except Exception as e:
        return {"mol_idx": mol_idx, "target_label": target_label, "score": None, "error": str(e)}


def _dock_single_target(args):
    """
    Worker function for multiprocessing. Docks a ligand against a single safety target.
    Returns a dict: {"case_name": str, "score": float or None, "error": str or None}
    """
    # 5-tuple adds an exhaustiveness override; the 4-tuple form keeps
    # run_vina_safe's default for callers that predate it.
    if len(args) == 5:
        case_row, processed_base_dir, save_poses_dir, ligand_pdbqt, exhaustiveness = args
    else:
        case_row, processed_base_dir, save_poses_dir, ligand_pdbqt = args
        exhaustiveness = None

    pdb_id = case_row["pdb_id"]
    chain_id = case_row["pdb_chain_id"]

    # Parse chain(s): "[A;B;C;D]" -> "A_B_C_D"
    if isinstance(chain_id, str) and chain_id.startswith("[") and chain_id.endswith("]"):
        chain_id = chain_id[1:-1].replace(";", "_")

    case_name = case_row["case_name"]
    processed_dir = os.path.join(processed_base_dir, "processed", case_name)

    receptor = os.path.join(processed_dir, f"{pdb_id}_{chain_id}.pdbqt")
    box_file = os.path.join(processed_dir, f"{pdb_id}_{chain_id}.box.txt")

    if not os.path.exists(receptor):
        return {"case_name": case_name, "score": None, "error": f"Missing receptor: {receptor}"}

    if not os.path.exists(box_file):
        return {"case_name": case_name, "score": None, "error": f"Missing box file: {box_file}"}

    docked_pose_path = None
    if save_poses_dir:
        docked_pose_path = os.path.join(
            save_poses_dir, f"docked_{case_name}_{pdb_id}_{chain_id}.pdbqt"
        )

    try:
        dock_kwargs = {} if exhaustiveness is None else {"exhaustiveness": exhaustiveness}
        vina_output = run_vina_safe(ligand_pdbqt, receptor, box_file,
                                    docked_pose_path, **dock_kwargs)
        score = float(vina_output[0][0])
        return {"case_name": case_name, "score": score, "error": None}
    except Exception as e:
        return {"case_name": case_name, "score": None, "error": str(e)}


def run_vina(
    ligand_pdbqt,
    receptor_pdbqt,
    box_file,
    output_pdbqt=None,
    exhaustiveness=20,
    n_poses=5,
    save_poses=True,
    timeout=None,
    cpu=1,
    seed=None,
):
    """Run docking using Meeko/Vina.

    ``cpu`` is forwarded to ``Vina(cpu=...)``. Default is 1 to avoid CPU
    oversubscription when this is called from inside a multiprocessing worker
    (which is the common case — each Vina() otherwise autodetects and tries to
    use every core, so N workers × C cores = N·C threads on C cores). Callers
    running vina directly in a single process can pass ``cpu=0`` for max speed.
    ``timeout`` param is ignored here (used by run_vina_safe).

    ``seed`` seeds Vina's Monte Carlo search; it defaults to ``DOCK_SEED`` so
    repeated runs reproduce. Reproducibility holds only at ``cpu=1``.
    """
    with open(box_file) as f:
        box_lines = f.readlines()

    # parse box
    box = {}
    for line in box_lines:
        k, v = line.strip().split("=")
        box[k.strip()] = float(v.strip())

    vina = Vina(cpu=cpu, seed=DOCK_SEED if seed is None else seed, verbosity=0)
    vina.set_receptor(receptor_pdbqt)
    vina.set_ligand_from_string(ligand_pdbqt)
    vina.compute_vina_maps(
        center=[box["center_x"], box["center_y"], box["center_z"]],
        box_size=[box["size_x"], box["size_y"], box["size_z"]],
    )
    with _suppress_stderr():
        vina.dock(exhaustiveness=exhaustiveness)

    out = vina.energies(n_poses=n_poses)

    if save_poses and output_pdbqt is not None:
        vina.write_poses(output_pdbqt, n_poses=n_poses, overwrite=True)
        print(f"Docked pose saved: {output_pdbqt}")

    return out
