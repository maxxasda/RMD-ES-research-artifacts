import numpy as np
import pandas as pd
from simulation import SIMULATION
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import logger
from multiprocessing import Pool, Manager
from multiprocessing import TimeoutError as MPTimeoutError

log = logger.get_logger(__name__)


def sim_task(x, step, sim_id, oasis_sim, shared_dict, progress_total=None, progress_offset=0):
    try:
        fit = oasis_sim.sim_exec(
            x,
            step,
            save_flag=True,
            progress_total=progress_total,
            progress_offset=progress_offset,
        )
        shared_dict[sim_id] = {"ok": True, "fit": fit, "error": ""}
    except Exception as exc:
        shared_dict[sim_id] = {"ok": False, "fit": None, "error": repr(exc)}


class SotifProblem(Problem):
    def __init__(self, config, parse_scenario):
        self.step = 0
        self.parse_scenario = parse_scenario
        self.config = config

        carla_parames = config.sim_task["carla_parames"]
        task_num = len(carla_parames)

        self.oasis_simulations = {}
        for i in range(task_num):
            self.oasis_simulations[i] = SIMULATION(self.config, self.parse_scenario, carla_parames[i])
        self.sim_nums = task_num

        super().__init__(
            n_var=config.n_var,
            n_obj=config.n_obj,
            n_constr=0,
            xl=np.array(config.xl),
            xu=np.array(config.xu),
        )

    def _build_penalty_fit(self, rows):
        if rows <= 0:
            return np.empty((0, self.n_obj), dtype=float)
        return np.full((rows, self.n_obj), 1e6, dtype=float)

    def _normalize_fit_block(self, payload, expected_rows):
        if payload is None:
            log.warning(f"[step={self.step}] missing payload, fallback rows={expected_rows}")
            return self._build_penalty_fit(expected_rows)

        raw_fit = payload
        if isinstance(payload, dict) and ("ok" in payload or "fit" in payload):
            ok = bool(payload.get("ok", False))
            if not ok:
                log.warning(f"[step={self.step}] worker returned error: {payload.get('error', '')}")
                return self._build_penalty_fit(expected_rows)
            raw_fit = payload.get("fit", None)

        if raw_fit is None:
            return self._build_penalty_fit(expected_rows)

        rows = []
        if isinstance(raw_fit, dict):
            for key in raw_fit:
                rows += raw_fit[key]
        else:
            rows = raw_fit

        try:
            arr = np.asarray(rows, dtype=float)
        except Exception as exc:
            log.warning(f"[step={self.step}] fit cast failed: {repr(exc)}")
            return self._build_penalty_fit(expected_rows)

        if arr.size == 0:
            return self._build_penalty_fit(expected_rows)

        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            if self.n_obj == 1:
                arr = arr.reshape(-1, 1)
            else:
                arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[1] < self.n_obj:
            pad = np.full((arr.shape[0], self.n_obj - arr.shape[1]), 1e6, dtype=float)
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > self.n_obj:
            arr = arr[:, :self.n_obj]

        if expected_rows <= 0:
            return np.empty((0, self.n_obj), dtype=float)

        if arr.shape[0] < expected_rows:
            arr = np.vstack([arr, self._build_penalty_fit(expected_rows - arr.shape[0])])
        elif arr.shape[0] > expected_rows:
            arr = arr[:expected_rows, :]

        return arr

    def _evaluate(self, x, out, *args, **kwargs):
        fit_blocks = []
        task_meta = []
        async_results = []

        with Manager() as manager:
            shared_dict = manager.dict()

            with Pool(processes=self.sim_nums) as pool:
                has_worker_failure = False
                # Timeout is per async task, not per entire generation.
                worker_timeout_s = float(getattr(self.config, "worker_eval_timeout_s",480.0))
                chunk_size = int(getattr(self.config, "eval_chunk_size", 5))
                chunk_size = max(1, chunk_size)

                sim_ids = list(self.oasis_simulations.keys())
                if len(sim_ids) == 0:
                    out["F"] = self._build_penalty_fit(x.shape[0])
                    self.step += 1
                    return

                cursor = 0
                task_id = 0
                while cursor < x.shape[0]:
                    l_idx = int(cursor)
                    r_idx = int(min(cursor + chunk_size, x.shape[0]))
                    x_slice = x[l_idx:r_idx]

                    sim_id = sim_ids[task_id % len(sim_ids)]
                    oasis_sim = self.oasis_simulations[sim_id]

                    # task_id is the shared_dict key, sim_id is only executor affinity.
                    task_meta.append((task_id, l_idx, r_idx))

                    async_res = pool.apply_async(
                        sim_task,
                        args=(x_slice, self.step, task_id, oasis_sim, shared_dict, int(x.shape[0]), l_idx),
                    )
                    async_results.append((task_id, async_res))

                    cursor = r_idx
                    task_id += 1

                # Make worker exceptions visible instead of silently failing.
                for task_id, async_res in async_results:
                    try:
                        async_res.get(timeout=worker_timeout_s)
                    except MPTimeoutError:
                        has_worker_failure = True
                        log.error(
                            f"[step={self.step}] task {task_id} timeout>{worker_timeout_s}s, terminate pool and fallback."
                        )
                        pool.terminate()
                        break
                    except Exception as exc:
                        has_worker_failure = True
                        log.error(f"[step={self.step}] task {task_id} raised: {repr(exc)}")
                        pool.terminate()
                        break

                if not has_worker_failure:
                    pool.close()
                pool.join()

            shared_keys = sorted(list(shared_dict.keys()))
            if len(shared_keys) != len(task_meta):
                log.warning(
                    f"[step={self.step}] shared_dict missing blocks, expect={len(task_meta)} actual={len(shared_keys)} keys={shared_keys}"
                )

            for task_id, l_idx, r_idx in task_meta:
                expected_rows = max(0, r_idx - l_idx)
                payload = shared_dict.get(task_id, None)
                fit_block = self._normalize_fit_block(payload, expected_rows)
                fit_blocks.append(fit_block)

        if len(fit_blocks) == 0:
            fit = self._build_penalty_fit(x.shape[0])
        else:
            fit = np.vstack(fit_blocks)
            if fit.shape[0] < x.shape[0]:
                fit = np.vstack([fit, self._build_penalty_fit(x.shape[0] - fit.shape[0])])
            elif fit.shape[0] > x.shape[0]:
                fit = fit[:x.shape[0], :]

        out["F"] = fit
        self.step += 1


class SotifMinimize:
    def __init__(self, problem, algorithm, config):
        self.problem = problem
        self.algorithm = algorithm
        self.config = config
        self.res = None

    def run(self):
        self.res = minimize(
            self.problem,
            self.algorithm,
            ("n_gen", self.config.n_gen),
            seed=self.config.seed,
            verbose=self.config.verbose,
        )