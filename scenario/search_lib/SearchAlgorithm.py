import os


from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.result import Result
import numpy as np
import pandas as pd
from pymoo.termination import get_termination  # 新增：导入终止条件工具
import time  # 手动管理时间

class NSGA2Algorithm(NSGA2):
    def __init__(self, config):
        self.algorithm_name = "NSGA-II Algorithm"

        super().__init__(
            pop_size=config.pop_size,
            n_offsprings=config.n_offsprings,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=config.crossoverProb, eta=config.crossoverEta),
            mutation=PM(prob=1 / config.n_var, eta=config.mutationEta),
            eliminate_duplicates=True,
            save_history=True)

class PSOAlgorithm(PSO):
    def __init__(self,config):
        super().__init__(pop_size=config.pop_size,save_history=True)
        self.algorithm_name = "PSOAlgorithm"

class RandomAlgorithm(RandomSearch):
    def __init__(self,config):
        super().__init__(n_points_per_iteration=config.pop_size,save_history=True)
        self.algorithm_name = "RandomAlgorithm"


class GCE3SAlgorithm(RandomSearch):
    def __init__(self, config, **kwargs):
        # 基础属性
        self.pop_size = config.pop_size
        self.n_var = config.n_var
        self.max_iter = config.n_gen if hasattr(config, "n_gen") else 3

        # 父类初始化
        super().__init__(
            pop_size=self.pop_size,
            sampling=FloatRandomSampling(),
            **kwargs
        )

        # 优化参数
        self.global_best_ind = None
        self.global_best_F = np.inf
        self.elite_ratio = 0.2
        self.search_radius = 0.3
        self.min_radius = 0.05
        self.radius_decay = 0.98

    def _create_population(self, samples):
        """
        核心修复：批量创建Population对象
        pymoo的Population不支持append，需用这种方式创建
        """
        individuals = []
        for x in samples:
            ind = Individual()
            ind.X = x.flatten()
            ind.evaluated = set()  # 必须初始化
            individuals.append(ind)
        # 使用Population构造函数创建（唯一正确方式）
        return Population(individuals)

    def _infill(self):
        """生成Population对象（完全符合pymoo规范）"""
        # 1. 先生成数值样本数组
        samples = []

        # 精英样本
        elite_size = max(1, int(self.pop_size * self.elite_ratio))
        if hasattr(self, 'pop') and self.pop is not None and len(self.pop) > 0:
            valid_inds = [ind for ind in self.pop if ind.F is not None and not np.isinf(ind.F[0])]
            if len(valid_inds) > 0:
                valid_inds.sort(key=lambda ind: ind.F[0])
                for ind in valid_inds[:elite_size]:
                    x = ind.X * np.random.uniform(1 - self.search_radius, 1 + self.search_radius, self.n_var)
                    x = np.clip(x, self.problem.xl, self.problem.xu)
                    samples.append(x)

        # 局部搜索样本
        local_size = int(self.pop_size * 0.5)
        if self.global_best_ind is not None:
            for _ in range(local_size):
                if len(samples) >= self.pop_size:
                    break
                x = self.global_best_ind.X * np.random.uniform(1 - self.search_radius, 1 + self.search_radius,
                                                               self.n_var)
                x = np.clip(x, self.problem.xl, self.problem.xu)
                samples.append(x)

        # 全局探索样本
        while len(samples) < self.pop_size:
            x = np.random.uniform(self.problem.xl, self.problem.xu, self.n_var)
            samples.append(x)

        # 确保样本数量正确
        samples = np.array(samples[:self.pop_size])

        # 2. 转换为Population对象（核心修复）
        infills = self._create_population(samples)

        # 自适应调整半径
        self.search_radius = max(self.min_radius, self.search_radius * self.radius_decay)

        return infills

    def _advance(self, infills=None, **kwargs):
        """迭代逻辑（更新全局最优）"""
        # 评估新样本
        if infills is not None and len(infills) > 0:
            self.evaluator.eval(self.problem, infills, algorithm=self)

        # 更新种群
        self.pop = infills

        # 更新全局最优
        for ind in self.pop:
            if ind.F is not None and ind.F[0] < self.global_best_F:
                self.global_best_F = ind.F[0]
                self.global_best_ind = ind

    def _solve(self):
        """主求解逻辑"""
        # 初始化
        self.global_best_ind = None
        self.global_best_F = np.inf
        self.search_radius = 0.3

        # 执行父类逻辑
        res = super()._solve()

        # 使用全局最优结果
        if self.global_best_ind is not None:
            res.X = self.global_best_ind.X
            res.F = self.global_best_ind.F
            self.opt = Population([self.global_best_ind])

        return res


class ImprovedDGGAAlgorithm(Algorithm):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pop_size = config.pop_size
        self.n_offsprings = config.n_offsprings
        self.n_gen = config.n_gen
        self.crossover_prob = config.crossoverProb
        self.crossover_eta = config.crossoverEta
        self.mutation_prob = 1 / config.n_var
        self.mutation_eta = config.mutationEta

        self.sampling = config.sampling if hasattr(config, "sampling") else None
        self.eliminate_duplicates = True
        self.save_history = True
        self.history = []

        self.diversity_threshold = 0.35
        self.elite_ratio = 0.15
        self.elite_num = max(1, int(self.pop_size * self.elite_ratio))
        self.collision_weight = 1.9
        self.extreme_collision_weight = 2.8
        self.normal_collision_weight = 1.5
        self.invalid_collision_weight = 0.45
        self.random_parent_ratio = 0.03
        self.random_survivor_ratio = 0.02
        self.max_random_ratio = 0.10
        self.restart_ratio = 0.05
        self.best_f_so_far = np.inf
        self.stagnation_counter = 0
        self.sim_failure_count = 0
        self.target_collision_ratio = 0.32
        self.high_collision_ratio = 0.40
        self.archive_size = max(self.pop_size, 60)
        self.risk_archive_X = np.empty((0, config.n_var))
        self.risk_archive_F = np.array([], dtype=float)

        self.ablation_mode = str(getattr(config, "ablation_mode", "full") or "full").lower()
        self.disable_risk_weight = self.ablation_mode in {"no_risk_weight", "no_risk", "risk_off"}
        self.disable_archive = self.ablation_mode in {"no_archive", "archive_off"}
        self.disable_stagnation = self.ablation_mode in {"no_stagnation", "stagnation_off"}
        self.disable_diversity = self.ablation_mode in {"no_diversity", "diversity_off", "no_duplicate", "no_dedup"}

        if self.disable_diversity:
            self.eliminate_duplicates = False
        if self.disable_archive:
            self.archive_size = 0

        self.algorithm_name = "X-DGGA Algorithm" if self.ablation_mode == "full" else f"X-DGGA Ablation ({self.ablation_mode})"
        self.termination = get_termination("n_gen", self.n_gen)
        self.start_time = None
        self.end_time = None
        self.pop = Population()
        self.result_csv = os.path.join(self.config.search_save_path, "sotif_search_results.csv")

        self.sim_metrics = {
            "collision": np.array([], dtype=bool),
            "min_TTC": np.array([], dtype=float),
            "collision_distance": np.array([], dtype=float),
            "collision_fit": np.array([], dtype=float),
        }

    def run(self):
        self.start_time = time.time()
        self.end_time = None

        if not self.problem:
            raise ValueError("Problem not set for Improved DGGA algorithm!")

        self._initialize()

        for gen in range(self.n_gen):
            print(f"\n=== Improved DGGA 迭代 {gen + 1}/{self.n_gen} ===")
            self._next(gen)

        self.end_time = time.time()
        return self._get_result()

    def _initialize(self):
        init_X = self._sample_population(self.pop_size)
        self.pop = self._safe_evaluate_population(init_X, stage_name="initial_population")

        if len(self.pop) < self.pop_size:
            extra_X = self._sample_population(self.pop_size - len(self.pop))
            extra_pop = self._safe_evaluate_population(extra_X, stage_name="initial_refill")
            self.pop = Population.merge(self.pop, extra_pop)
            self.pop = self._filter_valid_population(self.pop)

        if len(self.pop) > self.pop_size:
            best_idx = np.argsort(self.pop.get("F").flatten())[:self.pop_size]
            self.pop = self.pop[best_idx]

        self._extract_sim_metrics()
        self.best_f_so_far = float(np.min(self.pop.get("F").flatten()))
        self.stagnation_counter = 0
        if not self.disable_archive:
            self._update_risk_archive(self.pop)

        if self.save_history:
            self.history.append(self.pop.copy())

        init_F = self.pop.get("F").flatten()
        print(f"Improved DGGA 初始化完成，种群数量：{len(self.pop)}")
        print(f"初始碰撞个体数：{int(np.sum(self.sim_metrics['collision']))}")
        print(f"初始适应度范围：[{np.min(init_F):.4f}, {np.max(init_F):.4f}]")
        print(f"初始最优适应度：{np.min(init_F):.4f}")

    def _sample_population(self, size):
        if self.sampling is None:
            return np.random.uniform(
                low=self.problem.xl,
                high=self.problem.xu,
                size=(size, self.problem.n_var)
            )

        sampled = self.sampling.do(self.problem, size)
        if isinstance(sampled, Population):
            return sampled.get("X")
        return np.asarray(sampled)

    def _evaluate_and_sync(self, pop):
        self.evaluator.eval(self.problem, pop, algorithm=self)
        metrics = self._load_metrics_from_csv(len(pop), pop.get("F").flatten())
        self._attach_metrics(pop, metrics)

    def _safe_evaluate_population(self, X, stage_name="population"):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, self.problem.xl, self.problem.xu)

        pop = Population.new("X", X)
        try:
            self._evaluate_and_sync(pop)
            self.sim_failure_count = 0
            return self._filter_valid_population(pop)
        except Exception as exc:
            self.sim_failure_count += 1
            print(f"[ImprovedDGGA] {stage_name} evaluate failed, retrying with safer samples: {exc}")

        recovery_X = self._build_recovery_samples(len(X))
        recovery_pop = Population.new("X", recovery_X)
        try:
            self._evaluate_and_sync(recovery_pop)
            return self._filter_valid_population(recovery_pop)
        except Exception as retry_exc:
            print(f"[ImprovedDGGA] {stage_name} retry failed: {retry_exc}")
            fallback_pop = self._build_surrogate_population(len(X), stage_name=stage_name)
            return fallback_pop

    def _build_recovery_samples(self, size):
        if size <= 0:
            return np.empty((0, self.problem.n_var))

        safe_X = self._sample_population(size)

        archive_count = min(size, max(1, int(np.ceil(size * 0.5))))
        archive_X = self._sample_archive_guided(archive_count, sigma=0.015)
        if archive_count > 0 and len(archive_X) > 0:
            use_archive = min(len(archive_X), size)
            safe_X[:use_archive] = archive_X[:use_archive]

        if len(self.pop) > 0:
            masks = self._classify_population(self.pop)
            valid_idx = np.where(masks["valid"])[0]
            elite_count = min(len(self.pop), max(1, int(np.ceil(size * 0.5))))
            if len(valid_idx) > 0:
                ordered_valid = valid_idx[np.argsort(self.pop.get("F").flatten()[valid_idx])]
                elite_idx = ordered_valid[:elite_count]
            else:
                elite_idx = np.argsort(self.pop.get("F").flatten())[:elite_count]

            elite_count = len(elite_idx)
            if elite_count == 0:
                return np.clip(safe_X, self.problem.xl, self.problem.xu)

            elite_X = self.pop.get("X")[elite_idx]
            local_count = min(size, elite_count)
            sigma = 0.02
            scale = self.problem.xu - self.problem.xl
            for i in range(local_count):
                base = elite_X[i % elite_count].copy()
                noise = np.random.normal(0.0, sigma, size=self.problem.n_var) * scale
                safe_X[i] = np.clip(base + noise, self.problem.xl, self.problem.xu)

        return np.clip(safe_X, self.problem.xl, self.problem.xu)

    def _build_surrogate_population(self, size, stage_name="population"):
        if size <= 0:
            return Population()

        if len(self.pop) == 0:
            x = self._sample_population(size)
            f = np.full((size, 1), 1e6, dtype=float)
            fallback = Population.new("X", x, "F", f)
            for ind in fallback:
                self._set_ind_value(ind, "collision", False)
                self._set_ind_value(ind, "min_TTC", 100.0)
                self._set_ind_value(ind, "collision_distance", 0.0)
                self._set_ind_value(ind, "collision_fit", 0.0)
            print(f"[ImprovedDGGA] {stage_name} fallback to synthetic safe population (cold start).")
            return fallback

        source_f = self.pop.get("F").flatten()
        order = np.argsort(source_f)
        take = np.resize(order, size)
        base_x = self.pop.get("X")[take].copy()
        base_f = source_f[take].copy()

        scale = self.problem.xu - self.problem.xl
        noise = np.random.normal(0.0, 0.01, size=base_x.shape) * scale
        x = np.clip(base_x + noise, self.problem.xl, self.problem.xu)

        penalty = np.abs(np.random.normal(0.0, 0.02, size=size))
        f = (base_f + penalty).reshape(-1, 1)
        fallback = Population.new("X", x, "F", f)

        for i, ind in enumerate(fallback):
            src = self.pop[take[i]]
            self._set_ind_value(ind, "collision", bool(self._get_ind_value(src, "collision", False)))
            self._set_ind_value(ind, "min_TTC", float(self._get_ind_value(src, "min_TTC", 100.0)))
            self._set_ind_value(ind, "collision_distance", float(self._get_ind_value(src, "collision_distance", 0.0)))
            self._set_ind_value(ind, "collision_fit", float(self._get_ind_value(src, "collision_fit", 0.0)))

        print(f"[ImprovedDGGA] {stage_name} fallback to surrogate population (sim pipeline unstable).")
        return fallback

    def _load_metrics_from_csv(self, expected_size, fallback_f):
        metrics = {
            "collision": np.zeros(expected_size, dtype=bool),
            "min_TTC": np.full(expected_size, 100.0, dtype=float),
            "collision_distance": np.zeros(expected_size, dtype=float),
            "collision_fit": np.zeros(expected_size, dtype=float),
        }

        if not os.path.exists(self.result_csv):
            return metrics

        try:
            df = pd.read_csv(self.result_csv)
        except Exception:
            return metrics

        if df.empty:
            return metrics

        if "step" in df.columns and hasattr(self.problem, "step"):
            eval_step = max(int(self.problem.step) - 1, 0)
            step_df = df[df["step"] == eval_step]
            if not step_df.empty:
                df = step_df

        df = df.tail(expected_size).reset_index(drop=True)
        row_count = len(df)
        if row_count == 0:
            return metrics

        min_ttc = self._pad_array(
            pd.to_numeric(df["min_TTC"], errors="coerce").fillna(100.0).to_numpy(dtype=float),
            expected_size,
            100.0,
        )

        if "collision" in df.columns:
            collision_raw = df["collision"]
            collision = collision_raw.astype(str).str.lower().isin(["1", "true", "yes"]).to_numpy(dtype=bool)
        else:
            collision = np.zeros(row_count, dtype=bool)

        obj_col = self.config.objs[0] if len(self.config.objs) > 0 else None
        if obj_col and obj_col in df.columns:
            obj_values = pd.to_numeric(df[obj_col], errors="coerce").to_numpy(dtype=float)
        else:
            obj_values = np.asarray(fallback_f[:row_count], dtype=float)

        collision_fit = obj_values - min_ttc[:row_count]
        collision = np.logical_or(collision, np.abs(collision_fit) > 1e-8)
        valid_collision = np.logical_and(collision, collision_fit < 0)
        collision_distance = np.where(valid_collision, 10.0, np.where(collision, 0.0, 100.0))

        metrics["collision"] = self._pad_array(collision.astype(bool), expected_size, False)
        metrics["min_TTC"] = min_ttc
        metrics["collision_fit"] = self._pad_array(collision_fit, expected_size, 0.0)
        metrics["collision_distance"] = self._pad_array(collision_distance, expected_size, 0.0)

        return metrics

    def _pad_array(self, values, expected_size, fill_value):
        arr = np.asarray(values)
        if arr.size >= expected_size:
            return arr[-expected_size:]
        pad = np.full(expected_size - arr.size, fill_value, dtype=arr.dtype if arr.dtype != object else object)
        return np.concatenate([arr, pad])

    def _attach_metrics(self, pop, metrics):
        for i, ind in enumerate(pop):
            self._set_ind_value(ind, "collision", bool(metrics["collision"][i]))
            self._set_ind_value(ind, "min_TTC", float(metrics["min_TTC"][i]))
            self._set_ind_value(ind, "collision_distance", float(metrics["collision_distance"][i]))
            self._set_ind_value(ind, "collision_fit", float(metrics["collision_fit"][i]))

    def _set_ind_value(self, ind, key, value):
        if hasattr(ind, "set"):
            ind.set(key, value)
            return
        if not hasattr(ind, "data") or ind.data is None:
            ind.data = {}
        ind.data[key] = value

    def _get_ind_value(self, ind, key, default):
        if hasattr(ind, "get"):
            try:
                value = ind.get(key)
            except Exception:
                value = None
            if value is not None:
                return value

        data = getattr(ind, "data", None)
        if isinstance(data, dict) and key in data:
            return data[key]

        attrs = getattr(ind, "attributes", None)
        if isinstance(attrs, dict) and key in attrs:
            return attrs[key]

        return default

    def _extract_sim_metrics(self):
        collision = []
        min_ttc = []
        collision_distance = []
        collision_fit = []

        for ind in self.pop:
            collision.append(bool(self._get_ind_value(ind, "collision", False)))
            min_ttc.append(float(self._get_ind_value(ind, "min_TTC", 100.0)))
            collision_distance.append(float(self._get_ind_value(ind, "collision_distance", 0.0)))
            collision_fit.append(float(self._get_ind_value(ind, "collision_fit", 0.0)))

        self.sim_metrics["collision"] = np.asarray(collision, dtype=bool)
        self.sim_metrics["min_TTC"] = np.asarray(min_ttc, dtype=float)
        self.sim_metrics["collision_distance"] = np.asarray(collision_distance, dtype=float)
        self.sim_metrics["collision_fit"] = np.asarray(collision_fit, dtype=float)

    def _filter_valid_population(self, pop):
        if len(pop) == 0:
            return pop
        valid_idx = np.where(np.isfinite(pop.get("F").flatten()))[0]
        return pop[valid_idx] if len(valid_idx) > 0 else Population()

    def _next(self, gen):
        if len(self.pop) == 0:
            self._initialize()
            return

        X = self.pop.get("X")
        diversity = self._calculate_diversity(X)
        current_collision_rate = float(np.mean(self.sim_metrics["collision"])) if len(self.sim_metrics["collision"]) > 0 else 0.0
        current_best = float(np.min(self.pop.get("F").flatten()))
        if current_best < self.best_f_so_far - 1e-6:
            self.best_f_so_far = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        print(f"种群多样性：{diversity:.4f}")

        cross_prob, mutate_prob = self._adaptive_prob(diversity, gen)
        print(f"交叉概率：{cross_prob:.4f}，变异概率：{mutate_prob:.4f}")

        selected_X = self._collision_weighted_selection(
            self.pop,
            n_select=self.n_offsprings,
            diversity=diversity,
            collision_rate=current_collision_rate,
            gen=gen,
        )
        if len(selected_X) % 2 != 0:
            selected_X = np.vstack([selected_X, selected_X[-1]])

        offspring_X = self._sbx_crossover(selected_X, cross_prob, self.crossover_eta)
        offspring_X = self._pm_mutation(offspring_X, mutate_prob, self.mutation_eta)
        offspring_X = self._inject_random_immigrants(offspring_X, diversity, current_collision_rate)
        offspring_X = self._balance_param_distribution(offspring_X)
        offspring = self._safe_evaluate_population(offspring_X[:self.n_offsprings], stage_name="offspring")

        combined = Population.merge(self.pop, offspring)
        combined = self._deduplicate_population(combined)
        if not self.disable_archive:
            self._update_risk_archive(combined)
        self.pop = self._environmental_selection(combined, current_collision_rate, gen)
        self._extract_sim_metrics()

        if self.save_history:
            self.history.append(self.pop.copy())

        self._print_generation_stats()

    def _collision_weighted_selection(self, pop, n_select, diversity=None, collision_rate=0.0, gen=0):
        X = pop.get("X")
        F = pop.get("F").flatten()
        masks = self._classify_population(pop)
        novelty = self._novelty_score(X)

        risk_pressure = self._compute_collision_pressure(collision_rate, gen)
        weights = np.ones(len(pop), dtype=float)
        if not self.disable_risk_weight:
            weights[masks["collision"]] *= self.collision_weight
            weights[masks["normal"]] *= self.normal_collision_weight
            weights[masks["extreme"]] *= self.extreme_collision_weight
            weights[masks["invalid"]] *= self.invalid_collision_weight
            weights[masks["valid"]] *= risk_pressure
            weights[masks["extreme"]] *= 1.0 + 0.2 * max(0.0, risk_pressure - 1.0)

        objective_score = 1.0 / (F - np.min(F) + 1e-6)
        prob = objective_score * weights * (1.0 + 0.25 * novelty)
        if not np.all(np.isfinite(prob)) or prob.sum() <= 0:
            prob = np.full(len(pop), 1.0 / len(pop))
        else:
            prob = prob / prob.sum()

        selected_idx = []
        used = set()

        if not self.disable_risk_weight:
            selected_idx = self._extend_with_best(selected_idx, used, np.where(masks["extreme"])[0], F, max(1, int(np.ceil(self.elite_num * 0.4))))
            selected_idx = self._extend_with_best(selected_idx, used, np.where(masks["normal"])[0], F, max(1, int(np.ceil(self.elite_num * 0.35))))
            if collision_rate < self.target_collision_ratio:
                selected_idx = self._extend_with_best(selected_idx, used, np.where(masks["valid"])[0], F, max(1, int(np.ceil(self.elite_num * 0.8))))
        selected_idx = self._extend_with_best(selected_idx, used, np.arange(len(pop)), F, self.elite_num)

        rest_size = max(0, n_select - len(selected_idx))
        archive_share = 0.10 if collision_rate < self.target_collision_ratio else 0.08
        archived_X = self._sample_archive_guided(max(1, int(np.ceil(n_select * archive_share))), sigma=0.025)
        if rest_size > 0:
            immigrant_ratio = self._compute_immigrant_ratio(diversity)
            immigrant_count = min(rest_size, int(np.ceil(rest_size * immigrant_ratio))) if immigrant_ratio > 0.0 else 0
            archive_count = min(rest_size - immigrant_count, len(archived_X))
            guided_count = rest_size - immigrant_count - archive_count

            if guided_count > 0:
                sampled = np.random.choice(len(pop), size=guided_count, replace=(guided_count > len(pop)), p=prob)
                selected_idx.extend(sampled.tolist())

            if archive_count > 0:
                archive_block = archived_X[:archive_count]
            else:
                archive_block = np.empty((0, self.problem.n_var))

            if immigrant_count > 0:
                candidate_pool = np.argsort(novelty)[::-1]
                immigrant_idx = []
                for idx in candidate_pool:
                    if int(idx) not in used:
                        immigrant_idx.append(int(idx))
                    if len(immigrant_idx) >= immigrant_count:
                        break
                if len(immigrant_idx) < immigrant_count:
                    random_idx = np.random.choice(len(pop), size=immigrant_count - len(immigrant_idx), replace=True)
                    immigrant_idx.extend(random_idx.tolist())
                selected_idx.extend(immigrant_idx)

            selected_X = X[selected_idx[:(n_select - archive_count)]]
            if archive_count > 0:
                selected_X = np.vstack([selected_X, archive_block])
            return selected_X[:n_select]

        return X[selected_idx[:n_select]]

    def _extend_with_best(self, selected_idx, used, candidate_idx, F, target_size):
        if target_size <= len(selected_idx) or len(candidate_idx) == 0:
            return selected_idx

        order = candidate_idx[np.argsort(F[candidate_idx])]
        for idx in order:
            if idx not in used:
                used.add(int(idx))
                selected_idx.append(int(idx))
            if len(selected_idx) >= target_size:
                break

        return selected_idx

    def _classify_population(self, pop):
        collision = np.array([bool(self._get_ind_value(ind, "collision", False)) for ind in pop], dtype=bool)
        min_ttc = np.array([float(self._get_ind_value(ind, "min_TTC", 100.0)) for ind in pop], dtype=float)
        collision_dist = np.array([float(self._get_ind_value(ind, "collision_distance", 0.0)) for ind in pop], dtype=float)

        valid = collision & (collision_dist > 3.5)
        extreme = valid & (min_ttc < 1.0)
        normal = valid & (min_ttc >= 1.0) & (min_ttc < 3.0)
        invalid = collision & (~valid)

        return {
            "collision": collision,
            "valid": valid,
            "extreme": extreme,
            "normal": normal,
            "invalid": invalid,
        }

    def _deduplicate_population(self, pop):
        if self.disable_diversity or not self.eliminate_duplicates or len(pop) <= 1:
            return pop

        X = pop.get("X")
        F = pop.get("F").flatten()
        masks = self._classify_population(pop)
        keep = {}

        for i in range(len(pop)):
            precision = 5 if masks["valid"][i] else 6
            key = tuple(np.round(X[i], precision))
            if key not in keep or F[i] < F[keep[key]]:
                keep[key] = i

        selected = sorted(keep.values())
        return pop[selected]

    def _update_risk_archive(self, pop):
        if self.disable_archive or len(pop) == 0:
            return

        masks = self._classify_population(pop)
        F = pop.get("F").flatten()
        X = pop.get("X")

        candidate_mask = masks["valid"] | masks["normal"] | masks["extreme"]
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.argsort(F)[:min(len(pop), max(5, self.elite_num))]

        cand_X = X[candidate_idx]
        cand_F = F[candidate_idx]

        if len(self.risk_archive_X) == 0:
            merged_X = cand_X.copy()
            merged_F = cand_F.copy()
        else:
            merged_X = np.vstack([self.risk_archive_X, cand_X])
            merged_F = np.concatenate([self.risk_archive_F, cand_F])

        keep = {}
        for i in range(len(merged_X)):
            key = tuple(np.round(merged_X[i], 5))
            if key not in keep or merged_F[i] < merged_F[keep[key]]:
                keep[key] = i

        selected = np.array(sorted(keep.values(), key=lambda idx: merged_F[idx]))
        selected = selected[:self.archive_size]
        self.risk_archive_X = merged_X[selected]
        self.risk_archive_F = merged_F[selected]

    def _environmental_selection(self, combined, current_collision_rate=0.0, gen=0):
        if len(combined) == 0:
            return Population()

        F = combined.get("F").flatten()
        X = combined.get("X")
        masks = self._classify_population(combined)
        novelty = self._novelty_score(X)

        risk_pressure = self._compute_collision_pressure(current_collision_rate, gen)
        score = -F
        if not self.disable_diversity:
            score += novelty * 0.15
        if not self.disable_risk_weight:
            score += masks["valid"] * (8.0 * risk_pressure)
            score += masks["normal"] * 2.5
            score += masks["extreme"] * (5.0 * (1.0 + 0.15 * max(0.0, risk_pressure - 1.0)))
            score -= masks["invalid"] * 4.0

        selected = []
        used = set()
        ranked = np.argsort(score)[::-1]

        if not self.disable_risk_weight:
            min_valid_keep = max(1, int(np.ceil(self.pop_size * min(self.high_collision_ratio, self.target_collision_ratio * risk_pressure))))
            selected = self._extend_until(selected, used, ranked[masks["extreme"][ranked]], max(1, int(self.pop_size * 0.2)))
            selected = self._extend_until(selected, used, ranked[masks["normal"][ranked]], max(1, int(self.pop_size * 0.45)))
            selected = self._extend_until(selected, used, ranked[masks["valid"][ranked]], max(min_valid_keep, int(self.pop_size * 0.6)))
        diversity = self._calculate_diversity(X)
        immigrant_ratio = self._compute_immigrant_ratio(diversity)
        reserve_for_immigrants = min(self.pop_size - len(selected), max(0, int(np.ceil(self.pop_size * immigrant_ratio * self.random_survivor_ratio / max(self.random_parent_ratio, 1e-6)))))
        target_before_fill = max(len(selected), self.pop_size - reserve_for_immigrants)
        selected = self._extend_until(selected, used, ranked, target_before_fill)

        if not self.disable_diversity and len(selected) < self.pop_size:
            high_novelty = np.argsort(novelty)[::-1]
            selected = self._extend_until(selected, used, high_novelty, self.pop_size)
        elif len(selected) < self.pop_size:
            selected = self._extend_until(selected, used, ranked, self.pop_size)

        survivors = Population([combined[i] for i in selected[:self.pop_size]])

        if len(survivors) < self.pop_size:
            add_X = self._sample_population(self.pop_size - len(survivors))
            add_pop = self._safe_evaluate_population(add_X, stage_name="survivor_refill")
            survivors = Population.merge(survivors, add_pop)

        if not self.disable_stagnation:
            survivors = self._restart_if_stagnated(survivors)

        return survivors[:self.pop_size]

    def _extend_until(self, selected, used, candidates, target_size):
        if len(selected) >= target_size:
            return selected
        for idx in candidates:
            if int(idx) not in used:
                used.add(int(idx))
                selected.append(int(idx))
            if len(selected) >= target_size:
                break
        return selected

    def _novelty_score(self, X):
        if self.disable_diversity:
            return np.zeros(len(X), dtype=float)
        if len(X) <= 1:
            return np.ones(len(X), dtype=float)

        X_norm = (X - self.problem.xl) / (self.problem.xu - self.problem.xl + 1e-12)
        dist = np.linalg.norm(X_norm[:, None, :] - X_norm[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        nearest = np.min(dist, axis=1)
        if np.max(nearest) <= 1e-12:
            return np.ones(len(X), dtype=float)
        return nearest / np.max(nearest)

    def _compute_immigrant_ratio(self, diversity):
        if self.disable_diversity:
            return 0.0

        if diversity is None:
            diversity = self.diversity_threshold

        ratio = 0.0
        if diversity < self.diversity_threshold:
            ratio = max(ratio, self.random_parent_ratio)
        if not self.disable_stagnation:
            if self.stagnation_counter >= 6:
                ratio += 0.02
            if self.stagnation_counter >= 10:
                ratio += 0.03
        if self.sim_failure_count > 0:
            ratio *= 0.5

        return float(np.clip(ratio, 0.0, self.max_random_ratio))

    def _compute_collision_pressure(self, collision_rate, gen):
        target = self.target_collision_ratio
        if gen <= 10:
            target += 0.05
        elif gen >= 35:
            target -= 0.02

        gap = max(0.0, target - collision_rate)
        pressure = 1.0 + min(0.8, gap * 3.0)
        return pressure

    def _sample_archive_guided(self, count, sigma=0.03):
        if self.disable_archive or count <= 0 or len(self.risk_archive_X) == 0:
            return np.empty((0, self.problem.n_var))

        base_idx = np.arange(min(len(self.risk_archive_X), max(5, count)))
        sampled_idx = np.random.choice(base_idx, size=count, replace=(count > len(base_idx)))
        base = self.risk_archive_X[sampled_idx].copy()
        scale = self.problem.xu - self.problem.xl
        noise = np.random.normal(0.0, sigma, size=base.shape) * scale
        return np.clip(base + noise, self.problem.xl, self.problem.xu)

    def _inject_random_immigrants(self, X, diversity, collision_rate=0.0):
        if len(X) == 0:
            return X

        immigrant_ratio = self._compute_immigrant_ratio(diversity)
        if immigrant_ratio <= 0.0:
            return X

        immigrant_count = min(len(X), max(1, int(np.ceil(len(X) * immigrant_ratio))))
        if immigrant_count <= 0:
            return X

        archive_portion = 0.75 if collision_rate < self.target_collision_ratio else 0.60
        archive_count = min(immigrant_count, max(0, int(np.ceil(immigrant_count * archive_portion))))
        archive_X = self._sample_archive_guided(archive_count, sigma=0.025)
        archive_count = min(archive_count, len(archive_X))
        fresh_X = self._sample_population(immigrant_count)
        if archive_count > 0:
            fresh_X[:archive_count] = archive_X[:archive_count]

        if len(self.pop) > 0:
            elite_count = min(max(1, immigrant_count), len(self.pop))
            elite_idx = np.argsort(self.pop.get("F").flatten())[:elite_count]
            elite_X = self.pop.get("X")[elite_idx]
            local_portion = 0.90 if collision_rate < self.target_collision_ratio else 0.80
            local_count = max(1, int(np.ceil(immigrant_count * local_portion)))
            if local_count > 0:
                local_X = []
                sigma = 0.03 + 0.008 * min(self.stagnation_counter, 8)
                scale = self.problem.xu - self.problem.xl
                for i in range(local_count):
                    base = elite_X[i % elite_count].copy()
                    noise = np.random.normal(0.0, sigma, size=self.problem.n_var) * scale
                    local_X.append(np.clip(base + noise, self.problem.xl, self.problem.xu))
                local_X = np.asarray(local_X)
                fresh_X[:local_count] = local_X

        replace_idx = np.random.choice(len(X), size=immigrant_count, replace=False)
        X[replace_idx] = fresh_X
        return X

    def _restart_if_stagnated(self, pop):
        if self.disable_stagnation or len(pop) == 0 or self.stagnation_counter < 12:
            return pop

        keep_count = max(1, int(np.floor(self.pop_size * (1.0 - self.restart_ratio))))
        if len(pop) > keep_count:
            best_idx = np.argsort(pop.get("F").flatten())[:keep_count]
            pop = pop[best_idx]

        refill_count = self.pop_size - len(pop)
        if refill_count <= 0:
            return pop

        archive_count = min(refill_count, max(1, int(np.ceil(refill_count * 0.5))))
        archive_X = self._sample_archive_guided(archive_count, sigma=0.035)
        archive_count = min(archive_count, len(archive_X))
        restart_X = self._sample_population(refill_count)
        if archive_count > 0:
            restart_X[:archive_count] = archive_X[:archive_count]
        restart_pop = self._safe_evaluate_population(restart_X, stage_name="restart_refill")
        pop = Population.merge(pop, restart_pop)
        return pop

    def _adaptive_prob(self, diversity, gen):
        if self.disable_diversity:
            return np.clip(self.crossover_prob, 0.70, 0.90), np.clip(self.mutation_prob, 0.08, 0.25)

        if gen <= 15:
            self.diversity_threshold = 0.40
        elif gen <= 35:
            self.diversity_threshold = 0.37
        else:
            self.diversity_threshold = 0.34

        if diversity < self.diversity_threshold:
            if gen <= 15:
                cross_prob = 0.78
                mutate_prob = 0.16
            elif gen <= 35:
                cross_prob = 0.74
                mutate_prob = 0.19
            else:
                cross_prob = 0.70
                mutate_prob = 0.22
        else:
            cross_prob = 0.82
            mutate_prob = 0.12

        if not self.disable_stagnation:
            if self.stagnation_counter >= 8:
                cross_prob = min(cross_prob, 0.76)
                mutate_prob += 0.02
            if self.stagnation_counter >= 12:
                cross_prob = min(cross_prob, 0.72)
                mutate_prob += 0.02
        if self.sim_failure_count > 0:
            cross_prob = min(cross_prob, 0.75)
            mutate_prob = min(mutate_prob, 0.16)

        return np.clip(cross_prob, 0.70, 0.90), np.clip(mutate_prob, 0.08, 0.25)

    def _balance_param_distribution(self, X):
        if len(X) < 5 or len(self.pop) == 0:
            return X

        ans_names = list(getattr(self.config, "ans", []))
        origin_X = self.pop.get("X")

        if "npc_relative_v" in ans_names:
            v_idx = ans_names.index("npc_relative_v")
            cur_ratio = np.mean(X[:, v_idx] < -8)
            target_ratio = 0.18
            if cur_ratio < target_ratio:
                pool_idx = np.where(origin_X[:, v_idx] < -8)[0]
                replace_idx = np.where(X[:, v_idx] >= -8)[0]
                need = min(len(pool_idx), len(replace_idx), int(np.ceil((target_ratio - cur_ratio) * len(X))))
                if need > 0:
                    X[replace_idx[:need]] = origin_X[np.random.choice(pool_idx, size=need, replace=False)]

        if "dis_to_trigger" in ans_names:
            d_idx = ans_names.index("dis_to_trigger")
            cur_ratio = np.mean((X[:, d_idx] >= 15) & (X[:, d_idx] <= 30))
            target_ratio = 0.22
            if cur_ratio < target_ratio:
                pool_idx = np.where((origin_X[:, d_idx] >= 15) & (origin_X[:, d_idx] <= 30))[0]
                replace_idx = np.where((X[:, d_idx] < 15) | (X[:, d_idx] > 30))[0]
                need = min(len(pool_idx), len(replace_idx), int(np.ceil((target_ratio - cur_ratio) * len(X))))
                if need > 0:
                    X[replace_idx[:need]] = origin_X[np.random.choice(pool_idx, size=need, replace=False)]

        return X

    def _print_generation_stats(self):
        if len(self.pop) == 0:
            print("本轮种群为空，跳过统计")
            return

        current_F = self.pop.get("F").flatten()
        current_collision = int(np.sum(self.sim_metrics["collision"]))
        current_extreme = int(np.sum((self.sim_metrics["min_TTC"] < 1.0) & self.sim_metrics["collision"]))
        current_X = self.pop.get("X")
        unique_count = len(np.unique(current_X.round(6), axis=0))
        repeat_ratio = 1 - (unique_count / len(current_X))

        print(f"本轮最优适应度：{np.min(current_F):.4f}")
        print(f"本轮种群适应度均值：{np.mean(current_F):.4f}")
        print(f"本轮碰撞个体数：{current_collision}（占比：{current_collision / len(self.pop):.4f}）")
        if current_collision > 0:
            print(f"本轮极端碰撞场景数：{current_extreme}（占碰撞数：{current_extreme / current_collision:.4f}）")
        else:
            print("本轮无碰撞")
        print(f"本轮重复个体占比：{repeat_ratio:.4f}")

    def _sbx_crossover(self, X, prob, eta):
        n, n_var = X.shape
        off = X.copy()
        for i in range(0, n, 2):
            if i + 1 >= n:
                break
            if np.random.random() <= prob:
                x1 = X[i]
                x2 = X[i + 1]
                for j in range(n_var):
                    if abs(x1[j] - x2[j]) > 1e-14 and x1[j] < x2[j]:
                        xl, xu = self.problem.xl[j], self.problem.xu[j]
                        beta = 1.0 + (2.0 * (x1[j] - xl)) / (x2[j] - x1[j])
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        rand = np.random.random()
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * (x1[j] + x2[j] - beta_q * (x2[j] - x1[j]))

                        beta = 1.0 + (2.0 * (xu - x2[j])) / (x2[j] - x1[j])
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        c2 = 0.5 * (x1[j] + x2[j] + beta_q * (x2[j] - x1[j]))

                        off[i, j] = np.clip(c1, xl, xu)
                        off[i + 1, j] = np.clip(c2, xl, xu)
        return off

    def _pm_mutation(self, X, prob, eta):
        n, n_var = X.shape
        off = X.copy()
        for i in range(n):
            for j in range(n_var):
                if np.random.random() <= prob:
                    x = X[i, j]
                    xl, xu = self.problem.xl[j], self.problem.xu[j]
                    delta1 = (x - xl) / (xu - xl)
                    delta2 = (xu - x) / (xu - xl)
                    rand = np.random.random()
                    mut_pow = 1.0 / (eta + 1.0)
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                        delta_q = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                        delta_q = 1.0 - val ** mut_pow
                    off[i, j] = np.clip(x + delta_q * (xu - xl), xl, xu)
        return off

    def _calculate_diversity(self, X):
        if len(X) < 2:
            return 1.0
        X_norm = (X - self.problem.xl) / (self.problem.xu - self.problem.xl + 1e-12)
        dists = []
        for i in range(len(X_norm)):
            for j in range(i + 1, len(X_norm)):
                dists.append(np.linalg.norm(X_norm[i] - X_norm[j]))
        return np.mean(dists) / np.sqrt(self.problem.n_var) if dists else 0.0

    def _get_result(self):
        res = Result()
        res.algorithm = self
        res.problem = self.problem
        res.start_time = self.start_time
        res.end_time = self.end_time
        res.exec_time = self.end_time - self.start_time if self.end_time is not None else 0.0
        res.history = self.history

        if len(self.pop) == 0:
            res.X = None
            res.F = None
            res.pop = None
            print("\n=== Improved DGGA 迭代完成 ===")
            print("警告：最终种群为空，无法统计结果")
            return res

        final_F = self.pop.get("F").flatten()
        best_idx = np.argmin(final_F)
        res.X = self.pop.get("X")[best_idx]
        res.F = self.pop.get("F")[best_idx]
        res.pop = self.pop

        final_collision = int(np.sum(self.sim_metrics["collision"]))
        final_collision_rate = final_collision / len(self.pop)
        final_extreme = int(np.sum((self.sim_metrics["min_TTC"] < 1.0) & self.sim_metrics["collision"]))
        final_normal = int(np.sum((self.sim_metrics["min_TTC"] >= 1.0) &
                                  (self.sim_metrics["min_TTC"] < 3.0) &
                                  self.sim_metrics["collision"]))
        final_X = self.pop.get("X")
        final_unique = len(np.unique(final_X.round(6), axis=0))
        final_repeat_ratio = 1 - (final_unique / len(final_X))

        print("\n=== Improved DGGA 迭代完成 ===")
        print(f"总迭代轮数：{self.n_gen}")
        print(f"最终最优参数：{res.X}")
        print(f"最终最优适应度：{res.F[0]:.4f}")
        print(f"最终碰撞个体数：{final_collision}（占比：{final_collision_rate:.4f}）")
        if final_collision > 0:
            print(f"最终极端碰撞场景数：{final_extreme}（占碰撞数：{final_extreme / final_collision:.4f}）")
            print(f"最终普通碰撞场景数：{final_normal}（占碰撞数：{final_normal / final_collision:.4f}）")
        else:
            print("最终无碰撞")
        print(f"最终重复个体占比：{final_repeat_ratio:.4f}")

        return res

# 最终稳定版 DGGAAlgorithm
# # 关闭仿真进度条输出（消除误导）
# sys.stdout = open('/dev/null', 'w')
# sys.stderr = open('/dev/null', 'w')
# sys.stdout = sys.__stdout__
# sys.stderr = sys.__stderr__



# 最终版：严格控制每轮生成3个结果，无额外日志干扰
class DGGAAlgorithm(Algorithm):
    def __init__(self, config):
        super().__init__()
        # 1. 项目配置
        self.config = config
        self.pop_size = config.pop_size
        self.n_offsprings = config.n_offsprings
        self.n_gen = config.n_gen
        self.crossover_prob = config.crossoverProb
        self.crossover_eta = config.crossoverEta
        self.mutation_prob = 1 / config.n_var
        self.mutation_eta = config.mutationEta

        # 2. 核心属性
        self.sampling = FloatRandomSampling()
        self.eliminate_duplicates = True
        self.save_history = True
        self.history = []

        # 3. DGGA参数
        self.diversity_threshold = 0.3
        self.elite_ratio = 0.1
        self.elite_num = max(1, int(self.pop_size * self.elite_ratio))

        # 4. 算法标识
        self.algorithm_name = "DGGA Algorithm"

        # 5. 终止条件
        self.termination = get_termination("n_gen", self.n_gen)

        # 6. 时间管理
        self.start_time = None
        self.end_time = None

        # 7. 预初始化种群
        self.pop = Population()

    def run(self):
        """完全接管算法运行流程"""
        self.start_time = time.time()
        self.end_time = None

        if not self.problem:
            raise ValueError("Problem not set for DGGA algorithm!")

        self._initialize()

        # 手动迭代所有轮数
        for gen in range(self.n_gen):
            print(f"\n=== DGGA 迭代 {gen + 1}/{self.n_gen} ===")
            self._next()

        self.end_time = time.time()
        res = self._get_result()
        return res

    def _initialize(self):
        """初始化种群"""
        # 生成初始种群
        init_X = np.random.uniform(
            low=self.problem.xl,
            high=self.problem.xu,
            size=(self.pop_size, self.problem.n_var)
        )
        self.pop = Population.new("X", init_X)

        # 评估初始种群
        self.evaluator.eval(self.problem, self.pop, algorithm=self)

        # 过滤无效个体
        valid_F = self.pop.get("F").flatten()
        valid_idx = np.isfinite(valid_F)
        self.pop = self.pop[valid_idx]

        # 兜底
        if len(self.pop) < self.pop_size:
            补充_X = np.random.uniform(
                low=self.problem.xl,
                high=self.problem.xu,
                size=(self.pop_size - len(self.pop), self.problem.n_var)
            )
            补充_pop = Population.new("X", 补充_X)
            self.evaluator.eval(self.problem, 补充_pop, algorithm=self)
            self.pop = Population.merge(self.pop, 补充_pop)

        # 保存历史
        if self.save_history:
            self.history.append(self.pop.copy())

        # 打印初始化信息
        init_F = self.pop.get("F").flatten()
        print(f"DGGA初始化完成，种群数量：{len(self.pop)}")
        print(f"初始适应度范围：[{np.min(init_F):.2f}, {np.max(init_F):.2f}]")
        print(f"初始最优适应度：{np.min(init_F):.4f}")

    def _next(self):
        """单轮迭代（修复交叉算子配对问题）"""
        # 1. 确保种群非空
        if len(self.pop) == 0:
            self._initialize()
        X = self.pop.get("X")
        F = self.pop.get("F").flatten()

        # 2. 计算多样性
        diversity = self._calculate_diversity(X)
        print(f"种群多样性：{diversity:.4f}")

        # 3. 自适应调整概率
        cross_prob, mutate_prob = self._adaptive_prob(diversity)
        print(f"交叉概率：{cross_prob:.4f}，变异概率：{mutate_prob:.4f}")

        # 4. 选择父代（确保数量为偶数，适配SBX配对）
        selected_X = self._selection(X, F, n_select=self.n_offsprings)
        # 确保父代数量为偶数（SBX需要成对）
        if len(selected_X) % 2 != 0:
            selected_X = np.vstack([selected_X, selected_X[-1]])

        # ========== 核心修复：SBX交叉算子正确配对 ==========
        # 方式1：直接用数组交叉（绕过pymoo的Population配对逻辑）
        offspring_X = self._sbx_crossover(selected_X, cross_prob, self.crossover_eta)

        # 方式2：变异（直接对数组操作）
        offspring_X = self._pm_mutation(offspring_X, mutate_prob, self.mutation_eta)

        # 转换为Population对象
        offspring = Population.new("X", offspring_X[:self.n_offsprings])

        # 7. 评估后代（执行CARLA仿真）
        self.evaluator.eval(self.problem, offspring, algorithm=self)

        # 8. 去重
        if self.eliminate_duplicates:
            offspring_X = offspring.get("X")
            _, unique_idx = np.unique(offspring_X.round(6), axis=0, return_index=True)
            unique_idx = sorted(unique_idx)
            offspring = offspring[unique_idx]

        # 9. 合并父代和子代
        combined = Population.merge(self.pop, offspring)
        combined_F = combined.get("F").flatten()

        # 10. 保留最优的pop_size个个体
        sorted_idx = np.argsort(combined_F)[:self.pop_size]
        self.pop = combined[sorted_idx]

        # 11. 保存历史+打印结果
        if self.save_history:
            self.history.append(self.pop.copy())
        current_F = self.pop.get("F").flatten()
        print(f"本轮最优适应度：{np.min(current_F):.4f}")
        print(f"本轮种群适应度均值：{np.mean(current_F):.4f}")

    def _sbx_crossover(self, X, prob, eta):
        """手动实现SBX交叉（绕过pymoo算子的配对限制）"""
        n, n_var = X.shape
        off = X.copy()

        # 遍历所有成对的父代
        for i in range(0, n, 2):
            # 跳过最后一个（如果数量为奇数）
            if i + 1 >= n:
                break

            # 以prob概率执行交叉
            if np.random.random() <= prob:
                x1 = X[i]
                x2 = X[i + 1]

                # SBX核心公式
                for j in range(n_var):
                    if np.random.random() <= 0.5:
                        if abs(x1[j] - x2[j]) > 1e-14:
                            if x1[j] < x2[j]:
                                xl, xu = self.problem.xl[j], self.problem.xu[j]
                                beta = 1.0 + (2.0 * (x1[j] - xl)) / (x2[j] - x1[j])
                                alpha = 2.0 - beta ** (-(eta + 1.0))
                                rand = np.random.random()
                                if rand <= 1.0 / alpha:
                                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                                else:
                                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                                c1 = 0.5 * (x1[j] + x2[j] - beta_q * (x2[j] - x1[j]))

                                beta = 1.0 + (2.0 * (xu - x2[j])) / (x2[j] - x1[j])
                                alpha = 2.0 - beta ** (-(eta + 1.0))
                                if rand <= 1.0 / alpha:
                                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                                else:
                                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                                c2 = 0.5 * (x1[j] + x2[j] + beta_q * (x2[j] - x1[j]))

                                # 边界限制
                                c1 = np.clip(c1, xl, xu)
                                c2 = np.clip(c2, xl, xu)

                                # 交换
                                off[i, j] = c1
                                off[i + 1, j] = c2

        return off

    def _pm_mutation(self, X, prob, eta):
        """手动实现PM变异"""
        n, n_var = X.shape
        off = X.copy()

        for i in range(n):
            for j in range(n_var):
                if np.random.random() <= prob:
                    x = X[i, j]
                    xl, xu = self.problem.xl[j], self.problem.xu[j]

                    # PM核心公式
                    delta1 = (x - xl) / (xu - xl)
                    delta2 = (xu - x) / (xu - xl)
                    rand = np.random.random()
                    mut_pow = 1.0 / (eta + 1.0)

                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                        delta_q = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                        delta_q = 1.0 - val ** mut_pow

                    x = x + delta_q * (xu - xl)
                    x = np.clip(x, xl, xu)
                    off[i, j] = x

        return off

    def _calculate_diversity(self, X):
        """计算种群多样性"""
        if len(X) < 2:
            return 1.0
        X_norm = (X - self.problem.xl) / (self.problem.xu - self.problem.xl)
        dists = []
        for i in range(len(X_norm)):
            for j in range(i + 1, len(X_norm)):
                dists.append(np.linalg.norm(X_norm[i] - X_norm[j]))
        return np.mean(dists) / np.sqrt(self.problem.n_var) if dists else 0.0

    def _adaptive_prob(self, diversity):
        """自适应调整概率"""
        if diversity < self.diversity_threshold:
            cross_prob = self.crossover_prob * 0.8
            mutate_prob = self.mutation_prob * 1.5
        else:
            cross_prob = self.crossover_prob
            mutate_prob = self.mutation_prob
        return np.clip(cross_prob, 0.5, 0.9), np.clip(mutate_prob, 0.01, 0.3)

    def _selection(self, X, F, n_select):
        """精英+轮盘赌选择"""
        # 精英保留
        elite_idx = np.argsort(F)[:self.elite_num]
        elite = X[elite_idx]

        # 轮盘赌选择剩余个体
        rest_size = max(0, n_select - len(elite))
        if rest_size == 0:
            return elite[:n_select]

        # 适应度归一化
        F_norm = 1 / (F - np.min(F) + 1e-6)
        F_norm = F_norm / F_norm.sum()

        # 轮盘赌采样
        try:
            rest_idx = np.random.choice(len(X), size=rest_size, p=F_norm, replace=True)
        except:
            rest_idx = np.random.choice(len(X), size=rest_size, replace=True)

        # 合并并返回
        selected = np.vstack([elite, X[rest_idx]])
        return selected[:n_select]

    def _get_result(self):
        """生成最终结果"""
        res = Result()
        res.algorithm = self
        res.problem = self.problem
        res.start_time = self.start_time
        res.end_time = self.end_time
        res.exec_time = self.end_time - self.start_time
        res.history = self.history

        # 确保返回有效结果
        final_F = self.pop.get("F").flatten()
        best_idx = np.argmin(final_F)
        res.X = self.pop.get("X")[best_idx]
        res.F = self.pop.get("F")[best_idx]
        res.pop = self.pop

        # 打印最终结果
        print("\n=== DGGA 迭代完成 ===")
        print(f"总迭代轮数：{self.n_gen}")
        print(f"最终最优参数：{res.X}")
        print(f"最终最优适应度：{res.F[0]:.4f}")
        print(f"总执行时间：{res.exec_time:.2f}秒")

        return res
