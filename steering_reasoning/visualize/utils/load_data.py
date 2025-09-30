import os
import pickle
from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import numpy as np
from loguru import logger

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.defs import Model, ModelGroup


def nanstd(values, mean, divisor):
    if np.all((divisor == 0) | (divisor == 1)):
        return 0

    var = np.nansum((values - mean) ** 2) / (divisor - 1)
    std = np.sqrt(var)

    return float(std)


def get_accuracy(problems: List[Problem]):
    rewards = [problem.solutions[0].reward for problem in problems]

    return np.mean(rewards)


def get_avg_accuracy(problems: List[Problem]):
    avg_rewards = []
    for problem in problems:
        avg_reward = np.mean([solution.reward for solution in problem.solutions])

        avg_rewards.append(avg_reward)

    return np.mean(avg_rewards)


def get_values_at_seeds(pkl_path: str, bench: str, callback):
    eval_seeds = [x for x in os.listdir(pkl_path) if "eval_seed" in x]
    seeds = []
    for eval_seed in eval_seeds:
        loadpath = os.path.join(pkl_path, eval_seed, f"{bench}.pkl")
        if not os.path.exists(loadpath):
            logger.warning(f"{loadpath} does not exist")
            continue
        with open(loadpath, "rb") as f:
            problems = pickle.load(f)

        value = callback(problems=problems)
        seeds.append(value)

    if len(seeds) > 0:
        seeds = np.stack(seeds)

    return seeds


def avg_accuracy_table(
    model_group: ModelGroup, bench: str
) -> Tuple[Dict[Model, Dict[str, float]], Dict[Model, np.ndarray]]:
    accuracies_dict = {}
    accuracies_dict_values = {}
    accuracies_mean = []
    accuracies_std = []
    labels = []

    for model in model_group.models:
        if model.pkl_path is None or not os.path.exists(model.pkl_path):
            logger.warning(f"{model.pkl_path} does not exist")
            continue

        is_greedy = "temp_0.0" in model.pkl_path
        values_at_seeds = get_values_at_seeds(
            pkl_path=model.pkl_path, bench=bench, callback=get_avg_accuracy
        )
        values_at_seeds *= 100
        if (
            not is_greedy
            and len(values_at_seeds) < 3
            or is_greedy
            and len(values_at_seeds) == 0
        ):
            continue
        mean = values_at_seeds.mean(0)
        std = values_at_seeds.std(0)

        # * 100 -> percent
        accuracies_mean.append(mean)
        accuracies_std.append(std)
        labels.append(model.label)

        accuracies_dict[model] = {"mean": mean, "std": std}
        accuracies_dict_values[model] = values_at_seeds

    return accuracies_dict, accuracies_dict_values


def accuracy_table(
    model_group: ModelGroup, bench: str
) -> Tuple[Dict[Model, Dict[str, float]], Dict[Model, np.ndarray]]:
    accuracies_dict = {}
    accuracies_dict_values = {}
    accuracies_mean = []
    accuracies_std = []
    labels = []

    for model in model_group.models:
        if model.pkl_path is None or not os.path.exists(model.pkl_path):
            logger.warning(f"{model.pkl_path} does not exist")
            continue
        is_greedy = "temp_0.0" in model.pkl_path

        values_at_seeds = get_values_at_seeds(
            pkl_path=model.pkl_path, bench=bench, callback=get_accuracy
        )
        values_at_seeds *= 100
        if (
            not is_greedy
            and len(values_at_seeds) < 3
            or is_greedy
            and len(values_at_seeds) == 0
        ):
            continue

        mean = values_at_seeds.mean(0)
        std = values_at_seeds.std(0)

        # * 100 -> percent
        accuracies_mean.append(mean)
        accuracies_std.append(std)
        labels.append(model.label)

        accuracies_dict[model] = {"mean": mean, "std": std}
        accuracies_dict_values[model] = values_at_seeds

    return accuracies_dict, accuracies_dict_values


def get_lens_at_1_max(problems: List[Problem], condition):
    lens = []
    for problem in problems:
        for solution in problem.solutions:
            if condition(solution):
                lens.append(solution.gen_length)

    return np.max(lens) if len(lens) != 0 else np.nan


def get_lens_at_1(problems: List[Problem], condition):
    lens = []
    for problem in problems:
        for solution in problem.solutions:
            if condition(solution):
                lens.append(solution.gen_length)

    return np.mean(lens)


def length_at_1_table(
    model_group: ModelGroup, bench: str
) -> Tuple[
    Dict[Model, Dict[str, float]],
    Dict[Model, np.ndarray],
    Dict[Model, Dict[str, float]],
    Dict[Model, np.ndarray],
    Dict[Model, Dict[str, float]],
    Dict[Model, np.ndarray],
    Dict[Model, Dict[str, float]],
    Dict[Model, np.ndarray],
]:
    all_lens_dict = {}
    all_lens_dict_values = {}
    all_correct_lens_dict = {}
    all_correct_lens_dict_values = {}
    all_max_correct_lens_dict = {}
    all_max_correct_lens_dict_values = {}
    all_incorrect_lens_dict = {}
    all_incorrect_lens_dict_values = {}
    all_lens_mean = []
    all_lens_std = []
    all_correct_lens_mean = []
    all_correct_lens_std = []
    all_max_correct_lens_mean = []
    all_max_correct_lens_std = []
    all_incorrect_lens_mean = []
    all_incorrect_lens_std = []
    labels = []

    for model in model_group.models:
        if model.pkl_path is None or not os.path.exists(model.pkl_path):
            logger.warning(f"{model.pkl_path} does not exist")
            continue
        is_greedy = "temp_0.0" in model.pkl_path

        # All lens
        lens = get_values_at_seeds(
            pkl_path=model.pkl_path,
            bench=bench,
            callback=lambda problems: get_lens_at_1(problems, lambda x: True),
        )
        if not is_greedy and len(lens) < 3 or is_greedy and len(lens) == 0:
            continue

        lens_mean = np.nanmean(lens)
        lens_std = nanstd(lens, mean=lens_mean, divisor=np.sum(~np.isnan(lens)))

        all_lens_mean.append(lens_mean)
        all_lens_std.append(lens_std)

        all_lens_dict[model] = {"mean": lens_mean, "std": lens_std}
        all_lens_dict_values[model] = lens

        # Correct lens
        correct_lens = get_values_at_seeds(
            pkl_path=model.pkl_path,
            bench=bench,
            callback=lambda problems: get_lens_at_1(problems, lambda x: x.reward == 1),
        )
        if (
            not is_greedy
            and len(correct_lens) < 3
            or is_greedy
            and len(correct_lens) == 0
        ):
            continue
        correct_lens_mean = np.nanmean(correct_lens, 0)
        correct_lens_std = nanstd(
            correct_lens,
            mean=correct_lens_mean,
            divisor=np.sum(~np.isnan(correct_lens)),
        )

        all_correct_lens_mean.append(correct_lens_mean)
        all_correct_lens_std.append(correct_lens_std)

        all_correct_lens_dict[model] = {
            "mean": correct_lens_mean,
            "std": correct_lens_std,
        }
        all_correct_lens_dict_values[model] = correct_lens

        # Correct lens
        max_correct_lens = get_values_at_seeds(
            pkl_path=model.pkl_path,
            bench=bench,
            callback=lambda problems: get_lens_at_1_max(
                problems, lambda x: x.reward == 1
            ),
        )
        if (
            not is_greedy
            and len(max_correct_lens) < 3
            or is_greedy
            and len(max_correct_lens) == 0
        ):
            continue
        max_correct_lens_mean = np.nanmean(max_correct_lens, 0)
        max_correct_lens_std = nanstd(
            max_correct_lens,
            mean=max_correct_lens_mean,
            divisor=np.sum(~np.isnan(max_correct_lens)),
        )

        all_max_correct_lens_mean.append(max_correct_lens_mean)
        all_max_correct_lens_std.append(max_correct_lens_std)

        all_max_correct_lens_dict[model] = {
            "mean": max_correct_lens_mean,
            "std": max_correct_lens_std,
        }
        all_max_correct_lens_dict_values[model] = max_correct_lens

        # Incorrect lens
        incorrect_lens = get_values_at_seeds(
            pkl_path=model.pkl_path,
            bench=bench,
            callback=lambda problems: get_lens_at_1(problems, lambda x: x.reward == 0),
        )

        if (
            not is_greedy
            and len(incorrect_lens) < 3
            or is_greedy
            and len(incorrect_lens) == 0
        ):
            continue
        incorrect_lens_mean = np.nanmean(incorrect_lens, 0)
        incorrect_lens_std = nanstd(
            incorrect_lens,
            mean=incorrect_lens_mean,
            divisor=np.sum(~np.isnan(incorrect_lens)),
        )

        all_incorrect_lens_mean.append(incorrect_lens_mean)
        all_incorrect_lens_std.append(incorrect_lens_std)

        all_incorrect_lens_dict[model] = {
            "mean": incorrect_lens_mean,
            "std": incorrect_lens_std,
        }
        all_incorrect_lens_dict_values[model] = incorrect_lens

        labels.append(model.label)

    return (
        all_lens_dict,
        all_lens_dict_values,
        all_correct_lens_dict,
        all_correct_lens_dict_values,
        all_max_correct_lens_dict,
        all_max_correct_lens_dict_values,
        all_incorrect_lens_dict,
        all_incorrect_lens_dict_values,
    )


def run_for_model_groups(
    model_groups: List[ModelGroup],
    benches: List[str],
    average_across_benches: Dict[Literal["acc", "avg_acc"], List[str]],
) -> Tuple[
    Dict[str, Dict[Model, Dict[str, float]]],
    Dict[str, Dict[Model, Dict[str, float]]],
    Dict[str, Dict[Model, Dict[str, float]]],
    Dict[str, Dict[Model, Dict[str, float]]],
    Dict[str, Dict[Model, Dict[str, float]]],
    Dict[str, Dict[Model, Dict[str, float]]],
]:
    accuracies_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(dict)
    accuracies_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(dict)

    avg_accuracies_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(dict)
    avg_accuracies_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(dict)

    all_lens_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(dict)
    all_lens_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(dict)
    all_correct_lens_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(dict)
    all_correct_lens_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(dict)
    all_max_correct_lens_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(
        dict
    )
    all_max_correct_lens_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(
        dict
    )
    all_incorrect_lens_dict: Dict[str, Dict[Model, Dict[str, float]]] = defaultdict(
        dict
    )
    all_incorrect_lens_dict_values: Dict[str, Dict[Model, np.ndarray]] = defaultdict(
        dict
    )
    for model_group in model_groups:
        for bench in benches:
            single_accuracies_dict, single_accuracies_dict_values = accuracy_table(
                model_group=model_group, bench=bench
            )
            single_avg_accuracies_dict, single_avg_accuracies_dict_values = (
                avg_accuracy_table(model_group=model_group, bench=bench)
            )

            (
                single_all_lens_dict,
                single_all_lens_dict_values,
                single_all_correct_lens_dict,
                single_all_correct_lens_dict_values,
                single_all_max_correct_lens_dict,
                single_all_max_correct_lens_dict_values,
                single_all_incorrect_lens_dict,
                single_all_incorrect_lens_dict_values,
            ) = length_at_1_table(model_group=model_group, bench=bench)
            bench = bench.replace(".pkl", "")
            accuracies_dict[bench] = {
                **accuracies_dict[bench],
                **single_accuracies_dict,
            }
            accuracies_dict_values[bench] = {
                **accuracies_dict_values[bench],
                **single_accuracies_dict_values,
            }

            avg_accuracies_dict[bench] = {
                **avg_accuracies_dict[bench],
                **single_avg_accuracies_dict,
            }
            avg_accuracies_dict_values[bench] = {
                **avg_accuracies_dict_values[bench],
                **single_avg_accuracies_dict_values,
            }

            all_lens_dict[bench] = {**all_lens_dict[bench], **single_all_lens_dict}
            all_lens_dict_values[bench] = {
                **all_lens_dict_values[bench],
                **single_all_lens_dict_values,
            }

            all_correct_lens_dict[bench] = {
                **all_correct_lens_dict[bench],
                **single_all_correct_lens_dict,
            }
            all_correct_lens_dict_values[bench] = {
                **all_correct_lens_dict_values[bench],
                **single_all_correct_lens_dict_values,
            }

            all_max_correct_lens_dict[bench] = {
                **all_max_correct_lens_dict[bench],
                **single_all_max_correct_lens_dict,
            }
            all_max_correct_lens_dict_values[bench] = {
                **all_max_correct_lens_dict_values[bench],
                **single_all_max_correct_lens_dict_values,
            }

            all_incorrect_lens_dict[bench] = {
                **all_incorrect_lens_dict[bench],
                **single_all_incorrect_lens_dict,
            }
            all_incorrect_lens_dict_values[bench] = {
                **all_incorrect_lens_dict_values[bench],
                **single_all_incorrect_lens_dict_values,
            }

    for model in accuracies_dict_values[benches[0]].keys():
        values = [
            accuracies_dict_values[bench][model]
            for bench in average_across_benches["acc"]
            if model in accuracies_dict_values[bench]
        ] + [
            avg_accuracies_dict_values[bench][model]
            for bench in average_across_benches["avg_acc"]
            if model in avg_accuracies_dict_values[bench]
        ]
        if len(values) != len(benches):
            continue
        assert all(isinstance(v, np.ndarray) for v in values), [type(v) for v in values]
        np_values = np.vstack(values)
        assert np_values.ndim == 2, np_values.ndim

        avg_values = np_values.mean(0)
        mean = avg_values.mean()
        std = avg_values.std()

        avg_dict = {model: {"mean": mean, "std": std}}
        accuracies_dict["Average"] = {**accuracies_dict["Average"], **avg_dict}
        avg_accuracies_dict["Average"] = {**avg_accuracies_dict["Average"], **avg_dict}

        values = [
            all_lens_dict_values[bench][model]
            for bench in benches
            if model in all_lens_dict_values[bench]
        ]
        avg_values = np.nanmean(np.vstack(values), 0)
        all_lens_dict["Average"] = {
            **all_lens_dict["Average"],
            model: {"mean": avg_values.mean(), "std": avg_values.std()},
        }

        values = [
            all_correct_lens_dict_values[bench][model]
            for bench in benches
            if model in all_correct_lens_dict_values[bench]
        ]
        avg_values = np.nanmean(np.vstack(values), 0)
        all_correct_lens_dict["Average"] = {
            **all_correct_lens_dict["Average"],
            model: {"mean": avg_values.mean(), "std": avg_values.std()},
        }

        values = [
            all_max_correct_lens_dict_values[bench][model]
            for bench in benches
            if model in all_max_correct_lens_dict_values[bench]
        ]
        avg_values = np.nanmean(np.vstack(values), 0)
        all_max_correct_lens_dict["Average"] = {
            **all_max_correct_lens_dict["Average"],
            model: {"mean": avg_values.mean(), "std": avg_values.std()},
        }

        values = [
            all_incorrect_lens_dict_values[bench][model]
            for bench in benches
            if model in all_incorrect_lens_dict_values[bench]
        ]
        avg_values = np.nanmean(np.vstack(values), 0)
        all_incorrect_lens_dict["Average"] = {
            **all_incorrect_lens_dict["Average"],
            model: {"mean": avg_values.mean(), "std": avg_values.std()},
        }

        if "Base Model" in model.name and "tau" not in model.name:
            for bench in benches + ["Average"]:
                accuracies_dict[bench][model]["std"] = 0.0
                avg_accuracies_dict[bench][model]["std"] = 0.0
                all_lens_dict[bench][model]["std"] = 0.0
                all_correct_lens_dict[bench][model]["std"] = 0.0
                all_max_correct_lens_dict[bench][model]["std"] = 0.0
                all_incorrect_lens_dict[bench][model]["std"] = 0.0

    return (
        accuracies_dict,
        avg_accuracies_dict,
        all_lens_dict,
        all_correct_lens_dict,
        all_max_correct_lens_dict,
        all_incorrect_lens_dict,
    )
