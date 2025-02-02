from functools import cached_property
from pathlib import Path
from typing import List, NamedTuple, Optional

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from large_gcs.algorithms.search_algorithm import AlgMetrics
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.utils.utils import use_type_1_fonts_in_plots
from large_gcs.visualize.colors import DEEPSKYBLUE3, FLESH


class SingleRunData(NamedTuple):
    cost: float
    wall_clock_time: float
    num_paths_expanded: int


class SamplingSingleRunData(NamedTuple):
    cost: float
    wall_clock_time: float
    num_paths_expanded: int
    num_samples: int


class GgcsSingleRunData(NamedTuple):
    cost: float
    wall_clock_time: float
    num_paths_expanded: int
    source: str
    target: str


class MultirunData:
    def __init__(self, run_data: List[SamplingSingleRunData | SingleRunData]) -> None:
        self.data = run_data

    @classmethod
    def load_from_folder(cls, data_dir: Path) -> "MultirunData":
        run_data = []

        all_runs_are_sampled = True
        # Iterate through the runs and generate figures
        for path in data_dir.iterdir():
            # We only care about the subfolders (which contain data for a run)
            if not path.is_dir():
                continue

            # Load the config file for this run
            config_path = path / "config.yaml"
            if not config_path.is_file():
                raise RuntimeError(f"The config file {config_path}" f"does not exist.")

            cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore

            sol_files = list(path.glob("*solution.pkl"))
            if not len(sol_files) == 1:
                raise RuntimeError(
                    f"Found more or fewer than one solution file in {path}."
                    f"This is not expected, so something is likely wrong."
                    f"{sol_files}"
                )
            sol_file = sol_files[0]
            sol = ShortestPathSolution.load(sol_file)

            metric_files = list(path.glob("*metrics.json"))
            if not len(metric_files) == 1:
                raise RuntimeError(
                    f"Found more or fewer than one metric file in {path}."
                    f"This is not expected, so something is likely wrong."
                    f"{metric_files}"
                )
            metric_file = metric_files[0]
            metrics = AlgMetrics.load(metric_file)

            n_paths_expanded = (
                metrics.n_vertices_expanded + metrics.n_vertices_reexpanded
            )
            if "source" in cfg.keys():
                data = GgcsSingleRunData(
                    sol.cost,
                    metrics.time_wall_clock,
                    n_paths_expanded,
                    OmegaConf.to_container(cfg.source, resolve=False),
                    OmegaConf.to_container(cfg.target, resolve=False),
                )
                all_runs_are_sampled = False
            elif "num_samples_per_vertex" in cfg.domination_checker.keys():
                num_samples = cfg.domination_checker.num_samples_per_vertex

                data = SamplingSingleRunData(
                    sol.cost,
                    metrics.time_wall_clock,
                    n_paths_expanded,
                    num_samples,
                )
            else:
                data = SingleRunData(
                    sol.cost,
                    metrics.time_wall_clock,
                    n_paths_expanded,
                )
                all_runs_are_sampled = False

            run_data.append(data)

        if all_runs_are_sampled:
            # sort data based on num_samples
            run_data = sorted(run_data, key=lambda d: d.num_samples)

        return cls(run_data)

    @cached_property
    def num_samples(self) -> List[int]:
        if not all([type(d) == SamplingSingleRunData for d in self.data]):
            raise RuntimeError(
                "Can only compute number of samples when all data points are from sampling runs"
            )
        return [d.num_samples for d in self.data]  # type: ignore

    @cached_property
    def solve_times(self) -> List[float]:
        return [d.wall_clock_time for d in self.data]

    @cached_property
    def costs(self) -> List[float]:
        return [d.cost for d in self.data]

    @cached_property
    def num_paths_expanded(self) -> List[int]:
        return [d.num_paths_expanded for d in self.data]

    def save(
        self,
        output_path: Path,
    ) -> None:
        """Saves all the datapoints in this run to a JSON file."""
        # Convert to dictionary and save to JSON
        data_dict_list = [data._asdict() for data in self.data]
        import json

        with open(output_path, "w") as f:
            json.dump(data_dict_list, f, indent=4)

    def make_sampling_comparison_plot(
        self,
        ah_baseline: Optional[SingleRunData] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        use_type_1_fonts_in_plots()

        fig_height = 4
        num_plots = 3
        fig, axs = plt.subplots(
            1, num_plots, figsize=(fig_height * num_plots, fig_height)
        )

        sampling_color = DEEPSKYBLUE3.diffuse()
        comparison_color = FLESH.diffuse()

        for ax in axs:
            ax.set_xlabel("Number of samples")

        axs[0].plot(self.num_samples, self.solve_times, color=sampling_color)
        axs[0].set_title("Solve time [s]")

        axs[1].plot(self.num_samples, self.num_paths_expanded, color=sampling_color)
        axs[1].set_title("Number of paths expanded")

        axs[2].plot(self.num_samples, self.costs, color=sampling_color)
        axs[2].set_title("Cost")

        if ah_baseline:
            axs[0].axhline(ah_baseline.wall_clock_time, color=comparison_color)
            axs[1].axhline(ah_baseline.num_paths_expanded, color=comparison_color)
            axs[2].axhline(ah_baseline.cost, color=comparison_color)

            axs[2].legend(["Sampling", "AH-containment"])

        if output_path:
            fig.savefig(output_path)
            plt.close()
        else:
            plt.show()
