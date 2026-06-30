from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp

try:
    from data_process.o3d_utils import build_radius_index, search_hybrid_neighbors
except ImportError:  # pragma: no cover - author layout
    from o3d_utils import build_radius_index, search_hybrid_neighbors
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import warp as wp
import cma
import pickle
import os


class OptimizerCMA:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame,
        mask_path=None,
        velocity_path=None,
        device="cuda:0",
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        if not os.path.exists(f"{cfg.base_dir}/optimizeCMA"):
            # Create directory if it doesn't exist
            os.makedirs(f"{cfg.base_dir}/optimizeCMA")

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            points = np.ascontiguousarray(object_points, dtype=np.float64)
            pcd_tree = build_radius_index(points)

            # Connect the springs of the objects first
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                _k, idx, _ = search_hybrid_neighbors(
                    pcd_tree, points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                controller_points = np.ascontiguousarray(controller_points, dtype=np.float64)
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    _k, idx, _ = search_hybrid_neighbors(
                        pcd_tree,
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = np.ascontiguousarray(object_points[mask == value], dtype=np.float64)
                temp_tree = build_radius_index(temp_points)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    _k, idx, _ = search_hybrid_neighbors(
                        temp_tree, temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def normalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return (value - min) / (max - min)

    def denormalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return value * (max - min) + min

    def build_initial_cma_parameters(self) -> np.ndarray:
        """Normalized initial CMA vector from cfg defaults (same as optimize())."""
        return np.array(
            [
                self.normalize(cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max),
                self.normalize(cfg.object_radius, 0.01, 0.05),
                self.normalize(cfg.object_max_neighbours, 10, 50),
                self.normalize(cfg.controller_radius, 0.01, 0.08),
                self.normalize(cfg.controller_max_neighbours, 10, 80),
                cfg.collide_elas,
                self.normalize(cfg.collide_fric, 0, 2),
                cfg.collide_object_elas,
                self.normalize(cfg.collide_object_fric, 0, 2),
                self.normalize(cfg.collision_dist, 0.01, 0.05),
                self.normalize(cfg.drag_damping, 0, 20),
                self.normalize(cfg.dashpot_damping, 0, 200),
            ],
            dtype=np.float32,
        )

    def optimal_results_to_cma_parameters(self, optimal_results: dict) -> np.ndarray:
        """Convert denormalized optimal_params.pkl dict to normalized CMA vector."""
        return np.array(
            [
                self.normalize(
                    float(optimal_results["global_spring_Y"]),
                    cfg.spring_Y_min,
                    cfg.spring_Y_max,
                ),
                self.normalize(float(optimal_results["object_radius"]), 0.01, 0.05),
                self.normalize(float(optimal_results["object_max_neighbours"]), 10, 50),
                self.normalize(float(optimal_results["controller_radius"]), 0.01, 0.08),
                self.normalize(
                    float(optimal_results["controller_max_neighbours"]), 10, 80
                ),
                float(optimal_results["collide_elas"]),
                self.normalize(float(optimal_results["collide_fric"]), 0, 2),
                float(optimal_results["collide_object_elas"]),
                self.normalize(float(optimal_results["collide_object_fric"]), 0, 2),
                self.normalize(float(optimal_results["collision_dist"]), 0.01, 0.05),
                self.normalize(float(optimal_results["drag_damping"]), 0, 20),
                self.normalize(float(optimal_results["dashpot_damping"]), 0, 200),
            ],
            dtype=np.float32,
        )

    def render_rollout_video(
        self,
        parameters: np.ndarray | list[float],
        *,
        output_path: str | os.PathLike[str],
        label: str,
    ) -> dict[str, object]:
        """Run one forward rollout and render an author-native diagnostic video."""
        output = os.fspath(output_path)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        params = np.asarray(parameters, dtype=np.float32)
        self.error_func(params, visualize=True, video_path=output)
        return {
            "output_path": output,
            "label": label,
            "frame_count": int(self.dataset.frame_len),
            "status": "ok",
        }

    def optimize(self, max_iter=100):
        x_init = self.build_initial_cma_parameters()

        self.error_func(
            x_init, visualize=True, video_path=f"{cfg.base_dir}/optimizeCMA/init.mp4"
        )

        std = 1 / 6
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        es.optimize(self.error_func, iterations=max_iter)

        # Get the results
        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        logger.info(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")

        final_global_spring_Y = self.denormalize(
            optimal_x[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        final_object_radius = self.denormalize(optimal_x[1], 0.01, 0.05)
        final_object_max_neighbours = int(self.denormalize(optimal_x[2], 10, 50))
        final_controller_radius = self.denormalize(optimal_x[3], 0.01, 0.08)
        final_controller_max_neighbours = int(self.denormalize(optimal_x[4], 10, 80))
        final_collide_elas = optimal_x[5]
        final_collide_fric = self.denormalize(optimal_x[6], 0, 2)
        final_collide_object_elas = optimal_x[7]
        final_collide_object_fric = self.denormalize(optimal_x[8], 0, 2)
        final_collision_dist = self.denormalize(optimal_x[9], 0.01, 0.05)
        final_drag_damping = self.denormalize(optimal_x[10], 0, 20)
        final_dashpot_damping = self.denormalize(optimal_x[11], 0, 200)

        self.error_func(
            optimal_x,
            visualize=True,
            video_path=f"{cfg.base_dir}/optimizeCMA/optimal.mp4",
        )

        optimal_results = {}
        optimal_results["global_spring_Y"] = final_global_spring_Y
        optimal_results["object_radius"] = final_object_radius
        optimal_results["object_max_neighbours"] = final_object_max_neighbours
        optimal_results["controller_radius"] = final_controller_radius
        optimal_results["controller_max_neighbours"] = final_controller_max_neighbours
        optimal_results["collide_elas"] = final_collide_elas
        optimal_results["collide_fric"] = final_collide_fric
        optimal_results["collide_object_elas"] = final_collide_object_elas
        optimal_results["collide_object_fric"] = final_collide_object_fric
        optimal_results["collision_dist"] = final_collision_dist
        optimal_results["drag_damping"] = final_drag_damping
        optimal_results["dashpot_damping"] = final_dashpot_damping

        # Save out all the initialized parameters
        with open(f"{cfg.base_dir}/optimal_params.pkl", "wb") as f:
            pickle.dump(optimal_results, f)

    def error_func(self, parameters, visualize=False, video_path=None):
        global_spring_Y = self.denormalize(
            parameters[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        object_radius = self.denormalize(parameters[1], 0.01, 0.05)
        object_max_neighbours = int(self.denormalize(parameters[2], 10, 50))
        controller_radius = self.denormalize(parameters[3], 0.01, 0.08)
        controller_max_neighbours = int(self.denormalize(parameters[4], 10, 80))
        collide_elas = parameters[5]
        collide_fric = self.denormalize(parameters[6], 0, 2)
        collide_object_elas = parameters[7]
        collide_object_fric = self.denormalize(parameters[8], 0, 2)
        collision_dist = self.denormalize(parameters[9], 0.01, 0.05)
        drag_damping = self.denormalize(parameters[10], 0, 20)
        dashpot_damping = self.denormalize(parameters[11], 0, 200)

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=global_spring_Y,
            collide_elas=collide_elas,
            collide_fric=collide_fric,
            dashpot_damping=dashpot_damping,
            drag_damping=drag_damping,
            collide_object_elas=collide_object_elas,
            collide_object_fric=collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            disable_backward=True,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if visualize == True:
            vertices = [
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
            ]

        if cfg.data_type == "real":
            self.simulator.set_acc_count(False)

        total_loss = 0.0
        if not visualize:
            # Only optimize on the train frames
            max_frame = cfg.train_frame
        else:
            max_frame = self.dataset.frame_len

        for j in range(1, max_frame):
            self.simulator.set_controller_target(j)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(self.simulator.graph)
            else:
                if cfg.data_type == "real":
                    with self.simulator.tape:
                        self.simulator.step()
                        self.simulator.calculate_loss()
                else:
                    with self.simulator.tape:
                        self.simulator.step()
                        self.simulator.calculate_simple_loss()

            if visualize == True:
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())

            if cfg.data_type == "real":
                if wp.to_torch(self.simulator.acc_count, requires_grad=False)[0] == 0:
                    self.simulator.set_acc_count(True)

                # Update the prev_acc used to calculate the acceleration loss
                self.simulator.update_acc()

            loss = wp.to_torch(self.simulator.loss, requires_grad=False)
            total_loss += loss.item()

            self.simulator.clear_loss()
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

        total_loss /= cfg.train_frame - 1

        if visualize == True:
            vertices = torch.stack(vertices, dim=0)
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

        return total_loss