from typing import List, Dict, Union, Optional, Tuple
from enum import Enum
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from IOData.IOData import IOData, InputRule
from .simple_track_generator import trackGenerator


class PlotStyle(Enum):
    """Enum for plot styles
    """
    TRAJECTORY = 1
    CONSTRAINT = 2
    TRAJECTORY_SLICE = 3


class Results:
    """Class containing results of simualtion and plotting methods
    """
    Ts: float
    num: int 
    # recorded input data [tau, delta]
    _input_obj: List[np.ndarray]
    _input_applied: List[np.ndarray]

    # recorded vehicle trajectory, including global and error states
    global_trajectory_noised: bool = False # indicate whether the global trajectory is noised
    _global_trajectory: List[np.ndarray]
    _global_trajectory_noise: List[np.ndarray]
    _global_trajectory_slices: List[Tuple[float, List[np.ndarray]]] # trajectory slices, (start time, trajectory)

    error_trajectory_noised: bool = False # indicate whether the error trajectory is noised
    _error_trajectory: List[np.ndarray]
    _segment_index_list: List[int] # index of the segment
    _error_trajectory_noise: List[np.ndarray]
    _error_trajectory_slices: List[Tuple[float, List[np.ndarray]]] # trajectory slices, (start time, trajectory)
    _predicted_error_trajectory_slices: List[Tuple[float, List[np.ndarray]]] # trajectory slices, (start time, trajectory)
    _predicted_error_with_slack_slices: List[Tuple[float, List[np.ndarray]]] # trajectory slices added with slack varialbes, (start time, trajectory)
    _proposed_input_slices: List[Tuple[float, List[np.ndarray]]] = [] # proposed input slices, (start time, trajectory)

    _calculation_time: List[float]
    _opt_value_list: List[float]
    _sigma_value_list: List[np.ndarray]

    # default styles of plotting
    trajecory_style: Dict = dict(color='tab:blue', linewidth=2, linestyle='-')
    constraint_style: Dict = dict(color='black', linewidth=1, linestyle=':')
    slice_style: Dict = dict(color='black', linewidth=1, linestyle='--')

    # default font size
    title_size: int = 24
    label_size: int = 22
    major_tick_size: int = 18
    minor_tick_size: int = 10

    # time_steps of violating constraints
    violating_time_steps: List[float] = []
    # time steps of returning to the track
    return_time_steps: List[float] = []
    # time spans of violating constraints
    violating_time_spans: List[Tuple[float, float]] = []
    violating_time: float = 0.0
    lap_time: float = 0.0
    mark_time_steps: List[int] = []
    # simulation time
    simulation_time: float = 0.0
    saved_dataset_name: str

    def __init__(self, Ts: float) -> None:
        """Contrainer for simulation results and 
        """
        self.Ts = Ts

        self._input_obj: List[np.ndarray] = []
        self._input_applied: List[np.ndarray] = []

        self.global_trajectory_noised: bool = False
        self._global_trajectory: List[np.ndarray] = []
        self._global_trajectory_noise: List[np.ndarray] = []
        self._global_trajectory_slices: List[Tuple[float, List[np.ndarray]]] = []

        self.error_trajectory_noised: bool = False # indicate whether the error trajectory is noised
        self._error_trajectory: List[np.ndarray] = []
        self._segment_index_list: List[int] = []
        self._error_trajectory_noise: List[np.ndarray] = []
        self._error_trajectory_slices: List[Tuple[float, List[np.ndarray]]] = []
        self._predicted_error_trajectory_slices: List[Tuple[float, List[np.ndarray]]] = []
        self._predicted_error_with_slack_slices: List[Tuple[float, List[np.ndarray]]] = []
        self._proposed_input_slices: List[Tuple[float, List[np.ndarray]]] = []

        self._calculation_time: List[float] = []
        self._opt_value_list: List[float] = []
        self._sigma_value_list: List[np.ndarray] = []

        self.violating_time_steps: List[float] = []
        self.mark_time_steps: List[int] = []

        self.return_time_steps: List[float] = []
        # time spans of violating constraints
        self.violating_time_spans: List[Tuple[float, float]] = []
        self.violating_time: float = 0.0
        self.lap_time: float = 0.0

    def add_point(self, input_obj: np.ndarray, input_applied: np.ndarray,
                  global_state: np.ndarray, global_noise: np.ndarray,
                  error_state: np.ndarray, error_noise: np.ndarray,
                  segment_index: int,) -> None:
        """Adds a point to the trajectory data
        """
        self._input_obj.append(np.array(input_obj).flatten())
        self._input_applied.append(np.array(input_applied).flatten())

        self._global_trajectory.append(global_state)
        self._global_trajectory_noise.append(global_noise)

        self._error_trajectory.append(error_state)
        self._error_trajectory_noise.append(error_noise)

        self._segment_index_list.append(segment_index)
    
    def add_global_slice(self, start_time: float, global_slice: List[np.ndarray]) -> None:
        """Adds a global slice to the trajectory data
        """
        # assert (end_time-start_time)/(len(global_slice)-1) == self.Ts, "Slice length does not match the time step"
        self._global_trajectory_slices.append((start_time, global_slice))

    def add_error_slice(self, start_time: float, error_slice: List[np.ndarray]) -> None:
        """Adds a error slice to the trajectory data
        """
        # assert (end_time-start_time)/(len(error_slice)-1) == self.Ts, "Slice length does not match the time step"
        self._error_trajectory_slices.append((start_time, error_slice))

    def add_predicted_error_slice(self, start_time: float, predicted_error_slice: List[np.ndarray]) -> None:
        # assert (end_time-start_time)/(len(predicted_error_slice)-1) == self.Ts, "Slice length does not match the time step"
        self._predicted_error_trajectory_slices.append((start_time, predicted_error_slice))
    
    def add_predicted_error_slack_slice(self, start_time: float, predicted_error_slack_slice: List[np.ndarray]) -> None:
        self._predicted_error_with_slack_slices.append((start_time, predicted_error_slack_slice))
    
    def add_calculation_time(self, calculation_time: float) -> None:
        self._calculation_time.append(calculation_time)
    
    def calculate_mean_calculation_time(self) -> float:
        mean_calculation_time = np.mean(self._calculation_time)
        self.mean_calculation_time = mean_calculation_time
        return mean_calculation_time
    
    def calculate_violation_time(self) -> float:
        """Calculate the total violation time
        """
        self.violating_time = 0.0
        for start, end in zip(self.violating_time_steps, self.return_time_steps):
            self.violating_time = self.violating_time + end - start
            self.violating_time_spans.append((start, end))
        return self.violating_time
    
    def calculate_lap_time(self) -> float:
        indices_at_start_segment = np.nonzero(np.array(self._segment_index_list) == 0)[0]
        if len(indices_at_start_segment) == 0:
            return np.nan
        for i, j in enumerate(indices_at_start_segment, indices_at_start_segment[0]):
            if i!=j:
                lap_time = self.Ts * indices_at_start_segment[i]
                self.lap_time = lap_time
                return lap_time
        return np.nan
    
    def add_opt_value(self, opt_value: float) -> None:
        self._opt_value_list.append(opt_value)
    
    def add_sigma_value(self, sigma_value: float) -> None:
        self._sigma_value_list.append(sigma_value)
    
    def end_simulation(self) -> None:
        """Calculate everything after the simulation
        """
        self.simulation_time  = self.Ts * len(self._input_obj)
        if self.simulation_time == 0:
            print('No simulation data collected! \n')
            return
        self.calculate_intervention()
        self.calculate_mean_calculation_time()
        self.calculate_sigma_infty_value()
        self.calculate_violation_time()
        self.calculate_lap_time()
        
    def calculate_sigma_infty_value(self) -> Tuple[float, float]:
        """Returns the minimum and maximum sigma_infty norm
        """
        self.sigma_infty_list = [np.linalg.norm(sigma, np.inf) for sigma in self._sigma_value_list]
        return min(self.sigma_infty_list), max(self.sigma_infty_list)

    def calculate_intervention(self) -> np.ndarray:
        """Calculate the root mean square intervention
        """
        avg_intervention = np.zeros(self._input_applied[0].shape)
        for i in range(self._input_applied[0].shape[0]):
            avg_intervention[i] = np.sqrt(np.mean(np.square([
                u[i]-u_obj[i] for u, u_obj in zip(self._input_applied, self._input_obj)
                ])))
        self.average_intervention = avg_intervention
        return avg_intervention

    def plot_vehicle_trajectory(self, ax: plt.Axes,
                                gen: Optional[trackGenerator] = None,
                                line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY,
                                plot_marked_steps: bool = True) -> plt.Axes:
        """Plots mass-center trajectory of the vehicle onto the given axe. p_x and p_y are first to elements of global_trajectory

        Args:
            ax (plt.Axes): matplotlib axes
            style (Union[PlotStyle, Dict], optional): style of the plot. Defaults to trajectory style.
        """
        line_style = self.get_line_style(line_style)
        if gen is not None:
            gen.plotPoints(ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=self.major_tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=self.minor_tick_size)
        ax.plot([y[0] for y in self._global_trajectory], [y[1] for y in self._global_trajectory], **line_style)

        if plot_marked_steps:
            for i in self.mark_time_steps:
                ax.scatter(self._global_trajectory[i][0], self._global_trajectory[i][1], c='red', marker='o')
    
        ax.set_title('Vehicle trajectory', fontsize=self.title_size)
        ax.set_xlabel('x [m]', fontsize=self.label_size)
        ax.set_ylabel('y [m]', fontsize=self.label_size)
        return ax
    
    def plot_error_trajectory(self, index: int, ax: plt.Axes,
                              line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY,
                              constraint: Optional[Tuple[float, float]] = None,
                              constraint_style: Union[PlotStyle, Dict] = PlotStyle.CONSTRAINT,
                              start_time: float = 0.0) -> plt.Axes:
        """Plot error_state[index] of the system
        """

        self.plot_time_sequence(ax, start_time, [y[index] for y in self._error_trajectory],
                                line_style, constraint, constraint_style)
        
        if index == 0:
            ax.set_ylabel(r'$e_{lat}$', fontsize=self.label_size)
        elif index == 1:
            ax.set_ylabel(r'$\mu$ [deg]', fontsize=self.label_size)
        elif index == 2:
            ax.set_ylabel(r'$v$ [m/s]', fontsize=self.label_size)
        elif index == 3:
            ax.set_ylabel(r'$l$ [m]', fontsize=self.label_size)
        return ax
    
    def plot_input_applied(self, index: int, ax: plt.Axes,
                           line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY,
                           constraint: Optional[Tuple[float, float]] = None,
                           constraint_style: Union[PlotStyle, Dict] = PlotStyle.CONSTRAINT,
                           start_time: float = 0.0) -> plt.Axes:
        """Plot input_applied[index] of the system
        """
        self.plot_time_sequence(ax, start_time, [u[index] for u in self._input_applied],
                                line_style, constraint, constraint_style)
        
        if index == 0:
            ax.set_ylabel(r'$\tau$ [m/$s^2$]', fontsize=self.label_size)
        elif index == 1:
            ax.set_ylabel(r'$\delta$ [rad]', fontsize=self.label_size)
        return ax
    
    def plot_input_obj(self, index: int, ax: plt.Axes,
                       line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY_SLICE,
                       constraint: Optional[Tuple[float, float]] = None,
                       constraint_style: Union[PlotStyle, Dict] = PlotStyle.CONSTRAINT,
                       start_time: float = 0.0) -> plt.Axes:
        """Plot input_objective[index] of the system
        """
        self.plot_time_sequence(ax, start_time, [u[index] for u in self._input_obj],
                                line_style, constraint, constraint_style)

        if index == 0:
            ax.set_ylabel(r'$\tau$', fontsize=self.label_size)
        elif index == 1:
            ax.set_ylabel(r'$\delta$ [rad]', fontsize=self.label_size)
        return ax
    
    def plot_predicted_error_slices(self, index: int, ax: plt.Axes,
                                    line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY_SLICE,
                                    label_y_axis: bool = False,
                                    time_span: List[Tuple[float, float]] = [],) -> plt.Axes:
        """Plot predicted error_state slices
        """
        line_style = self.get_line_style(line_style)
        for t, trajectory_slice in self._predicted_error_trajectory_slices:
            if self.time_in_span(t, time_span):
                self.plot_time_sequence(ax, t, [y[index] for y in trajectory_slice], line_style)
        if label_y_axis:
            if index == 0:
                ax.set_ylabel(r'$e_{lat}$', fontsize=self.label_size)
            elif index == 1:
                ax.set_ylabel(r'$\mu$ [deg]', fontsize=self.label_size)
            elif index == 2:
                ax.set_ylabel(r'$v$ [m/s]', fontsize=self.label_size)
            elif index == 3:
                ax.set_ylabel(r'$l$ [m]', fontsize=self.label_size)
        ax.plot([], [], **line_style, label='Predicted error trajectory (y)')
        ax.legend()
        return ax
    
    def plot_predicted_error_with_slack_slices(self, index: int, ax: plt.Axes,
                                               line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY_SLICE,
                                              label_y_axis: bool = False,
                                              time_span: List[Tuple[float, float]] = [],) -> plt.Axes:
        line_style = self.get_line_style(line_style)
        for t, trajectory_slice in self._predicted_error_with_slack_slices:
            if self.time_in_span(t, time_span):
                self.plot_time_sequence(ax, t, [y[index] for y in trajectory_slice], line_style)
        if label_y_axis:
            if index == 0:
                ax.set_ylabel(r'$e_{lat}$', fontsize=self.label_size)
            elif index == 1:
                ax.set_ylabel(r'$\mu$ [deg]', fontsize=self.label_size)
            elif index == 2:
                ax.set_ylabel(r'$v$ [m/s]', fontsize=self.label_size)
            elif index == 3:
                ax.set_ylabel(r'$l$ [m]', fontsize=self.label_size)
        ax.plot([], [], **line_style, label=r'Predicted error trajectory with slack (y-$\sigma$)')
        ax.legend()
        return ax

    def plot_error_slices(self, index: int, ax: plt.Axes,
                                    line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY_SLICE,
                                    label_y_axis: bool = False,
                                    time_span: List[Tuple[float, float]] = [],) -> plt.Axes:
        """Plot predicted error_state slices
        """
        line_style = self.get_line_style(line_style)
        for t, trajectory_slice in self._error_trajectory_slices:
            if self.time_in_span(t, time_span):
                self.plot_time_sequence(ax, t, [y[index] for y in trajectory_slice], line_style)
        if label_y_axis:
            if index == 0:
                ax.set_ylabel(r'$e_{lat}$', fontsize=self.label_size)
            elif index == 1:
                ax.set_ylabel(r'$\mu$ [deg]', fontsize=self.label_size)
            elif index == 2:
                ax.set_ylabel(r'$v$ [m/s]', fontsize=self.label_size)
            elif index == 3:
                ax.set_ylabel(r'$l$ [m]', fontsize=self.label_size)
        ax.plot([], [], **line_style, label='Real trajectory')
        ax.legend()
        return ax

    def time_in_span(self, t: float, time_span: List[Tuple[float, float]]) -> bool:
        """Returns true if t is in the time span
        If time_span is empty, returns True
        """
        if not time_span:
            return True
        for start, end in time_span:
            if t >= start and t <= end:
                return True
        return False

    def plot_time_sequence(self, ax: plt.Axes,
                           start_time: float, data: List[float],
                           line_style: Union[PlotStyle, Dict] = PlotStyle.TRAJECTORY,
                           constraint: Optional[Tuple[float, float]] = None,
                           constraint_style: Union[PlotStyle, Dict] = PlotStyle.CONSTRAINT,) -> plt.Axes:
        """Plot a time sequence
        """
        line_style = self.get_line_style(line_style)

        ax.tick_params(axis='both', which='major', labelsize=self.major_tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=self.minor_tick_size)
        num = len(data)
        ax.plot(np.arange(num)*self.Ts + start_time, data, **line_style)
        ax.set_xlabel('time [s]', fontsize=self.label_size)

        if constraint is not None:
            constraint_style = self.get_line_style(constraint_style)
            ax.plot([start_time, start_time+self.Ts*(num-1)],
                    [constraint[0], constraint[0]], **constraint_style)
            ax.plot([start_time, start_time+self.Ts*(num-1)],
                    [constraint[1], constraint[1]], **constraint_style)
        return ax

    def get_line_style(self, style: Union[PlotStyle, Dict]) -> Dict:
        """Returns the style of the line

        Args:
            style (Union[PlotStyle, Dict]): style of the line

        Returns:
            Dict: style of the line
        """
        if isinstance(style, PlotStyle):
            if style ==  PlotStyle.TRAJECTORY:
                return self.trajecory_style
            elif style == PlotStyle.CONSTRAINT:
                return self.constraint_style
            elif style == PlotStyle.TRAJECTORY_SLICE:
                return self.slice_style
        else:
            return_style = deepcopy(self.trajecory_style)
            for key, value in style.items():
                if key == 'color':
                    return_style[key] = value
                elif key == 'linestyle':
                    return_style[key] = value
                elif key == 'linewidth':
                    return_style[key] = value
        return return_style
