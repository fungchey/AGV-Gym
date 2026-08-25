from enum import IntEnum
from typing import Any, Callable, Tuple, Union, Optional
import json
import logging
import pkg_resources
from scipy.spatial import distance
import numpy as np
import pandas as pd


class Jobs(IntEnum):
    """Defines the different job types that AGVs may have."""
    NULL = -1
    IDLE = 0
    REPOSITION = 1
    SETUP = 2
    PROCESS = 3
    CHARGING = 4


class AGVGeometry():
    """
    Defines the layout geometry for the AGV dispatching environment.
    Handles station data loading, parameter storage, and spatial calculations.
    """

    def __init__(self, seed: int, params: dict, num_vehicles: Optional[int] = None):
        """Initializes AGV geometry with stations and parameters."""
        self.seed = seed
        self.params = params
        self.num_vehicles = num_vehicles if num_vehicles is not None else self.params.get('fleet_size', [9])[0]
        
        # Load stations using parameters
        self._load_stations()
        
        # Initialize random generators first (needed for vehicle positions)
        self._initial_seeding()
        # Initialize vehicle positions at valid stations
        self._init_vehicle_positions()

    def _initial_seeding(self):
        """Initialize random number generators."""
        self._seed_spawner = np.random.SeedSequence(self.seed)
        self.vehicle_sampler = None
        self.request_sampler = None
        self.reseed()
    
    def reseed(self):
        """Reseed all RNGs for new episode."""
        spawns = self._seed_spawner.spawn(2)
        self.vehicle_sampler = np.random.default_rng(spawns[0])
        self.request_sampler = np.random.default_rng(spawns[1])

    def _load_stations(self) -> None:
        """Load AGV stations from CSV file."""
        with pkg_resources.resource_stream(__name__, 'agv_data/KWME_stations.csv') as stream:
            self.stations = pd.read_csv(stream)
        
        # Clean and prepare station data
        self.stations = self.stations.rename(columns={
            'X': 'x',
            'Y': 'y',
            'Type': 'type',
            'Stations number': 'stations_number'
        })
        
        # Map station type abbreviations to full names using parameters
        type_map = {list(t.items())[0][0]: list(t.items())[0][1] 
                   for t in self.params['stations_types']}
        self.stations['type'] = self.stations['type'].map(type_map)
        
        # Filter stations by type
        self.pd_stations = self.stations[self.stations['type'] == 'Pickup and Delivery'].copy()
        self.charging_stations = self.stations[self.stations['type'] == 'Charging station'].copy()
        self.parking_stations = self.stations[self.stations['type'] == 'Parking station'].copy()
        
        # Filter Parking stations and Charging stations for initial placement
        self.init_stations = self.stations[self.stations['type'].isin(['Parking station', 'Charging station'])].copy()
        
        # Total number of stations
        self.num_stations = len(self.stations)

        # Set coordinate ranges with margin
        self._xrange = (float(self.stations['x'].min() - 30), float(self.stations['x'].max() + 30))
        self._yrange = (float(self.stations['y'].min() - 30), float(self.stations['y'].max() + 30))

    def _init_vehicle_positions(self) -> None:
        """Randomly places vehicles at Parking or Charging stations."""
        num_vehicles = self.num_vehicles
        if not isinstance(num_vehicles, int) or num_vehicles <= 0:
            logging.warning(f"Invalid fleet_size {num_vehicles}, defaulting to 1.")
            num_vehicles = 1

        if self.init_stations.empty:
            logging.error("No Parking or Charging stations available for initial vehicle placement.")
            fallback_loc = self.stations.iloc[[0]][['x', 'y']].to_numpy()
            self.vehicle_locations = np.repeat(fallback_loc, num_vehicles, axis=0)
            return

        num_init_stations = len(self.init_stations)
        replace = num_vehicles > num_init_stations
        chosen_station_indices = self.vehicle_sampler.choice(
            self.init_stations.index,
            size=num_vehicles,
            replace=replace
        )
        self.vehicle_locations = self.stations.loc[chosen_station_indices, ['x', 'y']].to_numpy().astype(np.float64)

    @property
    def lots(self):
        """Returns all station locations, analogous to 'lots'."""
        return self.stations[['x', 'y']].copy()

    @property
    def x_range(self):
        return self._xrange

    @property
    def y_range(self):
        return self._yrange

    def dist(self, o: np.ndarray, d: np.ndarray, pairwise: bool = False) -> Union[np.ndarray, float]:
        """Compute Manhattan distance between points."""
        return self._dist_manhattan(o, d, pairwise)

    def _dist_manhattan(self, o: np.ndarray, d: np.ndarray, pairwise: bool) -> np.ndarray:
        """Calculate Manhattan distance between coordinates."""
        o = np.asarray(o, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)

        if o.shape == (2,) and d.shape == (2,):
            return np.abs(o - d).sum()
        
        if o.ndim == 1:
            o = o.reshape(1, 2)
        if d.ndim == 1:
            d = d.reshape(1, 2)

        if pairwise or o.shape != d.shape:
            result = distance.cdist(o, d, "cityblock")
            return result.flatten() if 1 in result.shape else result

        result = np.abs(o - d).sum(axis=1)
        return result[0] if result.size == 1 else result

    def travel_time(self, o: np.ndarray, d: np.ndarray, mean: bool = False, pairwise: bool = False) -> Union[np.ndarray, float]:
        """Compute travel time using speed from parameters."""
        speed = float(self.params['Travel_speed']['speed'])
        if speed <= 0:
            raise ValueError("Travel speed must be positive.")
        distance_val = self.dist(o, d, pairwise=pairwise)
        return distance_val / speed

    def get_nearest_charging_station(self, loc: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """Find the nearest charging station to a given coordinate."""
        loc = np.asarray(loc).reshape(1, 2)
        c_locs = self.charging_stations[['x', 'y']].to_numpy()
        dists = distance.cdist(loc, c_locs, "cityblock")[0]
        nearest_idx_in_c = np.argmin(dists)
        station_global_idx = self.charging_stations.index[nearest_idx_in_c]
        station_loc = c_locs[nearest_idx_in_c]
        min_dist = dists[nearest_idx_in_c]
        return station_global_idx, station_loc, min_dist
