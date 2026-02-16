"""Top-level package for EdgeFL."""

from .data import (
    EdgeDataset,
    RegionProfile,
    SeasonalComponent,
    TargetParams,
    VariableParams,
    generate_clients_data,
    generate_environmental_data,
    generate_mixed_environmental_data,
    load_region_profile,
    load_region_profiles,
)
from .agents import (
    Agent,
    AgentCoordinator,
    AgentState,
    AgentUpdate,
    AlwaysParticipatePolicy,
    LocalTrainingPolicy,
    ParticipationPolicy,
    ReportingPolicy,
    StateReportingPolicy,
    TrainingPolicy,
)
from .models import (
    BaseModel,
    SklearnLinearModel,
    SklearnRandomForestModel,
    TorchLinearModel,
    TorchModelAdapter,
)
from .nodes import ClientNode
from .server import AggregationStrategy, FederatedServer
from .simulation.simulation import (
    SimulationResult,
    run_agent_simulation,
    run_simulation,
)
from .utils import (
    ModelVersion,
    ModelVersionTracker,
    bytes_to_weights,
    weights_to_bytes,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AggregationStrategy",
    "Agent",
    "AgentCoordinator",
    "AgentState",
    "AgentUpdate",
    "AlwaysParticipatePolicy",
    "BaseModel",
    "ClientNode",
    "FederatedServer",
    "LocalTrainingPolicy",
    "ModelVersion",
    "ModelVersionTracker",
    "ParticipationPolicy",
    "ReportingPolicy",
    "StateReportingPolicy",
    "SklearnLinearModel",
    "SklearnRandomForestModel",
    "bytes_to_weights",
    "EdgeDataset",
    "RegionProfile",
    "SeasonalComponent",
    "TargetParams",
    "VariableParams",
    "generate_clients_data",
    "generate_environmental_data",
    "generate_mixed_environmental_data",
    "load_region_profile",
    "load_region_profiles",
    "run_simulation",
    "run_agent_simulation",
    "SimulationResult",
    "TrainingPolicy",
    "weights_to_bytes",
]

if TorchLinearModel is not None and TorchModelAdapter is not None:  # pragma: no cover
    __all__ += ["TorchLinearModel", "TorchModelAdapter"]
