
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Environment:
    name: str
    zone: str
    predicted_carbon: float
    current_carbon: float
    cost_per_hour: float
    latency_ms: float
    available: bool = True

    # Optional resource capacity information.
    # Leave as None until actual resource measurements are available.
    available_cpu_cores: Optional[float] = None
    available_memory_gb: Optional[float] = None


@dataclass
class WorkloadRequirements:
    """Hard constraints that an environment must satisfy."""

    max_latency_ms: Optional[float] = None
    min_cpu_cores: Optional[float] = None
    min_memory_gb: Optional[float] = None


@dataclass
class SchedulingDecision:
    policy: str
    selected_environment: str
    selected_zone: str
    predicted_carbon: float
    estimated_cost_per_hour: float
    estimated_latency_ms: float
    score: Optional[float]
    explanation: str


class MetaScheduler:
    """Compare workload placement policies across candidate environments."""

    def __init__(
        self,
        environments: list[Environment],
        carbon_weight: float = 1 / 3,
        cost_weight: float = 1 / 3,
        latency_weight: float = 1 / 3,
    ):
        if not environments:
            raise ValueError("At least one environment is required")

        weights = [
            carbon_weight,
            cost_weight,
            latency_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative")

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1")

        self.environments = environments
        self.weights = weights
        self.round_robin_index = 0

    def _eligible(
        self,
        requirements: Optional[WorkloadRequirements] = None,
    ):
        """Return available environments that satisfy hard constraints."""

        requirements = requirements or WorkloadRequirements()

        candidates = []

        for env in self.environments:
            if not env.available:
                continue

            # Enforce the maximum latency requirement.
            if requirements.max_latency_ms is not None:
                if env.latency_ms > requirements.max_latency_ms:
                    continue

            # If a resource requirement is specified, unknown capacity
            # is not sufficient to prove that the environment qualifies.
            if requirements.min_cpu_cores is not None:
                if env.available_cpu_cores is None:
                    continue

                if env.available_cpu_cores < requirements.min_cpu_cores:
                    continue

            if requirements.min_memory_gb is not None:
                if env.available_memory_gb is None:
                    continue

                if env.available_memory_gb < requirements.min_memory_gb:
                    continue

            candidates.append(env)

        if not candidates:
            raise RuntimeError(
                "No eligible environments satisfy the workload "
                "requirements. Check availability, latency, CPU "
                "and memory constraints."
            )

        return candidates

    @staticmethod
    def _decision(
        policy,
        env,
        score=None,
        explanation="",
    ):
        return SchedulingDecision(
            policy=policy,
            selected_environment=env.name,
            selected_zone=env.zone,
            predicted_carbon=env.predicted_carbon,
            estimated_cost_per_hour=env.cost_per_hour,
            estimated_latency_ms=env.latency_ms,
            score=score,
            explanation=explanation,
        )

    def round_robin(self, requirements=None):
        candidates = self._eligible(requirements)

        env = candidates[
            self.round_robin_index % len(candidates)
        ]
        self.round_robin_index += 1

        return self._decision(
            "round_robin",
            env,
            explanation=(
                "Selected by rotating through available environments "
                "that satisfy workload constraints"
            ),
        )

    def carbon_only(self, requirements=None):
        env = min(
            self._eligible(requirements),
            key=lambda e: e.predicted_carbon,
        )

        return self._decision(
            "carbon_only",
            env,
            explanation=(
                "Lowest predicted carbon intensity among "
                "SLA-eligible environments"
            ),
        )

    def cost_only(self, requirements=None):
        env = min(
            self._eligible(requirements),
            key=lambda e: e.cost_per_hour,
        )

        return self._decision(
            "cost_only",
            env,
            explanation=(
                "Lowest estimated hourly cost among "
                "SLA-eligible environments"
            ),
        )

    def reactive_carbon_aware(self, requirements=None):
        env = min(
            self._eligible(requirements),
            key=lambda e: e.current_carbon,
        )

        return self._decision(
            "reactive_carbon_aware",
            env,
            explanation=(
                "Lowest currently observed carbon intensity among "
                "SLA-eligible environments"
            ),
        )

    def multi_objective_predictive(self, requirements=None):
        candidates = self._eligible(requirements)

        def normalise(values):
            low = min(values)
            high = max(values)

            if high == low:
                return [0.0] * len(values)

            return [
                (value - low) / (high - low)
                for value in values
            ]

        carbon_scores = normalise(
            [env.predicted_carbon for env in candidates]
        )
        cost_scores = normalise(
            [env.cost_per_hour for env in candidates]
        )
        latency_scores = normalise(
            [env.latency_ms for env in candidates]
        )

        ranked = []

        for index, env in enumerate(candidates):
            score = (
                self.weights[0] * carbon_scores[index]
                + self.weights[1] * cost_scores[index]
                + self.weights[2] * latency_scores[index]
            )

            ranked.append((score, env))

        score, env = min(
            ranked,
            key=lambda item: item[0],
        )

        return self._decision(
            "multi_objective_predictive",
            env,
            score=score,
            explanation=(
                "Lowest weighted normalised score across predicted "
                "carbon, estimated cost and latency among "
                "SLA-eligible environments"
            ),
        )

    def schedule(self, policy, requirements=None):
        """Schedule a workload using the selected policy and constraints."""

        policies = {
            "round_robin": self.round_robin,
            "carbon_only": self.carbon_only,
            "cost_only": self.cost_only,
            "reactive_carbon_aware": self.reactive_carbon_aware,
            "multi_objective_predictive": (
                self.multi_objective_predictive
            ),
        }

        if policy not in policies:
            raise ValueError(f"Unknown policy: {policy}")

        decision = policies[policy](requirements)

        return asdict(decision)
