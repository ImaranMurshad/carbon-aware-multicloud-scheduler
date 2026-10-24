
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

        weights = [carbon_weight, cost_weight, latency_weight]

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative")

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1")

        self.environments = environments
        self.weights = weights
        self.round_robin_index = 0

    def _eligible(self):
        candidates = [
            env for env in self.environments if env.available
        ]

        if not candidates:
            raise RuntimeError("No eligible environments available")

        return candidates

    @staticmethod
    def _decision(policy, env, score=None, explanation=""):
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

    def round_robin(self):
        candidates = self._eligible()

        env = candidates[
            self.round_robin_index % len(candidates)
        ]
        self.round_robin_index += 1

        return self._decision(
            "round_robin",
            env,
            explanation="Selected by rotating through eligible environments",
        )

    def carbon_only(self):
        env = min(
            self._eligible(),
            key=lambda e: e.predicted_carbon,
        )

        return self._decision(
            "carbon_only",
            env,
            explanation="Lowest predicted carbon intensity",
        )

    def cost_only(self):
        env = min(
            self._eligible(),
            key=lambda e: e.cost_per_hour,
        )

        return self._decision(
            "cost_only",
            env,
            explanation="Lowest estimated hourly cost",
        )

    def reactive_carbon_aware(self):
        env = min(
            self._eligible(),
            key=lambda e: e.current_carbon,
        )

        return self._decision(
            "reactive_carbon_aware",
            env,
            explanation="Lowest currently observed carbon intensity",
        )

    def multi_objective_predictive(self):
        candidates = self._eligible()

        def normalise(values):
            low = min(values)
            high = max(values)

            if high == low:
                return [0.0] * len(values)

            return [(v - low) / (high - low) for v in values]

        carbon_scores = normalise(
            [e.predicted_carbon for e in candidates]
        )
        cost_scores = normalise(
            [e.cost_per_hour for e in candidates]
        )
        latency_scores = normalise(
            [e.latency_ms for e in candidates]
        )

        ranked = []

        for i, env in enumerate(candidates):
            score = (
                self.weights[0] * carbon_scores[i]
                + self.weights[1] * cost_scores[i]
                + self.weights[2] * latency_scores[i]
            )

            ranked.append((score, env))

        score, env = min(ranked, key=lambda item: item[0])

        return self._decision(
            "multi_objective_predictive",
            env,
            score=score,
            explanation=(
                "Lowest weighted normalised score across predicted "
                "carbon, estimated cost and latency"
            ),
        )

    def schedule(self, policy):
        policies = {
            "round_robin": self.round_robin,
            "carbon_only": self.carbon_only,
            "cost_only": self.cost_only,
            "reactive_carbon_aware": self.reactive_carbon_aware,
            "multi_objective_predictive": self.multi_objective_predictive,
        }

        if policy not in policies:
            raise ValueError(f"Unknown policy: {policy}")

        return asdict(policies[policy]())
