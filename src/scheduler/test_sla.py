
from src.scheduler.meta_scheduler import (
    Environment,
    MetaScheduler,
    WorkloadRequirements,
)


environments = [
    Environment(
        name="AWS",
        zone="IE",
        predicted_carbon=142,
        current_carbon=142,
        cost_per_hour=0.03,
        latency_ms=35,
        available_cpu_cores=4,
        available_memory_gb=8,
    ),
    Environment(
        name="Azure-local",
        zone="DE",
        predicted_carbon=89,
        current_carbon=89,
        cost_per_hour=0.02,
        latency_ms=60,
        available_cpu_cores=4,
        available_memory_gb=8,
    ),
    Environment(
        name="GCP-local",
        zone="FR",
        predicted_carbon=22,
        current_carbon=22,
        cost_per_hour=0.01,
        latency_ms=45,
        available_cpu_cores=2,
        available_memory_gb=4,
    ),
]

scheduler = MetaScheduler(environments)

# The workload requires latency <= 40 ms, at least 2 CPU cores,
# and at least 4 GB of available memory.
requirements = WorkloadRequirements(
    max_latency_ms=40,
    min_cpu_cores=2,
    min_memory_gb=4,
)

for policy in [
    "round_robin",
    "carbon_only",
    "cost_only",
    "reactive_carbon_aware",
    "multi_objective_predictive",
]:
    decision = scheduler.schedule(
        policy,
        requirements=requirements,
    )

    assert decision["estimated_latency_ms"] <= 40
    assert decision["selected_environment"] == "AWS"

    print(f"\nPolicy: {policy}")
    print(decision)

print("\nAll SLA tests passed!")




# Test that scheduling fails safely when no environment meets the SLA.
strict_requirements = WorkloadRequirements(
    max_latency_ms=20,
    min_cpu_cores=2,
    min_memory_gb=4,
)

try:
    scheduler.schedule(
        "multi_objective_predictive",
        requirements=strict_requirements,
    )
except RuntimeError as error:
    print("\nNo-eligible-environment test passed!")
    print(f"Expected error: {error}")
else:
    raise AssertionError(
        "Scheduler should reject workloads when no environment qualifies."
    )
