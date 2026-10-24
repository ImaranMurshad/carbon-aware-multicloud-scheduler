
from src.scheduler.meta_scheduler import Environment, MetaScheduler


def main():
    environments = [
        Environment(
            name="AWS",
            zone="IE",
            predicted_carbon=142,
            current_carbon=142,
            cost_per_hour=0.03,
            latency_ms=35,
        ),
        Environment(
            name="Azure-local",
            zone="DE",
            predicted_carbon=89,
            current_carbon=89,
            cost_per_hour=0.02,
            latency_ms=60,
        ),
        Environment(
            name="GCP-local",
            zone="FR",
            predicted_carbon=22,
            current_carbon=22,
            cost_per_hour=0.01,
            latency_ms=45,
        ),
    ]

    scheduler = MetaScheduler(environments)

    policies = [
        "round_robin",
        "carbon_only",
        "cost_only",
        "reactive_carbon_aware",
        "multi_objective_predictive",
    ]

    for policy in policies:
        print(f"\nPolicy: {policy}")
        print(scheduler.schedule(policy))


if __name__ == "__main__":
    main()
