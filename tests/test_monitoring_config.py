from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class MonitoringConfigTest(unittest.TestCase):
    def test_prometheus_scrapes_required_targets(self) -> None:
        config = yaml.safe_load((ROOT / "monitoring/prometheus/prometheus.yml").read_text())
        jobs = {item["job_name"] for item in config["scrape_configs"]}
        self.assertTrue(
            {"reid-api", "triton", "qdrant", "node", "containers", "gpu", "alertmanager"} <= jobs
        )
        self.assertTrue(config["rule_files"])

    def test_required_alerts_are_present(self) -> None:
        rules = yaml.safe_load((ROOT / "monitoring/prometheus/rules/reid-alerts.yml").read_text())
        names = {
            rule["alert"]
            for group in rules["groups"]
            for rule in group["rules"]
            if "alert" in rule
        }
        self.assertTrue(
            {
                "ReIDMonitoringTargetDown",
                "ReIDServingComponentUnhealthy",
                "ReIDAPIHighErrorRate",
                "ReIDAPIHighLatency",
            }
            <= names
        )

    def test_grafana_dashboards_are_valid_and_unique(self) -> None:
        dashboards = [
            json.loads(path.read_text())
            for path in sorted((ROOT / "monitoring/grafana/dashboards").glob("*.json"))
        ]
        self.assertEqual(len(dashboards), 2)
        self.assertEqual(len({dashboard["uid"] for dashboard in dashboards}), 2)
        self.assertTrue(all(dashboard["panels"] for dashboard in dashboards))


if __name__ == "__main__":
    unittest.main()
