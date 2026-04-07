SCENARIOS = {
    "easy": {
        "alerts": [
            {"type": "timeout", "service": "database"},
            {"type": "high_latency", "service": "api"},
        ],
        "logs": ["DB connection timeout", "retry failed"],
        "initial_visible": ["api"],
        "dependencies": {
            "api": ["database"],
            "database": [],
            "cache": ["database"],
        },
        "service_logs": {
            "database": ["ERROR: max connections reached"],
        },
        "noise_alerts": [],
        "root_cause": "database",
    },

    "medium": {
        "alerts": [
            {"type": "high_latency", "service": "api"},
            {"type": "cache_miss", "service": "cache"},
            {"type": "timeout", "service": "database"},
            {"type": "cpu_spike", "service": "frontend"},  # noise
        ],
        "logs": ["cache miss rate high", "api latency 2000ms", "DB slow query"],
        "initial_visible": ["api", "cache"],
        "dependencies": {
            "frontend": ["api"],
            "api": ["database", "cache"],
            "database": [],
            "cache": ["database"],
        },
        "service_logs": {
            "database": ["slow query log: 5s", "lock timeout"],
        },
        "noise_alerts": [{"type": "cpu_spike", "service": "frontend"}],
        "root_cause": "database",
    },

    "hard": {
        "alerts": [
            {"type": "high_latency", "service": "api"},
            {"type": "error_rate", "service": "frontend"},
            {"type": "memory_spike", "service": "api"},  # misleading
        ],
        "logs": ["api memory growing", "frontend 500 errors", "config load failed"],
        "initial_visible": ["frontend"],
        "dependencies": {
            "frontend": ["api"],
            "api": ["config_service", "database"],
            "config_service": [],
            "database": [],
        },
        "service_logs": {
            "api": ["config_service unreachable", "using stale config"],
            "config_service": ["ERROR: deployment failed", "version mismatch"],
        },
        "noise_alerts": [{"type": "memory_spike", "service": "api"}],
        "root_cause": "config_service",
    },
}