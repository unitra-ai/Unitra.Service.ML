"""Settings and configuration for the MT service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MTSettings:
    """Machine Translation service settings."""

    # Modal app configuration
    app_name: str = "unitra-mt"
    volume_name: str = "unitra-models"
    model_path: str = "/models"

    # Model configuration
    model_id: str = "google/madlad400-3b-mt"
    model_dir: str = "madlad-400-3b-mt"

    # GPU configuration
    gpu_type: str = "a10g"  # A10G for cost/performance balance
    container_idle_timeout: int = 300  # 5 minutes
    concurrent_inputs: int = 16
    retries: int = 2

    # Inference configuration
    max_input_length: int = 512
    max_output_length: int = 256
    max_batch_size: int = 16
    num_beams: int = 4
    early_stopping: bool = True

    # Performance targets
    target_latency_ms: int = 500  # Target warm latency
    target_cold_start_s: int = 30  # Target cold start

    @property
    def full_model_path(self) -> str:
        """Get full path to model directory."""
        return f"{self.model_path}/{self.model_dir}"


# Global settings instance
settings = MTSettings()
