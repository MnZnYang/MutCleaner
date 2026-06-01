from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .base_config import BaseCleanerConfig
from .basic_cleaners import (
    apply_mutations_to_sequences,
    convert_to_mutation_dataset_format,
)
from .Chitosanase_custom_cleaners import parse_chitosanase_raw_file
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "ChitosanaseCleanerConfig",
    "create_chitosanase_cleaner",
    "clean_chitosanase_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class ChitosanaseCleanerConfig(BaseCleanerConfig):
    """Configuration for the Chitosanase cleaning pipeline.

    Holds dataset-specific defaults for the Chitosanase pipeline. The
    pipeline expects each raw input file to contain a CSV block followed by
    a WT sequence separated by ``wt_separator``.

    Attributes
    ----------
    infer_mut_workers : int
        Number of workers used when inferring/applying mutations (default 16).
    pipeline_name : str
        Human-readable pipeline name used in logs and artifacts.
    wt_separator : str
        Token that separates CSV block and WT sequence in raw files.
    """

    infer_mut_workers: int = 16
    pipeline_name: str = "Chitosanase"
    wt_separator: str = '">wt'

    def validate(self) -> None:
        super().validate()


def create_chitosanase_cleaner(
    dataset_or_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[ChitosanaseCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    """Create a configured Pipeline for cleaning Chitosanase raw files.

    Parameters
    ----------
    dataset_or_path : Optional[Union[str, Path]]
        Path to a raw Chitosanase input file (or a DataFrame for programmatic
        callers). Raw files must contain a CSV block followed by the WT
        sequence separated by the configured ``wt_separator``.
    config : Optional[Union[ChitosanaseCleanerConfig, Dict[str, Any], str, Path]]
        Pipeline configuration. Accepts a `ChitosanaseCleanerConfig` instance,
        a dict of overrides merged with defaults, or a path to a JSON
        configuration file.

    Returns
    -------
    Pipeline
        A :class:`Pipeline` instance ready for execution via
        ``pipeline.execute()``.

    Examples
    --------
    >>> pipeline = create_chitosanase_cleaner("/path/to/Chitosanase_Dataset.csv")
    >>> pipeline.execute()
    """
    # Handle config
    if config is None:
        final_config = ChitosanaseCleanerConfig()
    elif isinstance(config, ChitosanaseCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        final_config = ChitosanaseCleanerConfig().merge(config)
    elif isinstance(config, (str, Path)):
        final_config = ChitosanaseCleanerConfig.from_json(config)
    else:
        raise TypeError(f"config has invalid type: {type(config)}")

    if dataset_or_path is None:
        raise TypeError("dataset_or_path must be a Chitosanase file path")

    try:
        pipeline = create_pipeline(dataset_or_path, final_config.pipeline_name)

        # Add cleaning steps
        pipeline = (
            pipeline.delayed_then(parse_chitosanase_raw_file, wt_separator=final_config.wt_separator)
            .delayed_then(
                apply_mutations_to_sequences,
                sequence_column="sequence",
                mutation_column="mut_info",
                sequence_type="protein",
                is_zero_based=False,
                num_workers=final_config.infer_mut_workers,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="name",
                mutation_column="mut_info",
                sequence_column="sequence",
                mutated_sequence_column="mut_seq",
                label_column="dTm",
                is_zero_based=False,
            )
        )
        return pipeline
    except Exception as e:
        logger.error(f"Error in creating Chitosanase cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating Chitosanase cleaning pipeline: {str(e)}")


def clean_chitosanase_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    """Run the Chitosanase pipeline and return the formatted dataset.

    Executes the provided :class:`Pipeline`, converts the pipeline output
    into a :class:`MutationDataset`, and returns both the executed pipeline
    and the dataset. Pipeline artifacts and diagnostics may be saved with
    ``pipeline.save_artifacts(path)`` by the caller.

    Parameters
    ----------
    pipeline : Pipeline
        The configured Chitosanase cleaning pipeline to execute.

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        - The executed :class:`Pipeline`.
        - The resulting :class:`MutationDataset` built from the formatted
          DataFrame emitted by the pipeline.

    Examples
    --------
    >>> pipeline = create_chitosanase_cleaner("/path/to/file.csv")
    >>> pipeline, dataset = clean_chitosanase_dataset(pipeline)
    """
    try:
        pipeline.execute()

        formatted_df, ref_dict = pipeline.data
        chitosanase_dataset = MutationDataset.from_dataframe(formatted_df, reference_sequences=ref_dict)

        logger.info(f"Successfully cleaned Chitosanase dataset: " f"{len(formatted_df)} mutations from {len(ref_dict)} proteins")
        return pipeline, chitosanase_dataset
    except Exception as e:
        logger.error(f"Error in running Chitosanase cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running Chitosanase cleaning pipeline: {str(e)}")
