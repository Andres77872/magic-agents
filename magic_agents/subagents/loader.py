"""Manifest loader for YAML files.

Loads *.agent.yaml files, validates with Pydantic, detects duplicates.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .models import SubagentManifest
from .errors import DuplicateSubagentError

logger = logging.getLogger(__name__)


class ManifestLoader:
    """Load and validate YAML manifest files.
    
    Scans directory for *.agent.yaml files.
    Validates each with Pydantic SubagentManifest model.
    Detects duplicate IDs (hard error).
    
    Example YAML:
        apiVersion: magic-agents/v1
        kind: TaskSubagent
        id: research.web
        name: Web Research
        description: Search and summarize web content
        version: 1.0.0
        input_schema:
          type: object
          properties:
            query: {type: string}
          required: [query]
        timeout_seconds: 30
        max_concurrency: 5
        max_depth: 3
    """
    
    def __init__(self, manifest_dir: Path):
        """Initialize loader with directory path.
        
        Args:
            manifest_dir: Directory containing *.agent.yaml files
        """
        self.manifest_dir = manifest_dir
        self._loaded_ids: dict[str, Path] = {}  # Track IDs for duplicate detection
    
    async def load_all(self) -> list[SubagentManifest]:
        """Load all manifest files from directory.
        
        Returns:
            List of validated SubagentManifest instances
            
        Raises:
            DuplicateSubagentError: If duplicate IDs found
        """
        manifests = []
        
        if not self.manifest_dir.exists():
            logger.debug(
                "Manifest directory %s does not exist — no subagents loaded",
                self.manifest_dir
            )
            return manifests
        
        yaml_files = list(self.manifest_dir.glob("*.agent.yaml"))
        
        if not yaml_files:
            logger.debug(
                "No *.agent.yaml files found in %s",
                self.manifest_dir
            )
            return manifests
        
        logger.info(
            "Loading %d manifest files from %s",
            len(yaml_files),
            self.manifest_dir
        )
        
        for yaml_file in yaml_files:
            try:
                manifest = await self._load_file(yaml_file)
                
                # Duplicate detection
                if manifest.id in self._loaded_ids:
                    raise DuplicateSubagentError(
                        agent_id=manifest.id,
                        existing_source=str(self._loaded_ids[manifest.id]),
                        new_source=str(yaml_file)
                    )
                
                self._loaded_ids[manifest.id] = yaml_file
                manifests.append(manifest)
                
            except DuplicateSubagentError:
                # Re-raise - this is a hard error
                raise
            except Exception as e:
                logger.error(
                    "Failed to load manifest %s: %s",
                    yaml_file,
                    e
                )
                # Skip invalid file, don't crash
                continue
        
        logger.info(
            "Successfully loaded %d subagent manifests",
            len(manifests)
        )
        
        return manifests
    
    async def _load_file(self, yaml_file: Path) -> SubagentManifest:
        """Load and validate a single YAML file.
        
        Args:
            yaml_file: Path to YAML manifest
            
        Returns:
            Validated SubagentManifest
            
        Raises:
            ValidationError: If Pydantic validation fails
        """
        content = yaml_file.read_text()
        data = yaml.safe_load(content)
        
        if data is None:
            raise ValueError(f"Empty YAML file: {yaml_file}")
        
        # Validate with Pydantic
        manifest = SubagentManifest(
            **data,
            source_file=yaml_file  # Track source for error messages
        )
        
        logger.debug(
            "Loaded manifest '%s' v%s from %s",
            manifest.id,
            manifest.version,
            yaml_file
        )
        
        return manifest
    
    def get_loaded_ids(self) -> dict[str, Path]:
        """Get map of loaded IDs to source files.
        
        Returns:
            Dict of agent_id -> source file path
        """
        return self._loaded_ids.copy()