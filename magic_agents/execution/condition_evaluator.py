"""
ConditionEvaluator Protocol — Pluggable condition evaluation strategy.

This protocol allows NodeConditional to use any evaluation engine
(Jinja2, Python expressions, JSONPath, etc.) instead of being hardcoded
to Jinja2.

The default implementation is Jinja2Evaluator which preserves backward
compatibility with existing templates.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol


class ConditionEvaluator(Protocol):
    """
    Protocol for condition evaluation strategies.
    
    Implementations must provide:
    - evaluate(): Render a template/expression with the given context
    - validate_syntax(): Check if a template/expression is syntactically valid
    
    The default implementation is Jinja2Evaluator.
    """
    
    def evaluate(self, template: str, context: Dict[str, Any]) -> str:
        """
        Evaluate a condition template with the given context.
        
        Args:
            template: The condition template/expression string
            context: Variables available for evaluation
            
        Returns:
            The evaluated result as a string (typically a handle name)
            
        Raises:
            Should raise appropriate exceptions for evaluation errors.
            The caller (NodeConditional.process()) handles these and
            converts them to debug_error + BYPASS_ALL signals.
        """
        ...
    
    def validate_syntax(self, template: str) -> bool:
        """
        Validate the syntax of a condition template.
        
        Args:
            template: The condition template/expression string
            
        Returns:
            True if syntax is valid, False otherwise
        """
        ...
