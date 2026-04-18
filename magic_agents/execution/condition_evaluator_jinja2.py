"""
Jinja2Evaluator — Default ConditionEvaluator implementation.

Wraps Jinja2 template evaluation to conform to the ConditionEvaluator protocol.
Preserves backward compatibility with existing NodeConditional templates.
"""

from __future__ import annotations

from typing import Any, Dict

import jinja2

from magic_agents.execution.condition_evaluator import ConditionEvaluator


class Jinja2Evaluator:
    """
    Jinja2-based condition evaluator.
    
    This is the default evaluator for NodeConditional, preserving
    backward compatibility with existing Jinja2 templates.
    
    Note: Uses Jinja2's default Undefined behavior (renders to empty string).
    To get strict undefined variable errors, use Jinja2StrictEvaluator instead.
    """
    
    def evaluate(self, template: str, context: Dict[str, Any]) -> str:
        """
        Evaluate a Jinja2 template with the given context.
        
        Args:
            template: Jinja2 template string
            context: Variables available for evaluation
            
        Returns:
            Rendered template as string
        """
        env = jinja2.Environment()
        tpl = env.from_string(template)
        return str(tpl.render(**context)).strip()
    
    def validate_syntax(self, template: str) -> bool:
        """
        Validate Jinja2 template syntax.
        
        Args:
            template: Jinja2 template string
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            jinja2.Environment().parse(template)
            return True
        except jinja2.TemplateSyntaxError:
            return False


class Jinja2StrictEvaluator:
    """
    Strict Jinja2 evaluator that raises UndefinedError for undefined variables.
    
    Use this when you want undefined variables to be errors rather than
    silently rendering to empty strings.
    """
    
    def evaluate(self, template: str, context: Dict[str, Any]) -> str:
        """
        Evaluate a Jinja2 template with strict undefined handling.
        
        Raises UndefinedError if any variable in the template is not
        defined in the context.
        """
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        tpl = env.from_string(template)
        return str(tpl.render(**context)).strip()
    
    def validate_syntax(self, template: str) -> bool:
        """Validate Jinja2 template syntax."""
        try:
            jinja2.Environment().parse(template)
            return True
        except jinja2.TemplateSyntaxError:
            return False
