from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

class ParameterDef(BaseModel):
    """Defines the type of a single parameter."""
    type: str

class FunctionDef(BaseModel):
    """Validates the structure of a function from functions_definition.json."""
    model_config = ConfigDict(extra='ignore')
    
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: Optional[Dict[str, str]] = None

class PromptInput(BaseModel):
    """Validates the input from function_calling_tests.json."""
    prompt: str

class FunctionCallOutput(BaseModel):
    """Validates the exact output format required by the subject."""
    prompt: str
    name: str
    parameters: Dict[str, Any]