from typing import List, Optional
from src.models import FunctionDef

class JSONStateMachine:
    STATE_PREFIX = 0
    STATE_FUNC_NAME = 1
    STATE_PARAM_START = 2
    STATE_PARAM_KEY = 3
    STATE_PARAM_VALUE = 4
    STATE_PARAM_NEXT = 5
    STATE_DONE = 6

    def __init__(self, functions: List[FunctionDef]):
        self.functions = functions
        self.func_map = {f.name: f for f in functions}
        self.current_state = self.STATE_PREFIX
        self.current_buffer = ""
        self.selected_func: Optional[FunctionDef] = None
        self.remaining_params: List[str] = []
        self.current_param_key: Optional[str] = None

    def get_allowed_strings(self) -> List[str]:
        if self.current_state == self.STATE_PREFIX:
            return ['{"name":"']
        elif self.current_state == self.STATE_FUNC_NAME:
            return list(self.func_map.keys())
        elif self.current_state == self.STATE_PARAM_START:
            return ['","parameters":{']
        elif self.current_state == self.STATE_PARAM_KEY:
            if self.remaining_params:
                param_name = self.remaining_params[0]
                param_type = self.selected_func.parameters[param_name].type
                # FORCING THE OPENING QUOTE: If it's a string, we force '"s":"'
                if param_type == "string":
                    return ['"' + param_name + '":"']
                else:
                    return ['"' + param_name + '":']
            return []
        elif self.current_state == self.STATE_PARAM_VALUE:
            param_type = self.selected_func.parameters[self.current_param_key].type
            return [f"__VALUE_{param_type.upper()}__"]
        elif self.current_state == self.STATE_PARAM_NEXT:
            param_type = self.selected_func.parameters[self.current_param_key].type
            # FORCING THE CLOSING QUOTE: Add the closing quote before the comma/brackets
            suffix = '"' if param_type == "string" else ""
            if self.remaining_params:
                return [suffix + ',']
            else:
                return [suffix + '}}']
        return []

    def update_state(self, printed_chunk: str):
        if printed_chunk == "__DONE__":
            self.current_state = self.STATE_PARAM_NEXT
            self.current_buffer = ""
            return

        self.current_buffer += printed_chunk

        if self.current_state == self.STATE_PREFIX:
            if self.current_buffer == '{"name":"':
                self.current_state = self.STATE_FUNC_NAME
                self.current_buffer = ""
        elif self.current_state == self.STATE_FUNC_NAME:
            if self.current_buffer in self.func_map:
                self.selected_func = self.func_map[self.current_buffer]
                self.remaining_params = list(self.selected_func.parameters.keys())
                self.current_state = self.STATE_PARAM_START
                self.current_buffer = ""
        elif self.current_state == self.STATE_PARAM_START:
            if self.current_buffer == '","parameters":{':
                if not self.remaining_params:
                    self.current_state = self.STATE_PARAM_NEXT
                else:
                    self.current_state = self.STATE_PARAM_KEY
                self.current_buffer = ""
        elif self.current_state == self.STATE_PARAM_KEY:
            param_name = self.remaining_params[0]
            param_type = self.selected_func.parameters[param_name].type
            target_key = '"' + param_name + '":"' if param_type == "string" else '"' + param_name + '":'

            if self.current_buffer == target_key:
                self.current_param_key = self.remaining_params.pop(0)
                self.current_state = self.STATE_PARAM_VALUE
                self.current_buffer = ""
        elif self.current_state == self.STATE_PARAM_NEXT:
            param_type = self.selected_func.parameters[self.current_param_key].type
            suffix = '"' if param_type == "string" else ""

            if self.current_buffer == suffix + ',':
                self.current_state = self.STATE_PARAM_KEY
                self.current_buffer = ""
            elif self.current_buffer == suffix + '}}':
                self.current_state = self.STATE_DONE
                self.current_buffer = ""