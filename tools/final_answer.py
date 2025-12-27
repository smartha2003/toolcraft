from typing import Any, Optional
from smolagents.tools import Tool
from smolagents.agent_types import AgentImage
import os
import tempfile

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        print(f"[DEBUG FinalAnswerTool] Received answer type: {type(answer)}")
        print(f"[DEBUG FinalAnswerTool] Received answer: {answer}")
        
        # Check if answer is a PIL Image and convert it to AgentImage
        try:
            from PIL import Image
            is_pil_image = isinstance(answer, Image.Image) or (hasattr(answer, '__class__') and 'PIL' in str(answer.__class__) and 'Image' in str(answer.__class__))
            print(f"[DEBUG FinalAnswerTool] Is PIL Image? {is_pil_image}")
            
            if is_pil_image:
                print(f"[DEBUG FinalAnswerTool] ✅ Detected PIL Image! Size: {answer.size}, Mode: {answer.mode}")
                # Save PIL Image to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                print(f"[DEBUG FinalAnswerTool] Saving to temp file: {temp_file.name}")
                answer.save(temp_file.name, 'PNG')
                temp_file.close()
                # Verify file exists and has content
                file_exists = os.path.exists(temp_file.name)
                file_size = os.path.getsize(temp_file.name) if file_exists else 0
                print(f"[DEBUG FinalAnswerTool] File exists: {file_exists}, Size: {file_size} bytes")
                
                if file_exists and file_size > 0:
                    # Convert to AgentImage
                    agent_image = AgentImage(temp_file.name)
                    print(f"[DEBUG FinalAnswerTool] ✅ Created AgentImage with path: {agent_image.to_string()}")
                    return agent_image
                else:
                    print(f"[DEBUG FinalAnswerTool] ❌ File validation failed!")
        except ImportError:
            print(f"[DEBUG FinalAnswerTool] ❌ PIL not available")
        except Exception as e:
            # If conversion fails, return original answer
            import sys
            import traceback
            print(f"[DEBUG FinalAnswerTool] ❌ Error converting PIL Image: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        # Return answer as-is if not a PIL Image or conversion failed
        print(f"[DEBUG FinalAnswerTool] Returning answer as-is: {type(answer)}")
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
