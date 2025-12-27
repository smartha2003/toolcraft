#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from typing import Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    
                    # Check if the observation contains an image path (from image_generator tool)
                    # Look for common image file extensions or AgentImage objects
                    image_paths = re.findall(r'([^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))', log_content, re.IGNORECASE)
                    if image_paths and os.path.exists(image_paths[0]):
                        # Display the image
                        yield gr.ChatMessage(
                            role="assistant",
                            content={"path": image_paths[0], "mime_type": "image/png"},
                            metadata={"title": "🖼️ Generated Image", "parent_id": parent_id, "status": "done"},
                        )
                        # Remove image path from text content
                        log_content = re.sub(r'[^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp)', '', log_content, flags=re.IGNORECASE).strip()
                    
                    # Also check if tool outputs contain AgentImage objects
                    if hasattr(step_log, "tool_calls") and step_log.tool_calls:
                        for tool_call in step_log.tool_calls:
                            if hasattr(tool_call, "output") and isinstance(tool_call.output, AgentImage):
                                yield gr.ChatMessage(
                                    role="assistant",
                                    content={"path": tool_call.output.to_string(), "mime_type": "image/png"},
                                    metadata={"title": "🖼️ Generated Image", "parent_id": parent_id, "status": "done"},
                                )
                    
                    if log_content:  # Only yield text if there's remaining content
                        yield gr.ChatMessage(
                            role="assistant",
                            content=f"{log_content}",
                            metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                        )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0
    pil_image_from_generator = None  # Store PIL Image from image_generator tool

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            input_tokens = agent.model.last_input_token_count or 0
            output_tokens = agent.model.last_output_token_count or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = input_tokens
                step_log.output_token_count = output_tokens

        # Check if this step contains an image_generator tool call and extract the PIL Image
        if isinstance(step_log, ActionStep) and hasattr(step_log, "tool_calls") and step_log.tool_calls:
            print(f"[DEBUG Step 5] Checking {len(step_log.tool_calls)} tool calls in ActionStep")
            for tool_call in step_log.tool_calls:
                print(f"[DEBUG Step 5] Tool call name: {tool_call.name}")
                # Check if python_interpreter was used (code execution)
                if tool_call.name == "python_interpreter":
                    print(f"[DEBUG Step 5] Found python_interpreter - checking for image_generator calls")
                    # Check observations for PIL Image output
                    if hasattr(step_log, "observations"):
                        obs_text = str(step_log.observations)
                        if 'image_generator' in obs_text.lower() or 'PIL' in obs_text:
                            print(f"[DEBUG Step 5] Found image_generator or PIL reference in observations")
                            # Try to get the actual PIL Image from the execution context
                            # The image might be in tool_call.output if it was returned
                            if hasattr(tool_call, "output"):
                                print(f"[DEBUG Step 5] python_interpreter.output type: {type(tool_call.output)}")
                                print(f"[DEBUG Step 5] python_interpreter.output: {tool_call.output}")
                                try:
                                    from PIL import Image
                                    if isinstance(tool_call.output, Image.Image) or (hasattr(tool_call.output, '__class__') and 'PIL' in str(tool_call.output.__class__) and 'Image' in str(tool_call.output.__class__)):
                                        pil_image_from_generator = tool_call.output
                                        print(f"[DEBUG Step 5] ✅ Captured PIL Image from python_interpreter output!")
                                except:
                                    pass
                elif tool_call.name == "image_generator":
                    print(f"[DEBUG Step 5] ✅ Found image_generator tool call")
                    if hasattr(tool_call, "output"):
                        print(f"[DEBUG Step 5] tool_call.output type: {type(tool_call.output)}")
                        print(f"[DEBUG Step 5] tool_call.output: {tool_call.output}")
                        try:
                            from PIL import Image
                            if isinstance(tool_call.output, Image.Image) or (hasattr(tool_call.output, '__class__') and 'PIL' in str(tool_call.output.__class__) and 'Image' in str(tool_call.output.__class__)):
                                pil_image_from_generator = tool_call.output
                                print(f"[DEBUG Step 5] ✅ Captured PIL Image! Size: {pil_image_from_generator.size}, Mode: {pil_image_from_generator.mode}")
                            else:
                                print(f"[DEBUG Step 5] ❌ tool_call.output is not a PIL Image")
                        except Exception as e:
                            print(f"[DEBUG Step 5] ❌ Error checking PIL Image: {e}")
                    else:
                        print(f"[DEBUG Step 5] ❌ tool_call has no 'output' attribute")
                elif tool_call.name == "final_answer":
                    print(f"[DEBUG Step 6] ✅ Found final_answer tool call")
                    if hasattr(tool_call, "output"):
                        print(f"[DEBUG Step 6] final_answer.output type: {type(tool_call.output)}")
                        print(f"[DEBUG Step 6] final_answer.output: {tool_call.output}")
                    if hasattr(tool_call, "arguments"):
                        print(f"[DEBUG Step 6] final_answer.arguments type: {type(tool_call.arguments)}")
                        print(f"[DEBUG Step 6] final_answer.arguments: {tool_call.arguments}")
                        # Try to extract PIL Image from arguments if it's a dict
                        if isinstance(tool_call.arguments, dict) and "answer" in tool_call.arguments:
                            answer_arg = tool_call.arguments["answer"]
                            print(f"[DEBUG Step 6] answer argument type: {type(answer_arg)}")
                            print(f"[DEBUG Step 6] answer argument: {answer_arg}")
                            try:
                                from PIL import Image
                                if isinstance(answer_arg, Image.Image) or (hasattr(answer_arg, '__class__') and 'PIL' in str(answer_arg.__class__) and 'Image' in str(answer_arg.__class__)):
                                    pil_image_from_generator = answer_arg
                                    print(f"[DEBUG Step 6] ✅ Captured PIL Image from final_answer arguments! Size: {pil_image_from_generator.size}")
                            except:
                                pass
        
        # Also check observations for PIL Image references
        if isinstance(step_log, ActionStep) and hasattr(step_log, "observations"):
            obs_text = str(step_log.observations)
            if 'PIL' in obs_text and 'Image' in obs_text and 'size=' in obs_text:
                print(f"[DEBUG Step 5] Found PIL Image reference in observations: {obs_text[:200]}...")

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    print(f"[DEBUG Step 8] Starting final answer extraction")
    print(f"[DEBUG Step 8] step_log type: {type(step_log)}")
    print(f"[DEBUG Step 8] pil_image_from_generator: {pil_image_from_generator}")
    if pil_image_from_generator:
        print(f"[DEBUG Step 8] pil_image_from_generator size: {pil_image_from_generator.size}")
    
    # Check if step_log is a FinalAnswerStep and extract the actual answer
    if hasattr(step_log, 'final_answer'):
        print(f"[DEBUG Step 8] ✅ Found FinalAnswerStep with final_answer attribute")
        print(f"[DEBUG Step 8] step_log.final_answer type: {type(step_log.final_answer)}")
        print(f"[DEBUG Step 8] step_log.final_answer: {step_log.final_answer}")
        # Extract the actual answer value
        final_answer_value = step_log.final_answer
        
        # If final_answer_value is an empty AgentImage, we need to use the PIL Image we captured
        if isinstance(final_answer_value, AgentImage):
            try:
                img_path = final_answer_value.to_string()
                if not img_path or not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                    print(f"[DEBUG Step 8] ⚠️ AgentImage is empty! Using pil_image_from_generator if available")
                    if pil_image_from_generator is not None:
                        print(f"[DEBUG Step 8] ✅ Found PIL Image from generator, will convert")
                        final_answer_value = pil_image_from_generator
            except:
                pass
    else:
        final_answer_value = step_log
    
    # First, try to find PIL Image from image_generator tool call in this step or previous steps
    pil_image_found = None
    if isinstance(final_answer, ActionStep):
        # Check tool calls for image_generator output
        if hasattr(final_answer, "tool_calls") and final_answer.tool_calls:
            for tool_call in final_answer.tool_calls:
                # Look for image_generator tool call
                if tool_call.name == "image_generator":
                    if hasattr(tool_call, "output"):
                        try:
                            from PIL import Image
                            if isinstance(tool_call.output, Image.Image) or (hasattr(tool_call.output, '__class__') and 'PIL' in str(tool_call.output.__class__) and 'Image' in str(tool_call.output.__class__)):
                                pil_image_found = tool_call.output
                                break
                        except:
                            pass
        
        # Also check observations for PIL Image string representation
        if pil_image_found is None and hasattr(final_answer, "observations"):
            try:
                from PIL import Image
                import re
                # Look for PIL Image in observations text
                obs_text = str(final_answer.observations)
                # Try to extract PIL Image object from the execution context
                # The image might be stored in a variable that was printed
                if 'PIL' in obs_text and 'Image' in obs_text and 'size=' in obs_text:
                    # Try to find the actual image object from the code execution
                    # This is a fallback - we'll try to get it from the final_answer tool call
                    pass
            except:
                pass
    
    # Extract actual final answer value from step_log if it's an ActionStep
    if isinstance(final_answer, ActionStep):
        # Check if there's a final_answer tool call
        if hasattr(final_answer, "tool_calls") and final_answer.tool_calls:
            for tool_call in final_answer.tool_calls:
                if tool_call.name == "final_answer":
                    # Get the actual answer value from the tool call
                    if hasattr(tool_call, "output"):
                        final_answer_value = tool_call.output
                    elif hasattr(tool_call, "arguments"):
                        # Extract from arguments if output not available
                        args = tool_call.arguments
                        if isinstance(args, dict) and "answer" in args:
                            final_answer_value = args["answer"]
                        elif isinstance(args, str):
                            final_answer_value = args
                        else:
                            final_answer_value = None
                    else:
                        final_answer_value = None
                    
                    # If final_answer_value is a PIL Image, use it
                    # Otherwise, if we found a PIL Image from image_generator, use that
                    try:
                        from PIL import Image
                        if isinstance(final_answer_value, Image.Image) or (hasattr(final_answer_value, '__class__') and 'PIL' in str(final_answer_value.__class__) and 'Image' in str(final_answer_value.__class__)):
                            pil_image_found = final_answer_value
                        elif pil_image_found is None:
                            # Try to get PIL Image from the variable that was passed to final_answer
                            # This might be stored in the execution context
                            pass
                    except:
                        pass
                    
                    if pil_image_found is None:
                        final_answer = final_answer_value if final_answer_value is not None else final_answer
                    break
    
    # Use the PIL Image we found from image_generator, or from final_answer, or try to convert the final_answer
    # Priority: pil_image_from_generator > pil_image_found > final_answer_value > final_answer
    pil_image_to_convert = None
    if pil_image_from_generator is not None:
        pil_image_to_convert = pil_image_from_generator
        print(f"[DEBUG Step 9] Using pil_image_from_generator")
    elif pil_image_found is not None:
        pil_image_to_convert = pil_image_found
        print(f"[DEBUG Step 9] Using pil_image_found")
    elif 'final_answer_value' in locals():
        pil_image_to_convert = final_answer_value
        print(f"[DEBUG Step 9] Using final_answer_value")
    else:
        pil_image_to_convert = final_answer
        print(f"[DEBUG Step 9] Using final_answer")
    
    print(f"[DEBUG Step 9] pil_image_to_convert type: {type(pil_image_to_convert)}")
    print(f"[DEBUG Step 9] pil_image_to_convert: {pil_image_to_convert}")
    
    # Check if final_answer is a PIL Image object and convert it BEFORE handle_agent_output_types
    # This is critical because handle_agent_output_types might not handle PIL Images correctly
    pil_image_converted = False
    try:
        from PIL import Image
        # Check if it's a PIL Image (including WebPImageFile)
        is_pil_image = isinstance(pil_image_to_convert, Image.Image) or (hasattr(pil_image_to_convert, '__class__') and 'PIL' in str(pil_image_to_convert.__class__) and 'Image' in str(pil_image_to_convert.__class__))
        print(f"[DEBUG Step 9] Is PIL Image? {is_pil_image}")
        if is_pil_image:
            print(f"[DEBUG Step 9] ✅ Found PIL Image! Size: {pil_image_to_convert.size}, Mode: {pil_image_to_convert.mode}")
            # Save PIL Image to temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            print(f"[DEBUG Step 9] Saving to temp file: {temp_file.name}")
            pil_image_to_convert.save(temp_file.name, 'PNG')
            temp_file.close()
            # Verify file exists and has content
            file_exists = os.path.exists(temp_file.name)
            file_size = os.path.getsize(temp_file.name) if file_exists else 0
            print(f"[DEBUG Step 9] File exists: {file_exists}, Size: {file_size} bytes")
            if file_exists and file_size > 0:
                # Convert to AgentImage
                final_answer = AgentImage(temp_file.name)
                print(f"[DEBUG Step 9] ✅ Created AgentImage with path: {final_answer.to_string()}")
                pil_image_converted = True
            else:
                print(f"[DEBUG Step 9] ❌ File validation failed!")
        else:
            print(f"[DEBUG Step 9] ❌ Not a PIL Image, cannot convert")
    except ImportError:
        print(f"[DEBUG Step 9] ❌ PIL not available")
    except Exception as e:
        # Log the error for debugging but continue
        import sys
        print(f"[DEBUG Step 9] ❌ Error converting PIL Image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    # Only apply handle_agent_output_types if we haven't already converted PIL Image
    # Otherwise it might create an empty AgentImage
    if not pil_image_converted:
        print(f"[DEBUG Step 9] Applying handle_agent_output_types...")
        print(f"[DEBUG Step 9] final_answer before handle_agent_output_types: {type(final_answer)} - {final_answer}")
        final_answer = handle_agent_output_types(final_answer)
        print(f"[DEBUG Step 9] final_answer after handle_agent_output_types: {type(final_answer)} - {final_answer}")
        
        # If handle_agent_output_types created an empty AgentImage, try to fix it
        if isinstance(final_answer, AgentImage):
            try:
                # Check if the AgentImage has a valid file path
                img_path = final_answer.to_string()
                print(f"[DEBUG Step 9] AgentImage path: {img_path}")
                print(f"[DEBUG Step 9] Path exists: {os.path.exists(img_path) if img_path else False}")
                if img_path:
                    print(f"[DEBUG Step 9] Path size: {os.path.getsize(img_path) if os.path.exists(img_path) else 0} bytes")
                if not os.path.exists(img_path) or (os.path.exists(img_path) and os.path.getsize(img_path) == 0):
                    print(f"[DEBUG Step 9] ❌ AgentImage is empty! Trying to fix...")
                    # AgentImage is empty, try to find PIL Image again
                    if pil_image_from_generator is not None:
                        print(f"[DEBUG Step 9] Using pil_image_from_generator to fix...")
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        pil_image_from_generator.save(temp_file.name, 'PNG')
                        temp_file.close()
                        if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                            final_answer = AgentImage(temp_file.name)
                            print(f"[DEBUG Step 9] ✅ Fixed AgentImage with path: {final_answer.to_string()}")
                    elif pil_image_found is not None:
                        print(f"[DEBUG Step 9] Using pil_image_found to fix...")
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        pil_image_found.save(temp_file.name, 'PNG')
                        temp_file.close()
                        if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                            final_answer = AgentImage(temp_file.name)
                            print(f"[DEBUG Step 9] ✅ Fixed AgentImage with path: {final_answer.to_string()}")
            except Exception as e:
                print(f"[DEBUG Step 9] ❌ Error fixing AgentImage: {e}")
                import traceback
                traceback.print_exc()
        
        # Double-check after handle_agent_output_types in case it didn't convert PIL Images
        try:
            from PIL import Image
            if isinstance(final_answer, Image.Image):
                # Save PIL Image to temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                final_answer.save(temp_file.name, 'PNG')
                temp_file.close()
                # Convert to AgentImage
                final_answer = AgentImage(temp_file.name)
        except ImportError:
            pass  # PIL not available
        except Exception:
            pass  # Not a PIL Image or conversion failed

    print(f"[DEBUG Step 10] Final answer type: {type(final_answer)}")
    print(f"[DEBUG Step 10] Final answer: {final_answer}")
    
    if isinstance(final_answer, AgentText):
        print(f"[DEBUG Step 10] ✅ Rendering as AgentText")
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        img_path = final_answer.to_string()
        print(f"[DEBUG Step 10] ✅ Rendering as AgentImage")
        print(f"[DEBUG Step 10] Image path: {img_path}")
        print(f"[DEBUG Step 10] Path exists: {os.path.exists(img_path) if img_path else False}")
        if img_path and os.path.exists(img_path):
            print(f"[DEBUG Step 10] File size: {os.path.getsize(img_path)} bytes")
        yield gr.ChatMessage(
            role="assistant",
            content={"path": img_path, "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        # Check if it's a string representation of a PIL Image and try to extract path
        final_str = str(final_answer)
        if "PIL." in final_str and "Image" in final_str:
            # Try to find if there's a file path in the final answer
            # This handles cases where handle_agent_output_types didn't convert properly
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Final answer:**\n⚠️ Image generated but display failed. The image was created successfully.\n{final_str}",
            )
        else:
            yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {final_str}")


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                ),
                resizable=True,
                scale=1,
                type="messages",  # Use messages format (ChatMessage objects) instead of tuples
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

        demo.launch(debug=True, **kwargs)  # Removed share=True as it's not supported on HF Spaces


__all__ = ["stream_to_gradio", "GradioUI"]