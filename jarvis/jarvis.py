from openai import OpenAI
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.syntax import Syntax
import pyperclip
import time  # Add time import
import requests
import json

# Model Constants
DEEPSEEK_MODEL = "deepseek-reasoner"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_DEEPSEEK = "deepseek-r1:14b"  

# Load environment variables
load_dotenv()

class ModelChain:
    def __init__(self):
        # Initialize clients based on available API keys
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.has_official_deepseek = bool(self.deepseek_api_key)
        self.has_openrouter = bool(self.openrouter_api_key)
        
        if self.has_official_deepseek:
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
        
        if self.has_openrouter:
            # Initialize OpenRouter client
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
        
        # Initialize Ollama client with native API path
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/api",
            api_key="ollama",
        )
        
        self.deepseek_messages = []
        self.openrouter_messages = []
        self.ollama_messages = []
        # Default to Ollama model if no OpenRouter API key is available
        self.current_model = OLLAMA_MODEL if not self.has_openrouter else OPENROUTER_MODEL
        self.show_reasoning = True
        self.use_ollama_deepseek = not self.has_official_deepseek  # Default based on API availability
        self.ollama_base_url = "http://localhost:11434/api"

    def set_model(self, model_name):
        if not self.has_openrouter and not model_name.startswith("ollama:"):
            rprint("\n[red]Warning: OpenRouter API key not found, falling back to Ollama model[/]")
            self.current_model = OLLAMA_MODEL
        else:
            self.current_model = model_name
        self.use_ollama_deepseek = model_name.startswith("ollama:deepseek")

    def get_model_display_name(self):
        return self.current_model

    def get_deepseek_reasoning(self, user_input):
        start_time = time.time()
        
        # Use Ollama if no official API or explicitly specified
        if not self.has_official_deepseek or self.use_ollama_deepseek:
            return self.get_ollama_deepseek_reasoning(user_input)
        else:
            return self.get_official_deepseek_reasoning(user_input)

    def get_official_deepseek_reasoning(self, user_input):
        """Use official Deepseek API for reasoning"""
        self.deepseek_messages.append({"role": "user", "content": user_input})
        
        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")
        
        response = self.deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=1,
            messages=self.deepseek_messages,
            stream=True
        )

        reasoning_content = ""
        final_content = ""

        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_piece = chunk.choices[0].delta.reasoning_content
                reasoning_content += reasoning_piece
                if self.show_reasoning:
                    print(reasoning_piece, end="", flush=True)
            elif chunk.choices[0].delta.content:
                final_content += chunk.choices[0].delta.content

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")
        
        if self.show_reasoning:
            print("\n")
        return reasoning_content

    def get_ollama_deepseek_reasoning(self, user_input):
        """Use Ollama version of Deepseek for reasoning"""
        start_time = time.time()
        
        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that thinks step by step. Format your response as: <think>your step by step reasoning</think> and stop immediately after closing the think tag."},
            {"role": "user", "content": f"Analyze this request step by step:\n{user_input}"}
        ]
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/chat",
                json={
                    "model": OLLAMA_DEEPSEEK.replace("ollama:", ""),
                    "messages": messages,
                    "stream": True
                },
                stream=True
            )
            
            reasoning_content = ""
            current_chunk = ""
            has_started_think = False
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            current_chunk += content
                            
                            # Handle streaming output with tags
                            if not has_started_think and "<think>" in current_chunk:
                                has_started_think = True
                                if self.show_reasoning:
                                    print("<think>", end="", flush=True)
                                current_chunk = current_chunk[current_chunk.find("<think>") + 7:]
                                reasoning_content += "<think>"  # Add opening tag to final content
                            
                            if has_started_think:
                                if "</think>" in current_chunk:
                                    end_idx = current_chunk.find("</think>")
                                    if self.show_reasoning:
                                        print(current_chunk[:end_idx], end="", flush=True)
                                        print("</think>", end="", flush=True)
                                    reasoning_content += current_chunk[:end_idx] + "</think>"  # Add content and closing tag
                                    response.close()  # Close the stream
                                    break  # Exit the loop after finding </think>
                                else:
                                    if self.show_reasoning:
                                        print(current_chunk, end="", flush=True)
                                    reasoning_content += current_chunk
                                    current_chunk = ""
                                
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            rprint(f"\n[red]Error in streaming response: {str(e)}[/]")
            return "Error occurred while streaming response"

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")
        
        if self.show_reasoning:
            print("\n")
        return reasoning_content  # Now includes <think> tags

    def get_openrouter_response(self, user_input, reasoning):
        combined_prompt = (
            f"<question>{user_input}</question>\n\n"
            f"{reasoning}\n\n"
        )
        
        self.openrouter_messages.append({"role": "user", "content": combined_prompt})
        
        rprint(f"[green]{self.get_model_display_name()}[/]")
        
        try:
            completion = self.openrouter_client.chat.completions.create(
                model=self.current_model,
                messages=self.openrouter_messages,
                stream=True
            )
            
            full_response = ""
            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content_piece = delta.content
                        full_response += content_piece
                        print(content_piece, end="", flush=True)
                except Exception as e:
                    rprint(f"\n[red]Error processing chunk: {str(e)}[/]")
                    continue
                    
        except Exception as e:
            rprint(f"\n[red]Error in streaming response: {str(e)}[/]")
            return "Error occurred while streaming response"
        
        self.deepseek_messages.append({"role": "assistant", "content": full_response})
        self.openrouter_messages.append({"role": "assistant", "content": full_response})
        
        print("\n")
        return full_response

    def get_ollama_response(self, user_input, reasoning):
        combined_prompt = (
            f"<question>{user_input}</question>\n\n"
            f"{reasoning}\n\n"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides clear and concise answers."},
            {"role": "user", "content": combined_prompt}
        ]
        
        rprint(f"[green]{self.get_model_display_name()}[/]")
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/chat",
                json={
                    "model": self.current_model.replace("ollama:", ""),
                    "messages": messages,
                    "stream": True
                },
                stream=True
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if "message" in chunk and "content" in chunk["message"]:
                            content_piece = chunk["message"]["content"]
                            full_response += content_piece
                            print(content_piece, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            rprint(f"\n[red]Error in streaming response: {str(e)}[/]")
            return "Error occurred while streaming response"
        
        self.deepseek_messages.append({"role": "assistant", "content": full_response})
        self.ollama_messages.append({"role": "assistant", "content": full_response})
        
        print("\n")
        return full_response

    def get_response(self, user_input, reasoning):
        # Use Ollama response if no OpenRouter API or if using Ollama model
        if not self.has_openrouter or self.current_model.startswith("ollama:"):
            return self.get_ollama_response(user_input, reasoning)
        else:
            return self.get_openrouter_response(user_input, reasoning)

def main():
    chain = ModelChain()
    
    # Initialize prompt session with styling
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    rprint(Panel.fit(
        "[bold cyan]Retrival augmented thinking[/]",
        title="[bold cyan]JARVIS 🧠[/]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <name>'[/] to change the model")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'clear'[/] to clear chat history")
    if not chain.has_openrouter:
        rprint(" • Using Ollama models (no OpenRouter API key found)")
    else:
        rprint(" • For Ollama models, use [bold magenta]'model ollama:<model_name>'[/]")
    if not chain.has_official_deepseek:
        rprint(" • Using Ollama Deepseek (no official API key found)")
    rprint("\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'clear':
                chain.deepseek_messages = []
                chain.openrouter_messages = []
                chain.ollama_messages = []
                rprint("\n[magenta]Chat history cleared![/]\n")
                continue
                
            if user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                chain.set_model(new_model)
                print(f"\nChanged model to: {chain.get_model_display_name()}\n")
                continue

            if user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning process is now {status}[/]\n")
                continue
            
            reasoning = chain.get_deepseek_reasoning(user_input)
            response = chain.get_response(user_input, reasoning)
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()