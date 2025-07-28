import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()
import tiktoken
import json
from datetime import datetime

# set defaults
DEFAULT_API_KEY = os.environ.get("OPEN_API_KEY")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_SYSTEM_MESSAGE = "You are a sassy assistant who is fed up with answering questions."
DEFAULT_TOKEN_BUDGET = 4096

# ConversationManager class
class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, history_file=None, temperature=None, max_tokens=None, token_budget=None, system_message=None):
        if not api_key:
            api_key=DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
        
        self.client = OpenAI(api_key=api_key) 
        self.client.base_url = base_url
        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file
        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.system_message = system_message if system_message else DEFAULT_SYSTEM_MESSAGE
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.system_messages = {
            "sassy_assistant": "A sassy assistant who is fed up with answering questions.",
            "angry_assistant": "An angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "A thoughtful assistant, always ready to dig deeper. This assistant asks clarifying questions to ensure understanding and approaches problems with a step-by-step methodology.",
            "expert_assistant": "An assistant that is an expert in the topic being discussed. This assistant provides answers with academic language and indicates what sources information came from."
        }
        #Set default persona
        self.system_message = self.system_messages["sassy_assistant"]
        self.load_conversation_history()
    
    #count number of tokens in the text
    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        return len(tokens)

    #track total number of tokens used in the conversation
    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)
    
    # ensures that total token count of conversation_history does not exceed the token_budget
    def enforce_token_budget(self):
        # Remove messages until the token budget is no longer exceeded
        while self.total_tokens_used() > self.token_budget:
            # Break the loop if only the system message is left
            if len(self.conversation_history) <= 1:
                break
            # Remove the first message after the system message
            self.conversation_history.pop(1)
    
    #change chatbot's persona by updating system message
    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")
    
    #select custom persona
    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')
    
    #update message in conversation history
    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content": self.system_message})
    
    # chat parameters        
    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        self.conversation_history.append({"role": "user", "content": prompt})
        
        self.enforce_token_budget()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            return None
        
        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        self.save_conversation_history()

        return ai_response

    # manage conversation history
    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f"An I/O error occurred while saving the conversation history: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the conversation history: {e}")
            
    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()  # Attempt to save the reset history to the file
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history: {e}")

conv_manager = ConversationManager()

# #chatbot prompts
# prompt = "Tell me about the war of 1812."
# response = conv_manager.chat_completion(prompt)
# print(response)

# prompt = "Tell me a joke."
# response = conv_manager.chat_completion(prompt)
# print(response)

prompt = "What is the best hostess gift to bring to a party?."
response = conv_manager.chat_completion(prompt)
print(response)

# # Test conversation history tracking
# # List all files
# all_files_and_directories = os.listdir()
# only_files = [f for f in all_files_and_directories if os.path.isfile(f)]
# sorted_files = sorted(only_files)
# for file in sorted_files:
#     print(file)

# new_conv_manager = ConversationManager()

# # Simulate chat completion
# new_conv_manager.chat_completion("Tell me about the types of native birds in Panama.")

# # List all files
# all_files_and_directories = os.listdir()
# only_files = [f for f in all_files_and_directories if os.path.isfile(f)]
# sorted_files = sorted(only_files)
# for file in sorted_files:
#     print(file)
    
# # Verify conversation histories are appended
# print("\nconv_manager:\n", conv_manager.conversation_history)
# print("\nnew_conv_manager:\n", new_conv_manager.conversation_history)

# # Change to a custom persona
# conv_manager.set_custom_system_message("You are a frantic assistant that is always late to your next meeting and forgetting things.")
# response = conv_manager.chat_completion(prompt)
# print("Custom Persona AI Response:", response)

# # test conversation history tracking
# print("Conversation history:")
# for message in conv_manager.conversation_history:
#     print(f'{message["role"].title()}: {message["content"]}') 

# # test count_tokens method
# print("Tokens in the last response:", conv_manager.count_tokens(response))

# # test total_tokens_used method
# print("Total tokens used so far:", conv_manager.total_tokens_used())