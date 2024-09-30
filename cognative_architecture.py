# Import necessary libraries for NLP, memory management, and logging
from transformers import GPT4Tokenizer, GPT4LMHeadModel
import json  # For handling long-term memory persistence
import os    # For file path and existence checks
import logging  # For logging errors and key events
import asyncio  # For handling asynchronous tasks

# Configure logging for debugging and tracking in production
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the cognitive architecture class that integrates memory and NLP models
class CognitiveArchitecture:
    def __init__(self, model_name="gpt4"):
        """
        Initialize the cognitive architecture with short-term and long-term memory and an NLP model.
        """
        self.short_term_memory = ShortTermMemory(max_cache_size=50)  # Cache limit for memory
        self.long_term_memory = LongTermMemory("long_term_memory.json")
        self.nlp_model = NLPModel(model_name)
        logging.info("CognitiveArchitecture initialized.")

    async def reason(self, input_text):
        """
        Main reasoning method: attempts to retrieve an answer from memory, and if unavailable,
        generates a response using the NLP model.
        """
        try:
            # Step 1: Check short-term memory
            context = self.short_term_memory.retrieve_context(input_text)
            if context:
                return f"Retrieved from short-term memory: {context}"

            # Step 2: Check long-term memory
            context = self.long_term_memory.retrieve_experience(input_text)
            if context:
                self.short_term_memory.store_context(input_text, context)
                return f"Retrieved from long-term memory: {context}"

            # Step 3: Use GPT-4 to generate response if no memory available
            response = await self.nlp_model.generate_response(input_text)
            self.short_term_memory.store_context(input_text, response)
            self.long_term_memory.store_experience(input_text, response)
            return f"Generated response: {response}"

        except Exception as e:
            logging.error(f"Error during reasoning process: {e}")
            return "An error occurred during reasoning."

# Short-Term Memory class to store recent queries with a cache limit
class ShortTermMemory:
    def __init__(self, max_cache_size=50):
        """
        Initializes short-term memory with a maximum cache size. Memory will hold recent data
        only for the current session.
        """
        self.memory = {}
        self.max_cache_size = max_cache_size

    def store_context(self, input_text, context):
        """
        Stores context in short-term memory. If the memory exceeds the cache limit, the oldest entries are removed.
        """
        if len(self.memory) >= self.max_cache_size:
            oldest_key = next(iter(self.memory))
            self.memory.pop(oldest_key)  # Remove the oldest memory to prevent overflow
        self.memory[input_text] = context
        logging.info(f"Stored context in short-term memory for '{input_text}'")

    def retrieve_context(self, input_text):
        """
        Retrieves context from short-term memory. Returns None if not found.
        """
        return self.memory.get(input_text, None)

# Long-Term Memory class with persistent storage in a JSON file
class LongTermMemory:
    def __init__(self, file_path):
        """
        Initializes long-term memory by loading data from a JSON file. If the file doesn't exist,
        a new memory file is created.
        """
        self.file_path = file_path
        self.memory = self._load_memory()

    def _load_memory(self):
        """
        Loads long-term memory from a JSON file. If the file doesn't exist, an empty memory is created.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                logging.info("Loaded long-term memory from file.")
                return json.load(file)
        logging.warning("No long-term memory file found. Initializing empty memory.")
        return {}

    def _save_memory(self):
        """
        Saves the current long-term memory to the JSON file.
        """
        with open(self.file_path, 'w') as file:
            json.dump(self.memory, file)
        logging.info("Saved long-term memory to file.")

    def store_experience(self, key, experience):
        """
        Stores a new experience in long-term memory and saves it to the file.
        """
        self.memory[key] = experience
        self._save_memory()

    def retrieve_experience(self, key):
        """
        Retrieves an experience from long-term memory.
        """
        return self.memory.get(key, None)

# NLP Model class (GPT-4 or other models can be plugged in)
class NLPModel:
    def __init__(self, model_name="gpt4"):
        """
        Initializes the NLP model using the GPT-4 pre-trained model.
        """
        self.tokenizer = GPT4Tokenizer.from_pretrained(model_name)
        self.model = GPT4LMHeadModel.from_pretrained(model_name)
        logging.info(f"NLP model {model_name} initialized.")

    async def generate_response(self, input_text):
        """
        Generates a response asynchronously using the pre-trained GPT-4 model.
        """
        try:
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            outputs = await asyncio.to_thread(self.model.generate, inputs, max_length=100, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generated response for '{input_text}'")
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise e

# Example usage of the CognitiveArchitecture system with real-world tasks
async def main():
    # Initialize the cognitive system using GPT-4
    cognitive_system = CognitiveArchitecture()

    # Example 1: Real-world Knowledge Reasoning
    print("Example 1: General Knowledge Reasoning")
    user_input = "What is the price of Bitcoin?"
    response = await cognitive_system.reason(user_input)
    print(response)

    # Example 2: Real-world Code Generation
    print("\nExample 2: Code Generation")
    code_prompt = "Write a Python function to sort a list of integers."
    response = await cognitive_system.reason(code_prompt)
    print(response)

    # Example 3: Real-world Language Translation Task
    print("\nExample 3: Translation Task")
    translation_prompt = "Translate 'Good morning, how are you?' to Spanish."
    response = await cognitive_system.reason(translation_prompt)
    print(response)

    # Example 4: Memory Retrieval after initial reasoning
    print("\nExample 4: Memory Retrieval Test")
    response_again = await cognitive_system.reason("What is the price of Bitcoin?")
    print(response_again)

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())  # Start the cognitive system and test reasoning with real-world examples
