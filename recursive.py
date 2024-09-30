# Import necessary libraries
from transformers import GPT4Tokenizer, GPT4LMHeadModel  # GPT-4 model for text generation
import logging  # For logging system activity and errors
import asyncio  # For asynchronous processing
import os  # For file handling in long-term memory
import json  # For saving long-term memory in JSON format

# Configure logging for production-level tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Feedback System Module
class FeedbackSystem:
    # The FeedbackSystem handles the process of requesting and receiving feedback from the user.
    # Feedback is critical for guiding the system's learning process.

    @staticmethod
    def request_feedback():
        # Requests feedback from the user.
        # In a real-world system, this could be a user input, survey, or automated feedback.
        feedback = input("Was the response satisfactory? (yes/no): ").strip().lower()
        # Returns "positive" or "negative" based on user input
        return "positive" if feedback == "yes" else "negative"

# Recursive Learning Service
class RecursiveLearningService:
    # The RecursiveLearningService manages the learning process after receiving feedback.
    # It interacts with memory and the text agent to reprocess inputs and improve responses.

    def __init__(self, text_agent, short_term_memory, long_term_memory):
        # Initializes the RecursiveLearningService with a text agent and memory systems.
        # The text agent is used for reprocessing inputs, and the memory stores results.
        self.text_agent = text_agent
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

    async def learn_from_feedback(self, input_text, initial_response):
        # Handles the recursive learning process based on feedback.
        # If feedback is negative, the input is reprocessed to improve the response.

        feedback = FeedbackSystem.request_feedback()  # Request user feedback

        if feedback == "negative":
            # If the feedback is negative, try to improve the response using recursive learning.
            logging.info(f"Negative feedback received for: {input_text}")
            improved_response = await self.recursive_learn(input_text)  # Reprocess input
            print(f"Improved Response: {improved_response}")
        else:
            # If the feedback is positive, store the response in both short-term and long-term memory.
            logging.info(f"Positive feedback received for: {input_text}")
            self.short_term_memory.store_context(input_text, initial_response)  # Store in short-term memory
            self.long_term_memory.store_experience(input_text, initial_response)  # Store in long-term memory

    async def recursive_learn(self, input_text):
        # Reprocesses the input to generate an improved response during recursive learning.
        # This function is called when negative feedback is received.
        logging.info(f"Attempting recursive learning for: {input_text}")
        improved_response = await self.text_agent.process(input_text)  # Reprocess the input via text agent
        return improved_response

# Memory Classes (Short-Term and Long-Term)
class ShortTermMemory:
    # ShortTermMemory stores recent responses for the current session.
    # It ensures that frequently repeated inputs don't require reprocessing every time.

    def __init__(self, max_cache_size=50):
        # Initializes short-term memory with a specified maximum cache size.
        # This limits memory growth and avoids excessive memory usage.
        self.memory = {}  # Dictionary to store input-response pairs
        self.max_cache_size = max_cache_size  # Maximum number of entries

    def store_context(self, input_text, context):
        # Stores an input-response pair in short-term memory.
        # If the memory exceeds the cache limit, the oldest entry is removed.

        if len(self.memory) >= self.max_cache_size:
            oldest_key = next(iter(self.memory))  # Identify the oldest key
            self.memory.pop(oldest_key)  # Remove the oldest entry from memory

        self.memory[input_text] = context  # Store the new input-response pair
        logging.info(f"Stored context in short-term memory for '{input_text}'")

    def retrieve_context(self, input_text):
        # Retrieves a stored response from short-term memory based on input text.
        # If the input is not found in memory, returns None.
        return self.memory.get(input_text, None)

class LongTermMemory:
    # LongTermMemory stores input-response pairs persistently across sessions.
    # The data is stored in a JSON file, making it persistent even after a system restart.

    def __init__(self, file_path):
        # Initializes long-term memory by loading data from a specified JSON file.
        # If the file does not exist, an empty memory dictionary is created.
        self.file_path = file_path  # Path to the JSON file for long-term memory
        self.memory = self._load_memory()  # Load existing memory or initialize a new one

    def _load_memory(self):
        # Loads long-term memory from a JSON file. If the file does not exist, returns an empty dictionary.
        if os.path.exists(self.file_path):  # Check if the memory file exists
            with open(self.file_path, 'r') as file:
                logging.info("Loaded long-term memory from file.")
                return json.load(file)  # Load memory from JSON file
        logging.warning("No long-term memory file found. Initializing empty memory.")
        return {}  # Return empty memory if file doesn't exist

    def _save_memory(self):
        # Saves the current state of long-term memory to a JSON file.
        with open(self.file_path, 'w') as file:
            json.dump(self.memory, file)  # Write memory dictionary to JSON file
        logging.info("Saved long-term memory to file.")

    def store_experience(self, key, experience):
        # Stores an input-response pair in long-term memory and saves it to the file.
        self.memory[key] = experience  # Store the input-response pair
        self._save_memory()  # Save memory to file after update

    def retrieve_experience(self, key):
        # Retrieves an input-response pair from long-term memory based on the input key.
        # Returns None if the input does not exist in long-term memory.
        return self.memory.get(key, None)

# Text Agent for Processing Text Inputs
class TextAgent:
    # The TextAgent handles natural language processing tasks using GPT-4.
    # It processes input text and generates responses by interacting with a pre-trained model.

    def __init__(self, model_name="gpt4"):
        # Initializes the TextAgent by loading the GPT-4 model and tokenizer from Hugging Face.
        self.tokenizer = GPT4Tokenizer.from_pretrained(model_name)  # Load the tokenizer
        self.model = GPT4LMHeadModel.from_pretrained(model_name)  # Load the GPT-4 model
        logging.info(f"TextAgent with model {model_name} initialized.")

    async def process(self, input_text):
        # Processes input text using GPT-4 to generate a response asynchronously.
        # This prevents blocking and ensures smooth execution.
        try:
            logging.debug(f"Processing text input: {input_text}")
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")  # Tokenize the input text
            outputs = await asyncio.to_thread(self.model.generate, inputs, max_length=100, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output
            logging.info(f"Generated text response: {response}")
            return response
        except Exception as e:
            # Logs any errors encountered during text processing and returns a fallback error message.
            logging.error(f"Error in TextAgent: {e}")
            return f"Error in processing text input: {e}"

# Modular Cognitive Architecture
class CognitiveArchitecture:
    # The CognitiveArchitecture class orchestrates the system by integrating recursive learning, 
    # feedback, and memory systems. It manages the flow of input processing and feedback-based learning.

    def __init__(self, model_name="gpt4"):
        # Initializes the cognitive architecture with recursive learning and feedback systems.
        # It uses the TextAgent for text processing and memory systems for storing input-response pairs.
        self.short_term_memory = ShortTermMemory(max_cache_size=50)  # Initialize short-term memory
        self.long_term_memory = LongTermMemory("long_term_memory.json")  # Initialize long-term memory
        self.text_agent = TextAgent(model_name)  # Initialize the text agent with GPT-4
        self.recursive_learning_service = RecursiveLearningService(self.text_agent, self.short_term_memory, self.long_term_memory)
        logging.info("CognitiveArchitecture with recursive learning and feedback system initialized.")

    async def process_input(self, input_text):
        # Processes an input text by generating a response and then handling feedback-based recursive learning.
        # The feedback system triggers additional processing if the response is unsatisfactory.

        # Step 1: Generate the initial response using the text agent
        response = await self.text_agent.process(input_text)
        print(f"Response: {response}")

        # Step 2: Trigger the recursive learning service based on feedback
        await self.recursive_learning_service.learn_from_feedback(input_text, response)

# Example usage of the system with modular integration of feedback and learning
async def main():
    # Initialize the cognitive system with feedback loops and recursive learning
    cognitive_system = CognitiveArchitecture()

    # Example 1: Process input with feedback and learning
    input_text = "What is the capital of France?"
    await cognitive_system.process_input(input_text)

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())  # Start the cognitive system with feedback loop and recursive learning
