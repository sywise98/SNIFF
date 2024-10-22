import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class ItemExtractorNode():
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    def user_input_callback(self, msg):
        user_message = msg.data
        identified_item = self.identify_item(user_message)
        print(f"The item the user is looking for is: {identified_item}")
        
    def create_prompt(self, user_query):  # Add 'self' as the first parameter
        return f"Identify the item the user is asking to find in the following query: '{user_query}'"

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_item(self, response):
        print(f"response: {response}")
        return response.split(":")[-1].strip()

    def identify_item(self, user_query):
        prompt = self.create_prompt(user_query)
        response = self.generate_response(prompt)
        item = self.extract_item(response)
        return item

def main(args=None):
    node = ItemExtractorNode()
    node.user_input_callback("Find the red cup")

if __name__ == '__main__':
    main()
